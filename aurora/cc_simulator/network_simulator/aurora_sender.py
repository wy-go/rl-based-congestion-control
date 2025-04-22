"""Aurora sender implementation using PyTorch."""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from cc_simulator.utils.constants import (
    BYTES_PER_PACKET, BITS_PER_BYTE, MAX_RATE, MIN_RATE, MI_RTT_PROPORTION
)
from cc_simulator.network_simulator.sender import Sender
from cc_simulator.network_simulator.packet import Packet
from cc_simulator.utils.trace import Trace


class SenderHistory:
    """Class for tracking history of sender metrics."""
    
    def __init__(self, max_len: int, feature_names: List[str], sender_id: int = 0):
        """Initialize history container.
        
        Args:
            max_len: Maximum history length
            feature_names: Names of features to track
            sender_id: Sender ID
        """
        self.max_len = max_len
        self.feature_names = feature_names
        self.sender_id = sender_id
        self.history: List[Dict[str, float]] = []
        
    def step(self, metrics: Dict[str, float]) -> None:
        """Add a new set of metrics to history.
        
        Args:
            metrics: Dictionary of metric values
        """
        self.history.append(metrics)
        if len(self.history) > self.max_len:
            self.history.pop(0)
            
    def as_array(self) -> np.ndarray:
        """Convert history to numpy array for RL model input.
        
        Returns:
            Numpy array of features
        """
        result = []
        # Fill with zeros if history is not full yet
        for _ in range(self.max_len - len(self.history)):
            result.extend([0.0] * len(self.feature_names))
            
        # Add actual history
        for entry in self.history:
            for feature in self.feature_names:
                if feature in entry:
                    result.append(entry[feature])
                else:
                    result.append(0.0)
                    
        return np.array(result, dtype=np.float32)
    
    def back(self) -> Dict[str, float]:
        """Get the most recent metrics.
        
        Returns:
            Dictionary of most recent metrics
        """
        if not self.history:
            return {}
        return self.history[-1]


class AuroraSender(Sender):
    """Aurora congestion control sender implementation."""
    
    def __init__(
        self, 
        pacing_rate: float, 
        features: List[str],
        history_len: int, 
        sender_id: int, 
        dest: int, 
        trace: Trace
    ):
        """Initialize Aurora sender.
        
        Args:
            pacing_rate: Initial pacing rate in bytes/sec
            features: List of features to track
            history_len: Length of history to maintain
            sender_id: Sender ID
            dest: Destination ID
            trace: Network trace
        """
        super().__init__(sender_id, dest)
        self.starting_rate = pacing_rate
        self.pacing_rate = pacing_rate
        self.history_len = history_len
        self.features = features
        self.history = SenderHistory(history_len, features, sender_id)
        self.trace = trace
        self.obs_start_time = 0.0
        self.rtt_samples_ts: List[float] = []
        self.prev_rtt_samples: List[float] = []
        
    def on_packet_sent(self, pkt: Packet) -> bool:
        """Handle packet sent event.
        
        Args:
            pkt: Packet being sent
            
        Returns:
            Whether packet was sent successfully
        """
        success = super().on_packet_sent(pkt)
        if success:
            self.schedule_send()
        return success
    
    def on_packet_acked(self, pkt: Packet) -> None:
        """Handle packet acknowledged event.
        
        Args:
            pkt: Packet being acknowledged
        """
        super().on_packet_acked(pkt)
        self.rtt_samples_ts.append(self.get_cur_time())
        if not self.got_data:
            self.got_data = len(self.rtt_samples) >= 1
    
    def apply_rate_delta(self, delta: float) -> None:
        """Apply a rate change delta from the RL agent.
        
        Args:
            delta: Rate change multiplier (-1 to 1)
        """
        delta = float(delta)
        if delta >= 0.0:
            self.set_rate(self.pacing_rate * (1.0 + delta))
        else:
            self.set_rate(self.pacing_rate / (1.0 - delta))
    
    def set_rate(self, new_rate: float) -> None:
        """Set the pacing rate.
        
        Args:
            new_rate: New pacing rate in bytes/sec
        """
        self.pacing_rate = new_rate
        if self.pacing_rate > MAX_RATE * BYTES_PER_PACKET:
            self.pacing_rate = MAX_RATE * BYTES_PER_PACKET
        if self.pacing_rate < MIN_RATE * BYTES_PER_PACKET:
            self.pacing_rate = MIN_RATE * BYTES_PER_PACKET
    
    def record_run(self) -> None:
        """Record current run data to history."""
        metrics = self.get_run_data()
        self.history.step(metrics)
    
    def get_obs(self) -> np.ndarray:
        """Get observation for RL agent.
        
        Returns:
            Numpy array of observations
        """
        return self.history.as_array()
    
    def get_run_data(self) -> Dict[str, float]:
        """Get data for the current monitoring interval.
        
        Returns:
            Dictionary of metrics
        """
        obs_end_time = self.get_cur_time()
        
        # Handle case with no RTT samples
        if not self.rtt_samples and self.prev_rtt_samples:
            rtt_samples = [np.mean(self.prev_rtt_samples)]
        else:
            rtt_samples = self.rtt_samples
        
        # Determine receive time bounds
        recv_start = self.history.back().get("recv_end", 0) if self.history.history else self.obs_start_time
        recv_end = self.rtt_samples_ts[-1] if self.rtt_samples else obs_end_time
        
        # Calculate bytes acknowledged
        bytes_acked = self.acked * BYTES_PER_PACKET
        
        # Calculate throughput and other metrics
        send_duration = obs_end_time - self.obs_start_time
        recv_duration = recv_end - recv_start if recv_end > recv_start else send_duration
        
        send_rate = self.bytes_sent / send_duration if send_duration > 0 else 0
        recv_rate = bytes_acked / recv_duration if recv_duration > 0 else 0
        
        # Convert to bits/sec
        send_rate_bps = send_rate * BITS_PER_BYTE
        recv_rate_bps = recv_rate * BITS_PER_BYTE
        
        # Calculate latency metrics
        avg_latency = np.mean(rtt_samples) if rtt_samples else 0
        min_latency = self.min_latency if self.min_latency is not None else avg_latency
        
        # Calculate normalized metrics
        latency_inflation = avg_latency / min_latency if min_latency > 0 else 1.0
        latency_ratio = avg_latency / self.trace.avg_delay if self.trace.avg_delay > 0 else 1.0
        
        # Calculate send/recv ratios
        link_capacity_bps = self.trace.avg_bw * 1e6  # Convert Mbps to bps
        send_ratio = send_rate_bps / link_capacity_bps if link_capacity_bps > 0 else 1.0
        recv_ratio = recv_rate_bps / link_capacity_bps if link_capacity_bps > 0 else 1.0
        
        # Loss rate
        loss_ratio = self.lost / max(1, self.sent)
        
        return {
            "send_start": self.obs_start_time,
            "send_end": obs_end_time,
            "recv_start": recv_start,
            "recv_end": recv_end,
            "bytes_sent": self.bytes_sent,
            "bytes_acked": bytes_acked,
            "bytes_lost": self.bytes_lost,
            "send rate": send_rate_bps,
            "recv rate": recv_rate_bps,
            "avg latency": avg_latency,
            "min latency": min_latency,
            "loss ratio": loss_ratio,
            "sent latency inflation": latency_inflation,
            "latency ratio": latency_ratio,
            "send ratio": send_ratio,
            "recv ratio": recv_ratio,
            "avg queue delay": np.mean(self.queue_delay_samples) if self.queue_delay_samples else 0,
        }
    
    def schedule_send(self, first_pkt: bool = False, on_ack: bool = False) -> None:
        """Schedule sending next packet.
        
        Args:
            first_pkt: Whether this is the first packet
            on_ack: Whether being called in response to ACK
        """
        assert self.net is not None, "Network not registered with sender"
        
        if first_pkt:
            next_send_time = 0
        else:
            next_send_time = self.get_cur_time() + BYTES_PER_PACKET / self.pacing_rate
            
        next_pkt = Packet(next_send_time, self, len(self.net.q))
        self.net.add_packet(next_pkt)
    
    def on_mi_start(self) -> None:
        """Start a new monitoring interval."""
        self.reset_obs()
    
    def on_mi_finish(self) -> Tuple[float, float]:
        """Finish a monitoring interval.
        
        Returns:
            Tuple of (reward, duration)
        """
        self.record_run()
        
        # Calculate reward
        sender_mi = self.history.back()
        throughput = sender_mi.get("recv rate", 0) / BITS_PER_BYTE / BYTES_PER_PACKET  # packets/sec
        latency = sender_mi.get("avg latency", 0)
        loss = sender_mi.get("loss ratio", 0)
        
        # Calculate reward (this would typically call pcc_aurora_reward)
        # For now just return throughput-based reward
        reward = np.log(1.0 + throughput) - 0.5 * latency - 0.01 * loss
        
        # Calculate duration for next MI
        if latency > 0.0:
            mi_duration = MI_RTT_PROPORTION * latency + np.mean(self.net.extra_delays)
        else:
            mi_duration = 0.01  # Default small duration
            
        return reward, mi_duration
    
    def reset_obs(self) -> None:
        """Reset observations for new monitoring interval."""
        self.sent = 0
        self.acked = 0
        self.lost = 0
        
        # Save previous RTT samples in case we get none this interval
        if self.rtt_samples:
            self.prev_rtt_samples = self.rtt_samples
            
        self.rtt_samples = []
        self.rtt_samples_ts = []
        self.queue_delay_samples = []
        self.obs_start_time = self.get_cur_time()
    
    def reset(self) -> None:
        """Reset sender state."""
        super().reset()
        self.pacing_rate = self.starting_rate
        self.history = SenderHistory(self.history_len, self.features, self.sender_id)
        self.prev_rtt_samples = []
        self.got_data = False 