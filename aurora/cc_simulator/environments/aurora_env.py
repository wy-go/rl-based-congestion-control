"""Aurora environment for Gymnasium."""

from typing import Dict, List, Optional, Tuple, Union, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cc_simulator.network_simulator.link import Link
from cc_simulator.network_simulator.network import Network
from cc_simulator.network_simulator.aurora_sender import AuroraSender
from cc_simulator.utils.constants import BYTES_PER_PACKET
from cc_simulator.utils.trace import Trace


class TraceScheduler:
    """Simple trace scheduler for training and testing."""
    
    def __init__(self, traces: List[Trace]):
        """Initialize scheduler with traces.
        
        Args:
            traces: List of network traces
        """
        self.traces = traces
        self.idx = 0
        
    def get_trace(self) -> Trace:
        """Get next trace.
        
        Returns:
            Selected trace
        """
        trace = self.traces[self.idx]
        self.idx = (self.idx + 1) % len(self.traces)
        return trace


class AuroraEnvironment(gym.Env):
    """Gymnasium environment for Aurora congestion control."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self, 
        traces: List[Trace],
        history_len: int = 10,
        features: List[str] = ["sent latency inflation", "latency ratio", "recv ratio"],
        record_pkt_log: bool = False,
        render_mode: Optional[str] = None
    ):
        """Initialize Aurora environment.
        
        Args:
            traces: List of network traces
            history_len: Length of history to maintain
            features: List of features to track
            record_pkt_log: Whether to record packet logs
            render_mode: Render mode
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.record_pkt_log = record_pkt_log
        self.trace_scheduler = TraceScheduler(traces)
        self.current_trace = self.trace_scheduler.get_trace()
        
        self.history_len = history_len
        self.features = features
        
        # Define action space: delta rate adjustment
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # Define observation space based on features
        # This is a conservative approach; actual limits would need tuning
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(len(features) * history_len,),
            dtype=np.float32
        )
        
        # Create network components
        self._create_network()
        
        # Stats tracking
        self.reward_sum = 0.0
        self.reward_ewma = 0.0
        self.episodes_run = 0
        
    def _create_network(self) -> None:
        """Create network components."""
        self.links = [Link(self.current_trace), Link(self.current_trace)]
        self.senders = [AuroraSender(
            10 * BYTES_PER_PACKET / (self.current_trace.get_delay(0) * 2/1000),
            self.features, 
            self.history_len, 
            0, 
            0, 
            self.current_trace
        )]
        self.net = Network(self.senders, self.links, self.record_pkt_log)
        self.run_dur = 0.01  # Initial duration
        
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: RNG seed
            options: Additional options
            
        Returns:
            Initial observation and info
        """
        super().reset(seed=seed)
        
        # Reset network components
        self.net.reset()
        self.current_trace = self.trace_scheduler.get_trace()
        self.current_trace.reset()
        
        # Reset runtime parameters
        self.run_dur = 0.01
        
        # Re-create network with new trace
        self._create_network()
        
        # Run initial monitoring interval
        self.senders[0].on_mi_start()
        self.net.run(self.run_dur)
        _, run_dur = self.senders[0].on_mi_finish()
        if run_dur != 0:
            self.run_dur = run_dur
            
        # Update episode stats
        self.episodes_run += 1
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        self.reward_sum = 0.0
        
        # Get initial observation
        obs = self._get_obs()
        
        return obs, {"episode": self.episodes_run}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Apply action
        self.senders[0].apply_rate_delta(action[0])
        
        # Start new monitoring interval
        self.senders[0].on_mi_start()
        
        # Run network for monitoring interval
        self.net.run(self.run_dur)
        
        # Finish monitoring interval
        reward, run_dur = self.senders[0].on_mi_finish()
        
        # Update run duration if needed
        if run_dur != 0:
            self.run_dur = run_dur
            
        # Get observation
        obs = self._get_obs()
        
        # Check if episode has ended
        terminated = self.current_trace.is_finished(self.net.get_cur_time())
        truncated = False
        
        # Update cumulative reward
        self.reward_sum += reward
        
        # Collect info
        info = {
            "throughput": self.senders[0].history.back().get("recv rate", 0),
            "latency": self.senders[0].history.back().get("avg latency", 0),
            "loss": self.senders[0].history.back().get("loss ratio", 0),
            "send_rate": self.senders[0].pacing_rate
        }
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation array
        """
        return self.senders[0].get_obs()
    
    def render(self) -> None:
        """Render the environment."""
        if self.render_mode != "human":
            return
            
        # Print current state to console
        curr_metrics = self.senders[0].history.back()
        print(f"Time: {self.net.get_cur_time():.2f}s, "
              f"Rate: {self.senders[0].pacing_rate / BYTES_PER_PACKET:.2f} pkts/s, "
              f"Throughput: {curr_metrics.get('recv rate', 0) / 1e6:.2f} Mbps, "
              f"Latency: {curr_metrics.get('avg latency', 0) * 1000:.2f} ms, "
              f"Loss: {curr_metrics.get('loss ratio', 0) * 100:.2f}%")
    
    def close(self) -> None:
        """Clean up resources."""
        pass 