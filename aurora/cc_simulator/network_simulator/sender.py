"""Base sender implementations for congestion control algorithms."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

# Import from TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from cc_simulator.network_simulator.packet import Packet
    from cc_simulator.network_simulator.network import Network
    
from cc_simulator.utils.constants import (
    BYTES_PER_PACKET, EVENT_TYPE_SEND, TCP_INIT_CWND
)


class Sender(ABC):
    """Abstract base class for congestion control senders."""
    
    def __init__(self, sender_id: int, dest: int):
        """Initialize a sender with default values.
        
        Args:
            sender_id: Unique identifier for this sender
            dest: Destination ID
        """
        # Basic sender attributes
        self.sender_id = sender_id
        self.dest = dest
        self.net: Optional['Network'] = None
        self.cwnd = TCP_INIT_CWND
        self.pacing_rate = 0.0
        self.bytes_in_flight = 0
        
        # Stats tracking
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_sent = 0
        self.bytes_acked = 0
        self.bytes_lost = 0
        self.rtt_samples: List[float] = []
        self.queue_delay_samples: List[float] = []
        self.first_sent_ts: Optional[float] = None
        self.last_sent_ts: Optional[float] = None
        self.first_ack_ts: Optional[float] = None
        self.last_ack_ts: Optional[float] = None
        self.min_latency: Optional[float] = None
        self.tot_sent = 0
        self.tot_acked = 0
        self.cur_avg_latency = 0.0
        self.got_data = False
        
    def register_network(self, net: 'Network') -> None:
        """Register the network this sender belongs to.
        
        Args:
            net: Network instance
        """
        self.net = net
    
    def get_cur_time(self) -> float:
        """Get current simulation time.
        
        Returns:
            Current simulation time
        """
        assert self.net is not None, "Network not registered with sender"
        return self.net.cur_time
    
    def reset(self) -> None:
        """Reset sender state."""
        self.bytes_in_flight = 0
        self.min_latency = None
        
        # Stats reset
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_sent = 0
        self.bytes_acked = 0
        self.bytes_lost = 0
        self.rtt_samples = []
        self.queue_delay_samples = []
        self.first_sent_ts = None
        self.last_sent_ts = None
        self.first_ack_ts = None
        self.last_ack_ts = None
        self.tot_sent = 0
        self.tot_acked = 0
        self.cur_avg_latency = 0.0
        self.got_data = False
    
    def on_packet_sent(self, pkt: 'Packet') -> bool:
        """Handle packet sent event.
        
        Args:
            pkt: The packet being sent
            
        Returns:
            Whether the packet was sent successfully
        """
        # Update stats
        self.sent += 1
        self.bytes_sent += BYTES_PER_PACKET
        self.bytes_in_flight += BYTES_PER_PACKET
        self.tot_sent += 1
        
        # Update timestamps
        if self.first_sent_ts is None:
            self.first_sent_ts = pkt.ts
        self.last_sent_ts = pkt.ts
        
        return True
    
    def on_packet_acked(self, pkt: 'Packet') -> None:
        """Handle packet acknowledged event.
        
        Args:
            pkt: The packet being acknowledged
        """
        # Update stats
        self.acked += 1
        self.bytes_acked += BYTES_PER_PACKET
        self.bytes_in_flight -= BYTES_PER_PACKET
        self.tot_acked += 1
        
        # Update timestamps
        if self.first_ack_ts is None:
            self.first_ack_ts = pkt.ts
        self.last_ack_ts = pkt.ts
        
        # Update latency tracking
        assert pkt.cur_latency is not None, "Packet has no latency information"
        self.rtt_samples.append(pkt.cur_latency)
        
        if pkt.queuing_delay is not None:
            self.queue_delay_samples.append(pkt.queuing_delay)
        
        # Update minimum latency
        if self.min_latency is None or pkt.cur_latency < self.min_latency:
            self.min_latency = pkt.cur_latency
            
        # Calculate current average latency
        if len(self.rtt_samples) > 0:
            self.cur_avg_latency = sum(self.rtt_samples) / len(self.rtt_samples)
    
    def on_packet_lost(self, pkt: 'Packet') -> None:
        """Handle packet loss event.
        
        Args:
            pkt: The packet that was lost
        """
        self.lost += 1
        self.bytes_lost += BYTES_PER_PACKET
        self.bytes_in_flight -= BYTES_PER_PACKET
    
    def stop_run(self, pkt: 'Packet', end_time: float) -> bool:
        """Determine if the simulation should stop.
        
        Args:
            pkt: Current packet
            end_time: End time of simulation
            
        Returns:
            Whether to stop the simulation
        """
        return pkt.event_type == EVENT_TYPE_SEND and pkt.ts >= end_time
    
    @abstractmethod
    def schedule_send(self, first_pkt: bool = False) -> None:
        """Schedule sending a packet.
        
        Args:
            first_pkt: Whether this is the first packet
        """
        pass 