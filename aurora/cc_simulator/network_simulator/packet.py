"""Packet implementation for network simulator."""

from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING

# Avoid circular import
if TYPE_CHECKING:
    from cc_simulator.network_simulator.sender import Sender

from cc_simulator.utils.constants import EVENT_TYPE_SEND


@dataclass
class Packet:
    """Class representing a packet in the network simulator.
    
    Attributes:
        ts: Timestamp for event (send or ack)
        sender: Reference to the sender
        pkt_id: Unique packet identifier
        event_type: Type of event (send 'S' or ack 'A')
        next_hop: Next hop for this packet
        dropped: Whether this packet was dropped
        queuing_delay: Queuing delay experienced by this packet
        cur_latency: Current latency (RTT) for this packet
    """
    
    ts: float
    sender: 'Sender'
    pkt_id: int
    event_type: str = EVENT_TYPE_SEND
    next_hop: int = 0
    dropped: bool = False
    queuing_delay: float = 0.0
    cur_latency: Optional[float] = None
    
    # For advanced congestion control algorithms
    delivered: int = 0
    delivered_time: float = 0.0
    first_sent_time: float = 0.0
    is_app_limited: bool = False
    
    def __lt__(self, other: 'Packet') -> bool:
        """Comparison operator for priority queue ordering."""
        return self.ts < other.ts
    
    def debug_print(self) -> None:
        """Print packet details for debugging."""
        print(f"Event {self.pkt_id}: ts={self.ts}, type={self.event_type}, "
              f"dropped={self.dropped}, cur_latency={self.cur_latency}") 