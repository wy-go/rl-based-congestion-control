"""Link module for network simulation."""

import random
from typing import List, Dict, Optional, Deque, TYPE_CHECKING
from collections import deque

# Avoid circular imports
if TYPE_CHECKING:
    from cc_simulator.network_simulator.packet import Packet
    from cc_simulator.utils.trace import Trace

from cc_simulator.utils.constants import BYTES_PER_PACKET, BITS_PER_BYTE


class Link:
    """Network link implementation for simulator.
    
    Simulates network links with bandwidth, latency, and queuing.
    """
    
    def __init__(self, trace: 'Trace'):
        """Initialize link with trace.
        
        Args:
            trace: Trace object containing network conditions
        """
        self.trace = trace
        self.queue: Deque['Packet'] = deque()
        self.current_buffer_size = 0
        
    def reset(self) -> None:
        """Reset link state."""
        self.queue.clear()
        self.current_buffer_size = 0
    
    def get_cur_queue_delay(self, ts: float) -> float:
        """Calculate the current queuing delay.
        
        Args:
            ts: Current timestamp
            
        Returns:
            Current queuing delay in seconds
        """
        bw = self.trace.get_bandwidth(ts)  # Mbps
        return self.current_buffer_size * BYTES_PER_PACKET * 8 / (bw * 1e6)
    
    def enqueue(self, pkt: 'Packet', ts: float) -> bool:
        """Enqueue a packet, subject to queue size limit.
        
        Args:
            pkt: Packet to enqueue
            ts: Current timestamp
            
        Returns:
            Whether packet was successfully enqueued
        """
        # Check for random loss
        if random.random() < self.trace.get_loss_rate(ts):
            pkt.dropped = True
            return False
            
        # Check queue limit
        queue_size_limit = self.trace.get_queue_size(ts)
        if self.current_buffer_size >= queue_size_limit:
            pkt.dropped = True
            return False
            
        # Add to queue
        self.queue.append(pkt)
        self.current_buffer_size += 1
        return True
    
    def dequeue(self) -> Optional['Packet']:
        """Dequeue a packet if available.
        
        Returns:
            Dequeued packet or None if queue is empty
        """
        if not self.queue:
            return None
            
        pkt = self.queue.popleft()
        self.current_buffer_size -= 1
        return pkt
    
    def get_propagation_delay(self, ts: float) -> float:
        """Get current propagation delay.
        
        Args:
            ts: Current timestamp
            
        Returns:
            One-way propagation delay in seconds
        """
        return self.trace.get_delay(ts) / 1000  # convert ms to seconds
    
    def get_transmission_delay(self, ts: float) -> float:
        """Calculate transmission delay for one packet.
        
        Args:
            ts: Current timestamp
            
        Returns:
            Transmission delay in seconds
        """
        bw = self.trace.get_bandwidth(ts)  # Mbps
        return BYTES_PER_PACKET * 8 / (bw * 1e6)  # seconds
    
    def get_next_transmission_time(self, ts: float) -> float:
        """Calculate the next time a packet can be transmitted.
        
        Args:
            ts: Current timestamp
            
        Returns:
            Next transmission time
        """
        return ts + self.get_transmission_delay(ts) 