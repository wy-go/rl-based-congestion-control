"""Network simulator for congestion control."""

import heapq
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING, Union

# Avoid circular imports
if TYPE_CHECKING:
    from cc_simulator.network_simulator.sender import Sender
    from cc_simulator.network_simulator.link import Link
    from cc_simulator.network_simulator.packet import Packet

from cc_simulator.utils.constants import (
    EVENT_TYPE_ACK, EVENT_TYPE_SEND, BYTES_PER_PACKET
)


class Network:
    """Network simulator that connects senders and links."""
    
    def __init__(
        self, 
        senders: List['Sender'], 
        links: List['Link'],
        record_pkt_log: bool = False
    ):
        """Initialize network simulator.
        
        Args:
            senders: List of senders
            links: List of links
            record_pkt_log: Whether to record packet logs
        """
        self.senders = senders
        self.links = links
        self.q = []  # Priority queue for events
        self.cur_time = 0.0
        self.pkt_log: List[List[Any]] = []
        self.record_pkt_log = record_pkt_log
        self.extra_delays: List[float] = []
        
        # Register network with senders
        for sender in senders:
            sender.register_network(self)
    
    def reset(self) -> None:
        """Reset the network state."""
        self.q = []
        self.cur_time = 0.0
        self.pkt_log = []
        self.extra_delays = []
        
        for link in self.links:
            link.reset()
            
        for sender in self.senders:
            sender.reset()
            # Start sending packets
            sender.schedule_send(first_pkt=True)
    
    def get_cur_time(self) -> float:
        """Get current simulation time.
        
        Returns:
            Current time in seconds
        """
        return self.cur_time
    
    def add_packet(self, pkt: 'Packet') -> None:
        """Add a packet event to the priority queue.
        
        Args:
            pkt: Packet to add
        """
        heapq.heappush(self.q, pkt)
    
    def run(self, duration: float) -> None:
        """Run the network simulation for a specified duration.
        
        Args:
            duration: Duration to run simulation in seconds
        """
        # Reset statistics for this run
        for sender in self.senders:
            sender.reset_obs()
            
        # Calculate end time for this run
        end_time = min(
            self.cur_time + duration, 
            self.links[0].trace.timestamps[-1]
        )
        
        # Clear extra delays for this run
        self.extra_delays = []
        
        # Process events until reaching end time
        while self.q:
            pkt = self.q[0]  # Peek at next event
            
            # Check if we should stop
            stop = False
            for sender in self.senders:
                if sender.stop_run(pkt, end_time):
                    stop = True
                    break
                    
            if stop or pkt.ts >= end_time:
                self.cur_time = end_time
                break
                
            # Process the next event
            pkt = heapq.heappop(self.q)
            self.cur_time = pkt.ts
            
            # Handle event based on type
            if pkt.event_type == EVENT_TYPE_SEND:
                self._process_send(pkt)
            elif pkt.event_type == EVENT_TYPE_ACK:
                self._process_ack(pkt)
    
    def _process_send(self, pkt: 'Packet') -> None:
        """Process a packet send event.
        
        Args:
            pkt: Packet being sent
        """
        # Update sender stats
        if not pkt.sender.on_packet_sent(pkt):
            return
            
        # Enqueue in outgoing link
        if pkt.next_hop < len(self.links):
            link = self.links[pkt.next_hop]
            
            # Log if enabled
            if self.record_pkt_log:
                self._log_packet(pkt, "send", link)
                
            # Calculate queue delay
            queue_delay = link.get_cur_queue_delay(self.cur_time)
            
            # Try to enqueue packet
            if link.enqueue(pkt, self.cur_time):
                # Calculate when this packet will be processed
                next_tx_time = link.get_next_transmission_time(self.cur_time)
                prop_delay = link.get_propagation_delay(self.cur_time)
                
                # Create ACK packet
                ack_ts = next_tx_time + prop_delay
                ack_pkt = type(pkt)(
                    ts=ack_ts,
                    sender=pkt.sender,
                    pkt_id=pkt.pkt_id
                )
                ack_pkt.event_type = EVENT_TYPE_ACK
                ack_pkt.next_hop = pkt.next_hop + 1
                ack_pkt.cur_latency = 2 * prop_delay + queue_delay
                ack_pkt.queuing_delay = queue_delay
                
                # Add to event queue
                self.add_packet(ack_pkt)
            else:
                # Packet was dropped
                pkt.sender.on_packet_lost(pkt)
                if self.record_pkt_log:
                    self._log_packet(pkt, "drop", link)
    
    def _process_ack(self, pkt: 'Packet') -> None:
        """Process a packet ACK event.
        
        Args:
            pkt: Packet being acknowledged
        """
        # Update sender stats
        pkt.sender.on_packet_acked(pkt)
        
        # Log if enabled
        if self.record_pkt_log:
            if pkt.next_hop < len(self.links):
                link = self.links[pkt.next_hop]
                self._log_packet(pkt, "ack", link)
    
    def _log_packet(self, pkt: 'Packet', event: str, link: 'Link') -> None:
        """Log packet event.
        
        Args:
            pkt: Packet being logged
            event: Event type (send, ack, drop)
            link: Link involved in the event
        """
        bandwidth = link.trace.get_bandwidth(self.cur_time)
        self.pkt_log.append([
            self.cur_time,                   # timestamp
            pkt.pkt_id,                     # packet id
            event,                          # event type
            BYTES_PER_PACKET,               # bytes
            pkt.cur_latency if pkt.cur_latency is not None else 0.0,  # latency
            pkt.queuing_delay if pkt.queuing_delay is not None else 0.0,  # queue delay
            link.current_buffer_size,       # packets in queue
            pkt.sender.pacing_rate,         # sending rate
            bandwidth                       # bandwidth
        ]) 