"""Trace module for network simulations."""

import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np


class Trace:
    """Network trace for simulation.
    
    This class represents a time series of network conditions including
    bandwidth, latency, and queue size.
    """
    
    def __init__(
        self, 
        timestamps: List[float], 
        bandwidths: List[float], 
        latencies: List[float], 
        queue_sizes: List[int],
        loss_rates: Optional[List[float]] = None,
        name: str = "unnamed_trace"
    ):
        """Initialize a trace object.
        
        Args:
            timestamps: List of timestamps in seconds
            bandwidths: List of bandwidths in Mbps
            latencies: List of one-way propagation delays in ms
            queue_sizes: List of queue sizes in packets
            loss_rates: List of random loss rates (0-1)
            name: Name of the trace
        """
        assert len(timestamps) == len(bandwidths) == len(latencies) == len(queue_sizes), \
            "All trace lists must have the same length"
        
        self.timestamps = timestamps
        self.bandwidths = bandwidths  # Mbps
        self.latencies = latencies    # ms (one-way delay)
        self.queue_sizes = queue_sizes  # packets
        self.loss_rates = loss_rates if loss_rates is not None else [0.0] * len(timestamps)
        self.name = name
        self.pointer = 0
        
        # Calculate averages
        self.avg_bw = np.mean(bandwidths)
        self.avg_delay = np.mean(latencies)
        
    def reset(self) -> None:
        """Reset trace pointer to the beginning."""
        self.pointer = 0
    
    def get_bandwidth(self, time: float) -> float:
        """Get bandwidth at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Bandwidth in Mbps
        """
        idx = self._get_index_at(time)
        return self.bandwidths[idx]
    
    def get_delay(self, time: float) -> float:
        """Get one-way delay at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            One-way delay in ms
        """
        idx = self._get_index_at(time)
        return self.latencies[idx]
    
    def get_queue_size(self, time: float) -> int:
        """Get queue size at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Queue size in packets
        """
        idx = self._get_index_at(time)
        return self.queue_sizes[idx]
    
    def get_loss_rate(self, time: float) -> float:
        """Get random loss rate at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Loss rate (0-1)
        """
        idx = self._get_index_at(time)
        return self.loss_rates[idx]
    
    def is_finished(self, time: float) -> bool:
        """Check if trace has finished at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Whether the trace has finished
        """
        return time >= self.timestamps[-1]
    
    def _get_index_at(self, time: float) -> int:
        """Get index of trace at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Index in the trace lists
        """
        # Handle boundary conditions
        if time <= self.timestamps[0]:
            return 0
        if time >= self.timestamps[-1]:
            return len(self.timestamps) - 1
            
        # Optimization: start from current pointer to avoid linear search
        if self.pointer < len(self.timestamps) - 1 and time >= self.timestamps[self.pointer]:
            # Search forward
            while (self.pointer < len(self.timestamps) - 1 and 
                   time >= self.timestamps[self.pointer + 1]):
                self.pointer += 1
            return self.pointer
        else:
            # Search backward
            while self.pointer > 0 and time < self.timestamps[self.pointer]:
                self.pointer -= 1
            return self.pointer
    
    def dump(self, filename: str) -> None:
        """Save trace to a JSON file.
        
        Args:
            filename: Output filename
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump({
                'name': self.name,
                'timestamps': self.timestamps,
                'bandwidths': self.bandwidths,
                'latencies': self.latencies,
                'queue_sizes': self.queue_sizes,
                'loss_rates': self.loss_rates,
            }, f)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'Trace':
        """Load trace from a JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded trace object
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls(
            timestamps=data['timestamps'],
            bandwidths=data['bandwidths'],
            latencies=data['latencies'],
            queue_sizes=data['queue_sizes'],
            loss_rates=data.get('loss_rates'),
            name=data.get('name', os.path.basename(filename))
        )


def generate_constant_trace(
    duration: float = 30.0,
    bandwidth: float = 10.0,  # Mbps
    latency: float = 20.0,    # ms
    queue_size: int = 5,      # packets
    loss_rate: float = 0.0,   # 0-1
    step_size: float = 0.1,   # seconds
    name: str = "constant_trace"
) -> Trace:
    """Generate a trace with constant values.
    
    Args:
        duration: Duration in seconds
        bandwidth: Constant bandwidth in Mbps
        latency: Constant one-way delay in ms
        queue_size: Constant queue size in packets
        loss_rate: Constant random loss rate (0-1)
        step_size: Time step size in seconds
        name: Name of the trace
        
    Returns:
        Generated trace
    """
    num_points = int(duration / step_size) + 1
    timestamps = [i * step_size for i in range(num_points)]
    bandwidths = [bandwidth] * num_points
    latencies = [latency] * num_points
    queue_sizes = [queue_size] * num_points
    loss_rates = [loss_rate] * num_points
    
    return Trace(
        timestamps=timestamps,
        bandwidths=bandwidths,
        latencies=latencies,
        queue_sizes=queue_sizes,
        loss_rates=loss_rates,
        name=name
    ) 