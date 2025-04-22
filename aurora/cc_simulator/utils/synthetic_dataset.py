"""Synthetic dataset utilities for congestion control."""

import glob
import json
import os
from typing import List, Dict, Any, Optional, Union

import numpy as np

from cc_simulator.utils.trace import Trace, generate_constant_trace


class SyntheticDataset:
    """Synthetic dataset for congestion control evaluation."""
    
    def __init__(
        self,
        count: int,
        config_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        """Initialize synthetic dataset.
        
        Args:
            count: Number of traces
            config_file: Path to trace generation config file
            config: Trace generation config dictionary
            seed: Random seed
        """
        np.random.seed(seed)
        self.count = count
        self.traces: List[Trace] = []
        self.config_file = config_file
        self.config = config
        
        # Generate traces if config is provided
        if self.config_file:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                
        if self.config:
            self._generate_traces()
    
    def _generate_traces(self) -> None:
        """Generate traces based on configuration."""
        for i in range(self.count):
            # Use configuration to generate random trace parameters
            if 'bandwidth' in self.config:
                bw_range = self.config['bandwidth']
                bandwidth = np.random.uniform(bw_range[0], bw_range[1])
            else:
                bandwidth = 10.0  # Default: 10 Mbps
                
            if 'latency' in self.config:
                lat_range = self.config['latency']
                latency = np.random.uniform(lat_range[0], lat_range[1])
            else:
                latency = 20.0  # Default: 20 ms
                
            if 'queue_size' in self.config:
                queue_range = self.config['queue_size']
                queue_size = np.random.randint(queue_range[0], queue_range[1] + 1)
            else:
                queue_size = 5  # Default: 5 packets
                
            if 'loss_rate' in self.config:
                loss_range = self.config['loss_rate']
                loss_rate = np.random.uniform(loss_range[0], loss_range[1])
            else:
                loss_rate = 0.0  # Default: no loss
                
            if 'duration' in self.config:
                duration = self.config['duration']
            else:
                duration = 30.0  # Default: 30 seconds
            
            # Create trace with generated parameters
            trace = generate_constant_trace(
                duration=duration,
                bandwidth=bandwidth,
                latency=latency,
                queue_size=queue_size,
                loss_rate=loss_rate,
                name=f"trace_{i:05d}"
            )
            
            self.traces.append(trace)
    
    def dump(self, save_dir: str) -> None:
        """Save all traces to disk.
        
        Args:
            save_dir: Directory to save traces
        """
        os.makedirs(save_dir, exist_ok=True)
        for i, trace in enumerate(self.traces):
            trace.dump(os.path.join(save_dir, f"trace_{i:05d}.json"))
    
    @staticmethod
    def load_from_file(trace_file: str) -> 'SyntheticDataset':
        """Load dataset from a file containing trace paths.
        
        Args:
            trace_file: File with one trace path per line
            
        Returns:
            Loaded dataset
        """
        traces = []
        with open(trace_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(Trace.load_from_file(line))
        
        dataset = SyntheticDataset(len(traces), None)
        dataset.traces = traces
        return dataset
    
    @staticmethod
    def load_from_dir(trace_dir: str) -> 'SyntheticDataset':
        """Load dataset from a directory of trace files.
        
        Args:
            trace_dir: Directory containing trace files
            
        Returns:
            Loaded dataset
        """
        files = sorted(glob.glob(os.path.join(trace_dir, "trace_*.json")))
        traces = []
        for file in files:
            traces.append(Trace.load_from_file(file))
        
        dataset = SyntheticDataset(len(traces), None)
        dataset.traces = traces
        return dataset
    
    def __len__(self) -> int:
        """Get number of traces in dataset.
        
        Returns:
            Number of traces
        """
        return len(self.traces)
    
    def __getitem__(self, idx: int) -> Trace:
        """Get trace by index.
        
        Args:
            idx: Trace index
            
        Returns:
            Trace at index
        """
        return self.traces[idx] 