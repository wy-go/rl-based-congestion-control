"""Reward functions for congestion control algorithms."""

import numpy as np


def pcc_aurora_reward(
    throughput: float,
    latency: float,
    loss_rate: float,
    avg_bw: float = None,
    avg_lat: float = None,
    delta: float = 0.5,
    lat_factor: float = 0.5,
    alpha: float = 0.01,
) -> float:
    """Calculate reward based on PCC-Aurora's reward function.
    
    Args:
        throughput: Throughput in packets/second
        latency: Latency in seconds
        loss_rate: Packet loss rate (0-1)
        avg_bw: Average bandwidth in packets/second (for normalization)
        avg_lat: Average latency in seconds (for normalization)
        delta: Weight for latency penalty
        lat_factor: Scaling factor for latency
        alpha: Weight for loss penalty
        
    Returns:
        float: Reward value
    """
    # Calculate throughput utility
    throughput_utility = np.log(1 + throughput)
    
    # Calculate latency penalty
    latency_penalty = latency_sigmoid(latency, lat_factor, avg_lat)
    
    # Apply penalties to throughput utility
    reward = throughput_utility
    
    if delta != 0:
        reward -= delta * latency_penalty
    
    if alpha != 0:
        reward -= alpha * loss_rate
    
    return reward


def latency_sigmoid(
    latency: float,
    lat_factor: float = 0.5,
    avg_lat: float = None
) -> float:
    """Calculate latency penalty using sigmoid function.
    
    Args:
        latency: Latency in seconds
        lat_factor: Scaling factor
        avg_lat: Average latency for normalization
        
    Returns:
        float: Latency penalty value
    """
    if avg_lat is not None:
        # Normalize latency if average is provided
        norm_latency = latency / avg_lat
        return 1 / (1 + np.exp(-lat_factor * (norm_latency - 1)))
    else:
        # Non-normalized version
        return latency


def normalize_reward(
    reward: float,
    scale: float = 0.001
) -> float:
    """Scale reward to reasonable range for RL algorithms.
    
    Args:
        reward: Raw reward value
        scale: Scaling factor
        
    Returns:
        float: Normalized reward
    """
    return reward * scale 