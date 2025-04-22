#!/usr/bin/env python3
"""Evaluate congestion control models on synthetic traces."""

import argparse
import os
import time
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm

from cc_simulator.models.aurora_rllib import AuroraRllib
from cc_simulator.network_simulator.cubic import Cubic
from cc_simulator.network_simulator.bbr import BBR
from cc_simulator.utils.trace import Trace
from cc_simulator.utils.synthetic_dataset import SyntheticDataset


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser("Congestion Control Evaluation")
    parser.add_argument('--save-dir', type=str, default="results",
                      help="Directory to save testing results")
    parser.add_argument('--dataset-dir', type=str, default="data/synthetic_dataset",
                      help="Directory of synthetic dataset")
    parser.add_argument('--cc', type=str, default="aurora",
                      choices=("aurora", "bbr", "cubic"),
                      help='Congestion control algorithm')
    parser.add_argument('--model-path', type=str,
                      help='Path to trained model (for RL-based algorithms)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--nproc', type=int, default=1,
                      help='Number of processes for parallel evaluation')
    parser.add_argument('--plot', action='store_true',
                      help='Generate plots of the results')
    parser.add_argument('--trace-count', type=int, default=None,
                      help='Number of traces to evaluate (default: all)')
    
    return parser.parse_args()


def evaluate_trace(
    cc_type: str,
    trace: Trace,
    save_dir: str,
    seed: int,
    model_path: Optional[str] = None,
    plot: bool = False
) -> Dict[str, Any]:
    """Evaluate a congestion control algorithm on a trace.
    
    Args:
        cc_type: Congestion control algorithm type
        trace: Network trace
        save_dir: Directory to save results
        seed: Random seed
        model_path: Path to model (for RL-based algorithms)
        plot: Whether to generate plots
        
    Returns:
        Evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if cc_type == "aurora":
        # Use Aurora RLlib
        agent = AuroraRllib(
            seed=seed,
            log_dir=save_dir,
            pretrained_model_path=model_path,
            record_pkt_log=True
        )
        reward, pkt_level_reward = agent.test(trace, save_dir, plot)
        agent.cleanup()
        
        return {
            "algorithm": "aurora",
            "mean_reward": reward,
            "throughput_reward": pkt_level_reward
        }
    
    elif cc_type == "bbr" or cc_type == "cubic":
        # This would require implementing traditional algorithms with the same interface
        # For now, return a placeholder
        return {
            "algorithm": cc_type,
            "mean_reward": 0.0,
            "throughput_reward": 0.0
        }
        
    else:
        raise ValueError(f"Unknown CC type: {cc_type}")


def evaluate_traces_parallel(
    cc_type: str,
    traces: List[Trace],
    save_dirs: List[str],
    seed: int,
    model_path: Optional[str] = None,
    plot: bool = False,
    nproc: int = 1
) -> List[Dict[str, Any]]:
    """Evaluate traces in parallel.
    
    Args:
        cc_type: Congestion control algorithm type
        traces: List of traces
        save_dirs: List of save directories
        seed: Random seed
        model_path: Path to model
        plot: Whether to generate plots
        nproc: Number of processes
        
    Returns:
        List of evaluation results
    """
    if nproc == 1:
        # Single process
        results = []
        for trace, save_dir in tqdm.tqdm(zip(traces, save_dirs), total=len(traces)):
            result = evaluate_trace(cc_type, trace, save_dir, seed, model_path, plot)
            results.append(result)
        return results
    else:
        # Multiple processes
        with mp.Pool(nproc) as pool:
            args = [(cc_type, trace, save_dir, seed, model_path, plot)
                   for trace, save_dir in zip(traces, save_dirs)]
            results = list(tqdm.tqdm(
                pool.starmap(evaluate_trace, args),
                total=len(args)
            ))
        return results


def main():
    """Main function."""
    args = parse_args()
    
    # Load dataset
    dataset = SyntheticDataset.load_from_dir(args.dataset_dir)
    
    # Limit number of traces if specified
    if args.trace_count is not None:
        traces = dataset.traces[:args.trace_count]
    else:
        traces = dataset.traces
    
    # Create save directories
    save_dirs = [os.path.join(args.save_dir, f'trace_{i:05d}')
                for i in range(len(traces))]
    
    # Run evaluation
    t_start = time.time()
    
    results = evaluate_traces_parallel(
        args.cc,
        traces,
        save_dirs,
        args.seed,
        args.model_path,
        args.plot,
        args.nproc
    )
    
    # Save summary results
    df = pd.DataFrame(results)
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(os.path.join(args.save_dir, f"{args.cc}_results.csv"), index=False)
    
    # Print summary
    print(f"Mean reward: {df['mean_reward'].mean():.4f}")
    print(f"Evaluation completed in {(time.time() - t_start) / 60:.2f} minutes")


if __name__ == "__main__":
    main() 