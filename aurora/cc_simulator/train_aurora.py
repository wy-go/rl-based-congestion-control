#!/usr/bin/env python3
"""Train Aurora congestion control using RLlib."""

import argparse
import os
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from cc_simulator.models.aurora_rllib import AuroraRllib
from cc_simulator.utils.trace import generate_constant_trace
from cc_simulator.utils.synthetic_dataset import SyntheticDataset


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser("Aurora Training")
    parser.add_argument('--log-dir', type=str, default="logs",
                      help="Directory for logs and checkpoints")
    parser.add_argument('--config-file', type=str,
                      help="Path to trace configuration file")
    parser.add_argument('--num-traces', type=int, default=100,
                      help="Number of training traces")
    parser.add_argument('--num-val-traces', type=int, default=20,
                      help="Number of validation traces")
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                      help="Total timesteps for training")
    parser.add_argument('--validation-interval', type=int, default=10000,
                      help="Interval for validation")
    parser.add_argument('--seed', type=int, default=42,
                      help="Random seed")
    parser.add_argument('--cuda', action='store_true',
                      help="Use CUDA if available")
    parser.add_argument('--num-workers', type=int, default=1,
                      help="Number of workers for training")
    parser.add_argument('--load-checkpoint', type=str,
                      help="Path to checkpoint to resume training from")
    
    return parser.parse_args()


def generate_traces(config_file: str, num_traces: int, seed: int) -> List[Trace]:
    """Generate traces for training or validation.
    
    Args:
        config_file: Path to trace configuration file
        num_traces: Number of traces to generate
        seed: Random seed
        
    Returns:
        List of generated traces
    """
    dataset = SyntheticDataset(num_traces, config_file=config_file, seed=seed)
    return dataset.traces


def main():
    """Main function."""
    args = parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Generate traces
    print("Generating training traces...")
    training_traces = generate_traces(
        args.config_file, 
        args.num_traces, 
        args.seed
    )
    
    print("Generating validation traces...")
    validation_traces = generate_traces(
        args.config_file, 
        args.num_val_traces, 
        args.seed + 1
    )
    
    # Initialize Aurora agent
    print("Initializing Aurora RLlib agent...")
    agent = AuroraRllib(
        seed=args.seed,
        log_dir=args.log_dir,
        pretrained_model_path=args.load_checkpoint,
        cuda=args.cuda,
        num_workers=args.num_workers
    )
    
    # Train the agent
    print(f"Starting training for {args.total_timesteps} timesteps...")
    t_start = time.time()
    
    results = agent.train(
        training_traces=training_traces,
        validation_traces=validation_traces,
        total_timesteps=args.total_timesteps,
        validation_interval=args.validation_interval,
        save_path=os.path.join(args.log_dir, "checkpoints")
    )
    
    # Save final model
    final_checkpoint = agent.save_model(os.path.join(args.log_dir, "final_model"))
    print(f"Final model saved to {final_checkpoint}")
    
    # Cleanup
    agent.cleanup()
    
    elapsed_time = (time.time() - t_start) / 60
    print(f"Training completed in {elapsed_time:.2f} minutes")


if __name__ == "__main__":
    main() 