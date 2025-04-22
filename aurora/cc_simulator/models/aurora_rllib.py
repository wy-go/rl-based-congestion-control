"""RLlib-based Aurora congestion control model."""

import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

from cc_simulator.environments.aurora_env import AuroraEnvironment
from cc_simulator.utils.trace import Trace, generate_constant_trace
from cc_simulator.utils.constants import REWARD_SCALE


def register_aurora_env():
    """Register the Aurora environment with RLlib."""
    register_env(
        "aurora_env",
        lambda config: AuroraEnvironment(
            traces=config.get("traces", []),
            history_len=config.get("history_len", 10),
            features=config.get("features", ["sent latency inflation", "latency ratio", "recv ratio"]),
            record_pkt_log=config.get("record_pkt_log", False),
            render_mode=config.get("render_mode", None)
        )
    )


class AuroraRllib:
    """RLlib-based implementation of Aurora congestion control."""
    
    def __init__(
        self,
        seed: int = 42,
        log_dir: str = "./logs",
        pretrained_model_path: Optional[str] = None,
        record_pkt_log: bool = False,
        cuda: bool = True,
        num_workers: int = 1
    ):
        """Initialize the Aurora RLlib agent.
        
        Args:
            seed: Random seed
            log_dir: Directory for logs
            pretrained_model_path: Path to pretrained model checkpoint
            record_pkt_log: Whether to record packet logs
            cuda: Whether to use CUDA for training
            num_workers: Number of workers for training
        """
        self.seed = seed
        self.log_dir = log_dir
        self.pretrained_model_path = pretrained_model_path
        self.record_pkt_log = record_pkt_log
        self.cuda = cuda and torch.cuda.is_available()
        self.num_workers = num_workers
        
        # RLlib setup
        self.ray_initialized = False
        self.model = None
        
        # Register environment
        register_aurora_env()
        
        # Default features
        self.features = ["sent latency inflation", "latency ratio", "recv ratio"]
        self.history_len = 10
        
        # Initialize
        self._initialize_ray()
        self._initialize_model()
        
    def _initialize_ray(self) -> None:
        """Initialize Ray if not already initialized."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            self.ray_initialized = True
        
    def _initialize_model(self) -> None:
        """Initialize the RLlib model."""
        # Create a dummy trace for initialization
        dummy_trace = generate_constant_trace(
            duration=30.0,
            bandwidth=10.0,
            latency=20.0,
            queue_size=5
        )
        
        # Create RLlib config
        config = (
            PPOConfig()
            .environment(
                "aurora_env", 
                env_config={
                    "traces": [dummy_trace],
                    "history_len": self.history_len,
                    "features": self.features,
                    "record_pkt_log": self.record_pkt_log
                }
            )
            .framework("torch")
            .training(
                gamma=0.99,
                lr=3e-4,
                kl_coeff=0.0,
                lambda_=0.95,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10
            )
            .resources(
                num_gpus=1 if self.cuda else 0,
                num_cpus_per_worker=1
            )
            .rollouts(
                num_rollout_workers=self.num_workers,
                rollout_fragment_length=200
            )
            .debugging(seed=self.seed)
        )
        
        # Create model
        self.model = PPO(config=config)
        
        # Load pretrained model if provided
        if self.pretrained_model_path and os.path.exists(self.pretrained_model_path):
            self.model.restore(self.pretrained_model_path)
    
    def train(
        self,
        training_traces: List[Trace],
        validation_traces: List[Trace],
        total_timesteps: int = 1000000,
        validation_interval: int = 10000,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            training_traces: List of traces for training
            validation_traces: List of traces for validation
            total_timesteps: Total timesteps to train for
            validation_interval: Interval for validation
            save_path: Path to save model checkpoints
            
        Returns:
            Training results
        """
        # Update environment config with training traces
        self.model.config.env_config["traces"] = training_traces
        
        # Create checkpoint config
        checkpoint_config = {
            "checkpoint_frequency": validation_interval,
            "checkpoint_at_end": True
        }
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            checkpoint_config["checkpoint_dir"] = save_path
        
        # Setup TuneConfig for validation
        def validation_fn(policy: Policy, *args, **kwargs) -> Dict[str, float]:
            """Validation function for RLlib."""
            total_reward = 0.0
            for trace in validation_traces:
                # Create test environment
                env = AuroraEnvironment(
                    traces=[trace],
                    history_len=self.history_len,
                    features=self.features,
                    record_pkt_log=False
                )
                
                # Run validation episode
                obs, info = env.reset()
                done = False
                episode_reward = 0.0
                
                while not done:
                    action = policy.compute_single_action(obs)[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                total_reward += episode_reward
            
            # Return average reward across validation traces
            avg_reward = total_reward / len(validation_traces)
            return {"validation_reward": avg_reward}
        
        # Train the model
        results = self.model.train()
        
        # Return training results
        return results
    
    def test(
        self,
        trace: Trace,
        save_dir: Optional[str] = None,
        plot_flag: bool = False
    ) -> Tuple[float, float]:
        """Test the model on a trace.
        
        Args:
            trace: Trace to test on
            save_dir: Directory to save results
            plot_flag: Whether to plot results
            
        Returns:
            Tuple of (mean reward, throughput)
        """
        # Create test environment
        env = AuroraEnvironment(
            traces=[trace],
            history_len=self.history_len,
            features=self.features,
            record_pkt_log=self.record_pkt_log
        )
        
        # Run test episode
        obs, info = env.reset()
        done = False
        reward_list = []
        throughput_list = []
        latency_list = []
        loss_list = []
        
        while not done:
            action = self.model.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_list.append(reward)
            throughput_list.append(info["throughput"])
            latency_list.append(info["latency"])
            loss_list.append(info["loss"])
            done = terminated or truncated
        
        # Calculate metrics
        mean_reward = np.mean(reward_list)
        pkt_level_reward = np.mean(reward_list) / REWARD_SCALE
        
        # Save logs if requested
        if save_dir and self.record_pkt_log:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save packet logs
            if hasattr(env, "net") and hasattr(env.net, "pkt_log"):
                import csv
                with open(os.path.join(save_dir, "aurora_packet_log.csv"), 'w', 1) as f:
                    pkt_logger = csv.writer(f, lineterminator='\n')
                    pkt_logger.writerow([
                        'timestamp', 'packet_event_id', 'event_type',
                        'bytes', 'cur_latency', 'queue_delay',
                        'packet_in_queue', 'sending_rate', 'bandwidth'
                    ])
                    pkt_logger.writerows(env.net.pkt_log)
            
            # Plot if requested
            if plot_flag:
                # This would require implementing plotting functionality
                pass
        
        return mean_reward, pkt_level_reward
    
    def test_on_traces(
        self,
        traces: List[Trace],
        save_dirs: List[str],
        plot_flag: bool = False
    ) -> List[Tuple[float, float]]:
        """Test the model on multiple traces.
        
        Args:
            traces: List of traces to test on
            save_dirs: List of directories to save results
            plot_flag: Whether to plot results
            
        Returns:
            List of (mean reward, throughput) tuples
        """
        results = []
        for trace, save_dir in zip(traces, save_dirs):
            result = self.test(trace, save_dir, plot_flag)
            results.append(result)
        return results
    
    def save_model(self, save_path: str) -> str:
        """Save the model.
        
        Args:
            save_path: Path to save model
            
        Returns:
            Path to saved model
        """
        os.makedirs(save_path, exist_ok=True)
        checkpoint_path = self.model.save(save_path)
        return checkpoint_path
    
    def load_model(self, load_path: str) -> None:
        """Load a saved model.
        
        Args:
            load_path: Path to load model from
        """
        self.model.restore(load_path)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False 