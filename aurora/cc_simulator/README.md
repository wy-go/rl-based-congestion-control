# Aurora CC Simulator

A modern implementation of Aurora congestion control using PyTorch, RLlib, and Gymnasium.

## Overview

This package provides a modern implementation of the Aurora, originally proposed in the paper "A Deep Reinforcement Learning Perspective on Internet Congestion Control" (ICML 2019). The implementation upgrades the original codebase [PCC-RL](https://github.com/PCCproject/PCC-RL) to use:

- **PyTorch**: Modern deep learning framework
- **RLlib**: Scalable reinforcement learning library
- **Gymnasium**: Updated RL environment standard

## Key Features

Aurora is a Reinforcement Learning (RL) based congestion control algorithm that enable capturing intricate patterns in data traffic and network conditions.

This upgraded implementation offers several improvements:

- **Distributed Training**: Scale across multiple GPUs/machines with RLlib
- **Modern ML Framework**: Better performance and GPU utilization with PyTorch
- **Type Annotations**: Comprehensive typing for better code understanding
- **Modular Design**: Clean separation of concerns for easier extension
- **Well-documented API**: Detailed docstrings and parameter descriptions

## Project Structure

```
cc_simulator/
├── __init__.py                 # Package initialization
├── evaluate_traces.py          # Trace evaluation script
├── train_aurora.py             # Aurora training script
├── environments/               # RL environments
│   ├── __init__.py
│   └── aurora_env.py           # Aurora Gymnasium environment
├── models/                     # RL model implementations
│   ├── __init__.py
│   └── aurora_rllib.py         # RLlib-based Aurora implementation
├── network_simulator/          # Network simulation components
│   ├── __init__.py
│   ├── packet.py               # Packet implementation
│   ├── sender.py               # Base sender class
│   ├── link.py                 # Network link implementation
│   ├── network.py              # Network simulator
│   └── aurora_sender.py        # Aurora sender implementation
└── utils/                      # Utility functions
    ├── __init__.py
    ├── constants.py            # Constants for simulation
    ├── reward_functions.py     # Reward functions
    ├── synthetic_dataset.py    # Dataset utilities
    └── trace.py                # Network trace implementation
```

## Usage

### Training

To train a new Aurora model:

```bash
python -m cc_simulator.train_aurora \
    --log-dir logs/aurora \
    --config-file config/traces/cellular.json \
    --num-traces 100 \
    --total-timesteps 1000000 \
    --cuda \
    --num-workers 4
```

### Evaluation

To evaluate a trained Aurora model:

```bash
python -m cc_simulator.evaluate_traces \
    --save-dir results/aurora \
    --dataset-dir data/synthetic_dataset \
    --cc aurora \
    --model-path logs/aurora/final_model \
    --nproc 4 \
    --plot
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Ray/RLlib 2.0+
- Gymnasium 0.26+
- NumPy
- Pandas (for evaluation)

## Citation

If you use this code, please cite the original Aurora paper:

```
@InProceedings{Jay_Aurora_2019,
  title = 	 {A Deep Reinforcement Learning Perspective on Internet Congestion Control},
  author = 	 {Jay, Nathan and Rotman, Noga and Godfrey, Brighten and Schapira, Michael and Tamar, Aviv},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {3050--3059},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {09--15 Jun},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/jay19a/jay19a.pdf},
  url = 	 {https://proceedings.mlr.press/v97/jay19a.html}
}
``` 

and this github repo:

```
@software{Zhao_rl-based-congestion-control_2025,
author = {Zhao, Wanyu},
doi = {10.5281/zenodo.1234},
month = apr,
title = {{rl-based-congestion-control}},
url = {https://github.com/wy-go/rl-based-congestion-control},
version = {1.0.0},
year = {2025}
}
```
