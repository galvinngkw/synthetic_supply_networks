"""
Network Generator: Synthetic Supply Network Generator

A PyTorch-based generator for realistic firm-level supply networks that match empirical properties observed in real-world data.

Quick Start:
    from network_generator import NetworkConfig, TargetConfig, TrainingConfig

    network_config = NetworkConfig(num_firms=10000)
    target_config = TargetConfig()
    training_config = TrainingConfig()
"""

from network_generator.config import (
    NetworkConfig,
    TargetConfig,
    TrainingConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "NetworkConfig",
    "TargetConfig",
    "TrainingConfig",
]
