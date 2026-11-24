"""
Network Generator: Synthetic Supply Network Generator

A PyTorch-based generator for realistic firm-level supply networks that match empirical properties observed in real-world data.

Quick Start:
    from network_generator import generate_synthetic_network

    result = generate_synthetic_network("HUN", 2015, num_firms=100_000, seed=1)

Or configure each step:
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
    # End-to-end generation
    "generate_synthetic_network",
    "SyntheticNetwork",
    "DegreeAcceptance",
]

_PIPELINE_EXPORTS = {"generate_synthetic_network", "SyntheticNetwork", "DegreeAcceptance"}


def __getattr__(name):
    # Imported lazily so that `import network_generator` (and the data command line) does not load torch.
    if name in _PIPELINE_EXPORTS:
        from network_generator import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'network_generator' has no attribute {name!r}")
