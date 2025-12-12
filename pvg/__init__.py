"""
pvg - Prover-Verifier Games with Noisy Labels

A framework for studying how label noise affects Stackelberg equilibria
in Prover-Verifier Games.

Main components:
- Models: MLPProver, MLPVerifier for synthetic tasks
- Training: StackelbergTrainer for bilevel optimization
- Config: ExperimentConfig, SyntheticConfig for experiment setup
- Noise: inject_label_noise for symmetric label corruption
- Losses: compute_step_metrics for training/evaluation
- Metrics: MetricsTracker, analyze_noise_sweep for experiment analysis
"""

from pvg.config import ExperimentConfig, SyntheticConfig
from pvg.losses import compute_step_metrics
from pvg.metrics import (
    MetricsTracker,
    aggregate_runs,
    analyze_noise_sweep,
    compute_equilibrium_deviation,
    save_analysis,
)
from pvg.models import (
    BaseProver,
    BaseVerifier,
    MLPProver,
    MLPVerifier,
    create_mlp_models,
)
from pvg.noise import inject_label_noise
from pvg.training import StackelbergTrainer

__all__ = [
    # Config
    "ExperimentConfig",
    "SyntheticConfig",
    # Models
    "BaseProver",
    "BaseVerifier",
    "MLPProver",
    "MLPVerifier",
    "create_mlp_models",
    # Training
    "StackelbergTrainer",
    # Noise
    "inject_label_noise",
    # Losses
    "compute_step_metrics",
    # Metrics
    "MetricsTracker",
    "compute_equilibrium_deviation",
    "aggregate_runs",
    "analyze_noise_sweep",
    "save_analysis",
]

__version__ = "0.1.0"
