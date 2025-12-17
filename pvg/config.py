"""Configuration dataclasses for PVG experiments."""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json

import torch


@dataclass
class ExperimentConfig:
    """Base configuration for PVG experiments."""

    experiment_name: str = "default"
    seed: int = 42
    epsilon: float = 0.0

    num_rounds: int = 100
    prover_steps_per_round: int = 10
    verifier_steps_per_round: int = 5

    prover_lr: float = 1e-4
    verifier_lr: float = 1e-4
    batch_size: int = 64

    log_every: int = 10
    eval_every: int = 20

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_wandb: bool = False
    wandb_project: str = "prover-verifier-game"
    wandb_entity: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**d)


@dataclass
class SyntheticConfig(ExperimentConfig):
    """Configuration for synthetic (Linear F2) experiments."""

    tier: Literal["synthetic"] = "synthetic"

    n_vars: int = 12
    n_equations: int = 10

    dataset_size: int = 5000
    eval_size: int = 1000

    hidden_dim: int = 128
    message_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    @property
    def input_dim(self) -> int:
        """Flattened [A, b] dimension."""
        return self.n_equations * (self.n_vars + 1)


@dataclass
class NLPConfig(ExperimentConfig):
    """Configuration for NLP experiments."""

    tier: Literal["nlp"] = "nlp"

    model_name: str = "distilbert-base-uncased"
    message_dim: int = 256
    freeze_encoder: bool = False
    max_length: int = 128

    dataset_name: str = "snli"
    train_samples: int = 50000
    eval_samples: int = 5000

    prover_lr: float = 2e-5
    verifier_lr: float = 2e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1

    batch_size: int = 16
    num_rounds: int = 50
    prover_steps_per_round: int = 5
    verifier_steps_per_round: int = 3
