"""Prover and Verifier model architectures."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseProver(nn.Module, ABC):
    """Abstract base for provers: x → z."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BaseVerifier(nn.Module, ABC):
    """Abstract base for verifiers: (x, z) → logit."""

    @abstractmethod
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        pass


class MLPProver(BaseProver):
    """MLP prover: x → [MLP] → tanh → z."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        message_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, message_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self.message_dim = message_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        return self.network(x_flat)


class MLPVerifier(BaseVerifier):
    """MLP verifier: concat(x, z) → [MLP] → logit."""

    def __init__(
        self,
        input_dim: int,
        message_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim + message_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.message_dim = message_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        combined = torch.cat([x_flat, z], dim=-1)
        return self.network(combined).squeeze(-1)


def create_mlp_models(
    input_dim: int,
    hidden_dim: int = 128,
    message_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> Tuple[MLPProver, MLPVerifier]:
    """Create matched MLP prover-verifier pair."""
    prover = MLPProver(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        message_dim=message_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    verifier = MLPVerifier(
        input_dim=input_dim,
        message_dim=message_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return prover, verifier


class TransformerProver(BaseProver):
    """Transformer prover: text → [encoder] → [CLS] → tanh → z."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        message_dim: int = 256,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, message_dim),
            nn.Tanh(),
        )
        self.message_dim = message_dim
        self.model_name = model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_output)


class TransformerVerifier(BaseVerifier):
    """Transformer verifier: concat([CLS], z) → classifier → logit."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        message_dim: int = 256,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.message_dim = message_dim
        self.model_name = model_name

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + message_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        message: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_output, message], dim=-1)
        return self.classifier(combined).squeeze(-1)


def create_transformer_models(
    model_name: str = "distilbert-base-uncased",
    message_dim: int = 256,
    freeze_encoder: bool = False,
) -> Tuple[TransformerProver, TransformerVerifier]:
    """Create matched Transformer prover-verifier pair."""
    prover = TransformerProver(
        model_name=model_name,
        message_dim=message_dim,
        freeze_encoder=freeze_encoder,
    )
    verifier = TransformerVerifier(
        model_name=model_name,
        message_dim=message_dim,
        freeze_encoder=freeze_encoder,
    )
    return prover, verifier
