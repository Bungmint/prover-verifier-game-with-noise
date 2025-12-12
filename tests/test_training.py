"""
tests/test_training.py

Tests for the Stackelberg training loop.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import pytest

from pvg.config import SyntheticConfig
from pvg.models import create_mlp_models
from pvg.training import StackelbergTrainer


def create_dummy_dataloaders(
    n_samples: int = 256,
    input_dim: int = 50,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Create dummy dataloaders for testing."""
    # Create random binary classification data
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, 2, (n_samples,))

    dataset = TensorDataset(x, y)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader


class TestStackelbergTrainer:
    """Tests for StackelbergTrainer."""

    @pytest.fixture
    def setup_trainer(self):
        """Create a trainer with small config for fast testing."""
        config = SyntheticConfig(
            experiment_name="test",
            seed=42,
            epsilon=0.0,
            num_rounds=5,
            prover_steps_per_round=2,
            verifier_steps_per_round=2,
            prover_lr=1e-3,
            verifier_lr=1e-3,
            batch_size=32,
            log_every=1,
            eval_every=2,
            device="cpu",
            n_vars=5,
            n_equations=8,
            hidden_dim=32,
            message_dim=16,
        )

        prover, verifier = create_mlp_models(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            message_dim=config.message_dim,
            num_layers=config.num_layers,
        )

        train_loader, eval_loader = create_dummy_dataloaders(
            n_samples=128,
            input_dim=config.input_dim,
            batch_size=config.batch_size,
        )

        trainer = StackelbergTrainer(
            prover=prover,
            verifier=verifier,
            config=config,
            train_loader=train_loader,
            eval_loader=eval_loader,
        )

        return trainer

    def test_initialization(self, setup_trainer):
        """Trainer should initialize without error."""
        trainer = setup_trainer
        assert trainer.prover is not None
        assert trainer.verifier is not None
        assert trainer.history == {"train": [], "eval": []}

    def test_train_runs_without_error(self, setup_trainer):
        """Training loop should complete without errors."""
        trainer = setup_trainer
        history = trainer.train()

        assert "train" in history
        assert "eval" in history
        assert len(history["train"]) > 0

    def test_evaluate_returns_metrics(self, setup_trainer):
        """Evaluation should return expected metrics."""
        trainer = setup_trainer
        metrics = trainer.evaluate()

        assert "clean_loss" in metrics
        assert "accuracy" in metrics
        assert "prover_success_rate" in metrics

        # Accuracy should be between 0 and 1
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["prover_success_rate"] <= 1

    def test_prover_frozen_during_verifier_step(self, setup_trainer):
        """Prover parameters should not change during verifier step."""
        trainer = setup_trainer

        # Get initial prover parameters
        trainer._freeze_prover()
        trainer._unfreeze_verifier()

        initial_params = [p.clone() for p in trainer.prover.parameters()]

        # Run verifier step
        trainer._verifier_step()

        # Check prover didn't change
        for p_initial, p_current in zip(initial_params, trainer.prover.parameters()):
            assert torch.allclose(p_initial, p_current), \
                "Prover should not change during verifier step"

    def test_verifier_frozen_during_prover_step(self, setup_trainer):
        """Verifier parameters should not change during prover step."""
        trainer = setup_trainer

        # Get initial verifier parameters
        trainer._freeze_verifier()
        trainer._unfreeze_prover()

        initial_params = [p.clone() for p in trainer.verifier.parameters()]

        # Run prover step
        trainer._prover_step()

        # Check verifier didn't change
        for p_initial, p_current in zip(initial_params, trainer.verifier.parameters()):
            assert torch.allclose(p_initial, p_current), \
                "Verifier should not change during prover step"

    def test_history_tracking(self, setup_trainer):
        """History should be properly tracked during training."""
        trainer = setup_trainer
        trainer.train()

        # Should have train history entries
        assert len(trainer.history["train"]) > 0

        # Each train entry should have expected keys
        train_entry = trainer.history["train"][0]
        assert "round" in train_entry
        assert "prover_loss" in train_entry
        assert "verifier_loss" in train_entry


class TestNoiseInjection:
    """Test that noise is properly injected during training."""

    def test_training_with_noise(self):
        """Training should work with non-zero epsilon."""
        config = SyntheticConfig(
            experiment_name="test_noise",
            seed=42,
            epsilon=0.2,  # 20% noise
            num_rounds=3,
            prover_steps_per_round=2,
            verifier_steps_per_round=2,
            device="cpu",
            n_vars=5,
            n_equations=8,
            hidden_dim=32,
            message_dim=16,
        )

        prover, verifier = create_mlp_models(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            message_dim=config.message_dim,
        )

        train_loader, eval_loader = create_dummy_dataloaders(
            n_samples=64,
            input_dim=config.input_dim,
            batch_size=32,
        )

        trainer = StackelbergTrainer(
            prover=prover,
            verifier=verifier,
            config=config,
            train_loader=train_loader,
            eval_loader=eval_loader,
        )

        # Should complete without error
        history = trainer.train()
        assert len(history["train"]) > 0


class TestBatchCycling:
    """Test that batch cycling works correctly."""

    def test_cycles_through_dataset(self):
        """Trainer should cycle through dataset when steps > dataset size."""
        config = SyntheticConfig(
            experiment_name="test_cycling",
            seed=42,
            epsilon=0.0,
            num_rounds=2,
            prover_steps_per_round=10,  # More steps than batches
            verifier_steps_per_round=10,
            device="cpu",
            n_vars=5,
            n_equations=8,
            hidden_dim=32,
            message_dim=16,
            batch_size=32,
        )

        prover, verifier = create_mlp_models(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            message_dim=config.message_dim,
        )

        # Small dataset - will need to cycle
        train_loader, eval_loader = create_dummy_dataloaders(
            n_samples=64,  # Only 2 batches of 32
            input_dim=config.input_dim,
            batch_size=32,
        )

        trainer = StackelbergTrainer(
            prover=prover,
            verifier=verifier,
            config=config,
            train_loader=train_loader,
            eval_loader=eval_loader,
        )

        # Should complete without error despite needing to cycle
        history = trainer.train()
        assert len(history["train"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
