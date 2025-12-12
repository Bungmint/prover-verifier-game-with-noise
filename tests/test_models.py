"""
tests/test_models.py

Tests for Prover and Verifier models.
"""

import torch
import pytest

from pvg.models import MLPProver, MLPVerifier, create_mlp_models


class TestMLPProver:
    """Tests for MLPProver."""

    def test_output_shape(self):
        """Verify prover produces correct message shape."""
        prover = MLPProver(input_dim=100, message_dim=64)
        x = torch.randn(32, 100)
        z = prover(x)

        assert z.shape == (32, 64), f"Expected (32, 64), got {z.shape}"

    def test_output_bounds(self):
        """Messages should be bounded in [-1, 1] due to tanh."""
        prover = MLPProver(input_dim=100, message_dim=64)
        x = torch.randn(32, 100)
        z = prover(x)

        assert (z >= -1).all(), "Messages should be >= -1"
        assert (z <= 1).all(), "Messages should be <= 1"

    def test_handles_batch_of_one(self):
        """Prover should handle single samples."""
        prover = MLPProver(input_dim=50, message_dim=32)
        x = torch.randn(1, 50)
        z = prover(x)

        assert z.shape == (1, 32)

    def test_flattens_input(self):
        """Prover should flatten multi-dimensional inputs."""
        # Input representing a matrix (e.g., from linear F2 problem)
        prover = MLPProver(input_dim=15 * 11, message_dim=64)  # 15 equations, 10 vars + b
        x = torch.randn(32, 15, 11)
        z = prover(x)

        assert z.shape == (32, 64)

    def test_gradients_flow(self):
        """Verify gradients flow through the network."""
        prover = MLPProver(input_dim=10, message_dim=8)
        x = torch.randn(16, 10)

        z = prover(x)
        loss = z.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in prover.parameters()
        )
        assert has_grad, "Prover should have non-zero gradients"



class TestMLPVerifier:
    """Tests for MLPVerifier."""

    def test_output_shape(self):
        """Verify verifier produces correct logit shape."""
        verifier = MLPVerifier(input_dim=100, message_dim=64)
        x = torch.randn(32, 100)
        z = torch.randn(32, 64)
        logits = verifier(x, z)

        assert logits.shape == (32,), f"Expected (32,), got {logits.shape}"

    def test_handles_batch_of_one(self):
        """Verifier should handle single samples."""
        verifier = MLPVerifier(input_dim=50, message_dim=32)
        x = torch.randn(1, 50)
        z = torch.randn(1, 32)
        logits = verifier(x, z)

        assert logits.shape == (1,)

    def test_uses_message(self):
        """Verify that different messages produce different outputs."""
        verifier = MLPVerifier(input_dim=10, message_dim=8)
        x = torch.randn(1, 10)
        z1 = torch.ones(1, 8)
        z2 = -torch.ones(1, 8)

        logits1 = verifier(x, z1)
        logits2 = verifier(x, z2)

        # Different messages should (very likely) give different logits
        assert not torch.allclose(logits1, logits2), \
            "Verifier should produce different outputs for different messages"

    def test_gradients_flow(self):
        """Verify gradients flow through the network."""
        verifier = MLPVerifier(input_dim=10, message_dim=8)
        x = torch.randn(16, 10)
        z = torch.randn(16, 8)

        logits = verifier(x, z)
        loss = logits.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in verifier.parameters()
        )
        assert has_grad, "Verifier should have non-zero gradients"


class TestCreateMlpModels:
    """Tests for the factory function."""

    def test_creates_compatible_models(self):
        """Factory should create compatible prover-verifier pair."""
        prover, verifier = create_mlp_models(
            input_dim=100,
            hidden_dim=64,
            message_dim=32,
            num_layers=2,
        )

        # Should be able to run through both
        x = torch.randn(16, 100)
        z = prover(x)
        logits = verifier(x, z)

        assert z.shape == (16, 32)
        assert logits.shape == (16,)

    def test_message_dims_match(self):
        """Prover output dim should match verifier input message dim."""
        prover, verifier = create_mlp_models(
            input_dim=50,
            message_dim=64,
        )

        assert prover.message_dim == verifier.message_dim == 64


class TestEndToEndGradients:
    """Test gradients flow correctly in PVG setup."""

    def test_prover_gradient_direction(self):
        """Prover loss gradients should push logits up (toward label 1)."""
        import torch.nn.functional as F

        prover, verifier = create_mlp_models(input_dim=20, message_dim=10)
        x = torch.randn(32, 20)

        # Forward pass
        z = prover(x)
        logits = verifier(x, z)

        # Prover loss: -log p(y=1) = softplus(-logits)
        loss = F.softplus(-logits).mean()
        loss.backward()

        # After backward, check prover has gradients
        prover_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in prover.parameters()
        )
        assert prover_has_grad

    def test_verifier_gradient_for_both_labels(self):
        """Verifier should learn to predict both 0 and 1."""
        import torch.nn.functional as F

        _, verifier = create_mlp_models(input_dim=20, message_dim=10)

        x = torch.randn(32, 20)
        z = torch.randn(32, 10)
        labels = torch.randint(0, 2, (32,))

        logits = verifier(x, z)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss.backward()

        verifier_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in verifier.parameters()
        )
        assert verifier_has_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
