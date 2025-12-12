"""
tests/test_linear_f2.py

Tests for the Linear F2 dataset.
"""

import numpy as np
import torch
import pytest

from experiments.synthetic.linear_f2 import (
    gaussian_elimination_f2,
    LinearF2Dataset,
    create_dataloaders,
)


class TestGaussianEliminationF2:
    """Tests for the F2 Gaussian elimination algorithm."""

    def test_satisfiable_system(self):
        """Test a known satisfiable system."""
        # Simple system: x1 + x2 = 1, x1 = 0 -> x2 = 1
        A = np.array([[1, 1], [1, 0]])
        b = np.array([1, 0])

        is_sat, solution = gaussian_elimination_f2(A, b)

        assert is_sat, "System should be satisfiable"
        assert solution is not None

        # Verify solution
        result = (A @ solution) % 2
        assert np.array_equal(result, b), "Solution should satisfy Ax = b"

    def test_unsatisfiable_system(self):
        """Test a known unsatisfiable system."""
        # Contradiction: x = 0 and x = 1
        A = np.array([[1], [1]])
        b = np.array([0, 1])

        is_sat, solution = gaussian_elimination_f2(A, b)

        assert not is_sat, "System should be unsatisfiable"
        assert solution is None

    def test_underdetermined_system(self):
        """Test an underdetermined system (more vars than equations)."""
        # One equation: x1 + x2 = 1, has solution x1=1, x2=0 (or x1=0, x2=1)
        A = np.array([[1, 1]])
        b = np.array([1])

        is_sat, solution = gaussian_elimination_f2(A, b)

        assert is_sat, "Underdetermined system should have solutions"
        if solution is not None:
            result = (A @ solution) % 2
            assert np.array_equal(result, b)

    def test_overdetermined_satisfiable(self):
        """Test an overdetermined but satisfiable system."""
        # Three equations, two variables, but all consistent
        A = np.array([[1, 0], [0, 1], [1, 1]])
        b = np.array([1, 1, 0])  # x1=1, x2=1, x1+x2=0 (mod 2)

        is_sat, solution = gaussian_elimination_f2(A, b)

        assert is_sat
        if solution is not None:
            result = (A @ solution) % 2
            assert np.array_equal(result, b)

    def test_identity_system(self):
        """Test identity matrix system."""
        A = np.eye(3, dtype=np.int64)
        b = np.array([1, 0, 1])

        is_sat, solution = gaussian_elimination_f2(A, b)

        assert is_sat
        assert solution is not None
        assert np.array_equal(solution, b)

    def test_zero_matrix(self):
        """Test zero matrix with zero vector."""
        A = np.zeros((2, 3), dtype=np.int64)
        b = np.zeros(2, dtype=np.int64)

        is_sat, solution = gaussian_elimination_f2(A, b)

        assert is_sat, "0x = 0 should be satisfiable"

    def test_zero_matrix_nonzero_b(self):
        """Test zero matrix with non-zero vector."""
        A = np.zeros((2, 3), dtype=np.int64)
        b = np.array([1, 0])

        is_sat, solution = gaussian_elimination_f2(A, b)

        assert not is_sat, "0x = [1, 0] should be unsatisfiable"


class TestLinearF2Dataset:
    """Tests for the LinearF2Dataset class."""

    def test_dataset_creation(self):
        """Dataset should be created without error."""
        dataset = LinearF2Dataset(n_samples=100, n_vars=5, n_equations=8)

        assert len(dataset) == 100

    def test_sample_shapes(self):
        """Samples should have correct shapes."""
        dataset = LinearF2Dataset(n_samples=50, n_vars=10, n_equations=15)

        x, y = dataset[0]

        # Input should be flattened [A, b]
        expected_dim = 15 * 10 + 15  # n_eq * n_var + n_eq
        assert x.shape == (expected_dim,), f"Expected ({expected_dim},), got {x.shape}"
        assert y.shape == (), f"Label should be scalar, got {y.shape}"

    def test_label_values(self):
        """Labels should be binary (0 or 1)."""
        dataset = LinearF2Dataset(n_samples=100, n_vars=5, n_equations=8)

        for _, y in dataset:
            assert y.item() in [0, 1], f"Label should be 0 or 1, got {y.item()}"

    def test_reproducibility(self):
        """Same seed should produce same dataset."""
        dataset1 = LinearF2Dataset(n_samples=50, seed=42)
        dataset2 = LinearF2Dataset(n_samples=50, seed=42)

        for (x1, y1), (x2, y2) in zip(dataset1, dataset2):
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different datasets."""
        dataset1 = LinearF2Dataset(n_samples=50, seed=42)
        dataset2 = LinearF2Dataset(n_samples=50, seed=123)

        # At least some samples should differ
        any_differ = False
        for (x1, _), (x2, _) in zip(dataset1, dataset2):
            if not torch.equal(x1, x2):
                any_differ = True
                break

        assert any_differ, "Different seeds should produce different data"

    def test_class_balance(self):
        """Dataset should have reasonable class balance for square systems."""
        # For balanced classes, use n_equations ≈ n_vars (square system)
        # Overdetermined systems (n_eq > n_var) tend to be unsatisfiable
        # Underdetermined systems (n_eq < n_var) tend to be satisfiable
        dataset = LinearF2Dataset(n_samples=2000, n_vars=10, n_equations=10)

        # Square random systems over F2 should be roughly 20-80% satisfiable
        # The exact ratio depends on the structure of random binary matrices
        assert 0.2 < dataset.satisfiable_ratio < 0.8, \
            f"Expected ~20-80% satisfiable for square system, got {dataset.satisfiable_ratio:.1%}"

    def test_input_dim_property(self):
        """input_dim property should return correct value."""
        dataset = LinearF2Dataset(n_samples=10, n_vars=8, n_equations=12)

        assert dataset.input_dim == 12 * (8 + 1)  # n_eq * (n_var + 1)


class TestCreateDataloaders:
    """Tests for the create_dataloaders function."""

    def test_creates_both_loaders(self):
        """Should create both train and eval loaders."""
        train_loader, eval_loader = create_dataloaders(
            train_size=100,
            eval_size=50,
            batch_size=16,
        )

        assert train_loader is not None
        assert eval_loader is not None

    def test_correct_sizes(self):
        """Loaders should have correct dataset sizes."""
        train_loader, eval_loader = create_dataloaders(
            train_size=100,
            eval_size=50,
        )

        assert len(train_loader.dataset) == 100
        assert len(eval_loader.dataset) == 50

    def test_batch_shapes(self):
        """Batches should have correct shapes."""
        train_loader, _ = create_dataloaders(
            n_vars=5,
            n_equations=8,
            train_size=100,
            batch_size=16,
        )

        x, y = next(iter(train_loader))

        expected_dim = 8 * (5 + 1)
        assert x.shape == (16, expected_dim)
        assert y.shape == (16,)

    def test_different_seeds_for_train_eval(self):
        """Train and eval should use different seeds to avoid overlap."""
        train_loader, eval_loader = create_dataloaders(
            train_size=50,
            eval_size=50,
            seed=42,
        )

        train_x = next(iter(train_loader))[0]
        eval_x = next(iter(eval_loader))[0]

        # Should be different (very high probability)
        assert not torch.equal(train_x, eval_x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
