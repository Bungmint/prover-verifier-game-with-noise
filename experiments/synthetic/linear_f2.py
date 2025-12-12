"""Linear equations over F_2 (binary field) dataset."""

from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def gaussian_elimination_f2(A: np.ndarray, b: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if Ax = b has a solution over F_2 using Gaussian elimination."""
    A = A.copy().astype(np.int64)
    b = b.copy().astype(np.int64)
    n_eq, n_var = A.shape

    aug = np.hstack([A, b.reshape(-1, 1)])

    pivot_row = 0
    pivot_cols = []

    for col in range(n_var):
        pivot_found = False
        for row in range(pivot_row, n_eq):
            if aug[row, col] == 1:
                aug[[pivot_row, row]] = aug[[row, pivot_row]]
                pivot_found = True
                break

        if not pivot_found:
            continue

        for row in range(n_eq):
            if row != pivot_row and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[pivot_row]) % 2

        pivot_cols.append(col)
        pivot_row += 1

    # Check for contradiction: [0 0 ... 0 | 1]
    for row in range(pivot_row, n_eq):
        if aug[row, -1] == 1:
            return False, None

    solution = np.zeros(n_var, dtype=np.int64)
    for i in range(len(pivot_cols) - 1, -1, -1):
        solution[pivot_cols[i]] = aug[i, -1]

    return True, solution


class LinearF2Dataset(Dataset):
    """Dataset of linear systems over F_2. Label is 1 if satisfiable."""

    def __init__(
        self,
        n_samples: int,
        n_vars: int = 10,
        n_equations: int = 15,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.samples: list[Tuple[torch.Tensor, torch.Tensor]] = []
        self.n_vars = n_vars
        self.n_equations = n_equations

        n_satisfiable = 0

        for _ in range(n_samples):
            A = rng.integers(0, 2, size=(n_equations, n_vars))
            b = rng.integers(0, 2, size=n_equations)

            is_sat, _ = gaussian_elimination_f2(A, b)

            x = np.concatenate([A.flatten(), b])
            y = int(is_sat)

            self.samples.append((
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            ))

            if is_sat:
                n_satisfiable += 1

        self.satisfiable_ratio = n_satisfiable / n_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

    @property
    def input_dim(self) -> int:
        return self.n_equations * (self.n_vars + 1)


def create_dataloaders(
    n_vars: int = 10,
    n_equations: int = 15,
    train_size: int = 5000,
    eval_size: int = 1000,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and eval dataloaders."""
    train_dataset = LinearF2Dataset(
        n_samples=train_size,
        n_vars=n_vars,
        n_equations=n_equations,
        seed=seed,
    )

    eval_dataset = LinearF2Dataset(
        n_samples=eval_size,
        n_vars=n_vars,
        n_equations=n_equations,
        seed=seed + 1000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, eval_loader


if __name__ == "__main__":
    dataset = LinearF2Dataset(n_samples=1000, n_vars=10, n_equations=15)
    print(f"Samples: {len(dataset)}, Satisfiable: {dataset.satisfiable_ratio:.1%}")
    x, y = dataset[0]
    print(f"Input shape: {x.shape}, Label: {y.item()}")
