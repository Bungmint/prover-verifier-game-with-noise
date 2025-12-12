"""
src/noise.py

Label noise injection utilities.
"""

import torch


def inject_label_noise(
    labels: torch.Tensor, epsilon: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """
    Flip binary labels with probability epsilon (symmetric noise).

    Args:
        labels: Binary tensor of shape (batch_size,) with values in {0, 1}
        epsilon: Flip probability in [0, 0.5)
        generator: Optional RNG for reproducibility
    Returns:
        Noisy labels, same shape as input
    """
    if not 0 <= epsilon < 0.5:
        raise ValueError(f"epsilon must be in [0, 0.5), got {epsilon}")
    if epsilon == 0:
        return labels.clone()

    flip_mask = torch.rand(labels.shape, device=labels.device, generator=generator) < epsilon
    noisy_labels = labels ^ flip_mask.to(labels.dtype)
    return noisy_labels
