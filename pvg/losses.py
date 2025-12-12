"""
pvg/losses.py

Helper to compute losses and metrics together for one train/eval step.
"""

import torch
import torch.nn.functional as F


def compute_step_metrics(
    logits: torch.Tensor,
    true_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute verifier loss, prover loss, and basic metrics.

    Args:
        logits: shape (batch_size,) — verifier outputs (logits)
        true_labels: ground-truth labels for evaluation (binary, shape (batch_size,))
        noisy_labels: noisy (potentially flipped) labels for training

    Returns:
        verifier_loss (noisy), prover_loss, metrics_dict
    """
    # Verifier loss (cross-entropy, noisy labels)
    v_loss = F.binary_cross_entropy_with_logits(logits, noisy_labels.float())
    # Prover loss (always defending label 1)
    p_loss = F.softplus(-logits).mean()
    # Verifier loss (w.r.t. clean labels, for eval/plotting only)
    v_loss_clean = F.binary_cross_entropy_with_logits(logits, true_labels.float())

    # Metrics (non-differentiable)
    with torch.no_grad():
        preds = (logits > 0).long()
        accuracy = (preds == true_labels).float().mean()
        prover_success_rate = preds.float().mean()  # Rate of predicting 1

    metrics = {
        "verifier_loss_noisy": v_loss.item(),
        "verifier_loss_clean": v_loss_clean.item(),
        "prover_loss": p_loss.item(),
        "accuracy": accuracy.item(),
        "prover_success_rate": prover_success_rate.item(),
    }

    return v_loss, p_loss, metrics
