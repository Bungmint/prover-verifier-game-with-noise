# tests/test_losses.py
import torch
import torch.nn.functional as F


def test_prover_loss_gradient():
    """Verify prover loss gradients have correct sign."""
    logits = torch.zeros(32, requires_grad=True)
    
    # Prover loss: -log(sigmoid(x)) = softplus(-x)
    loss = F.softplus(-logits).mean()
    loss.backward()
    
    # Gradient should be NEGATIVE: increasing logit → decreasing loss
    # d/dx [softplus(-x)] = -sigmoid(-x) < 0
    assert (logits.grad < 0).all(), "Prover loss gradient should be negative"
    print("✓ Prover wants to INCREASE logits (make verifier predict 1)")


def test_verifier_loss_gradient():
    """Verify verifier loss gradients flow correctly."""
    logits = torch.zeros(32, requires_grad=True)
    labels = torch.randint(0, 2, (32,))
    
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    loss.backward()
    
    # Gradient sign depends on label:
    # - label=1: gradient < 0 (want to increase logit)
    # - label=0: gradient > 0 (want to decrease logit)
    assert logits.grad is not None
    assert ((labels == 1) == (logits.grad < 0)).all()
    print("✓ Verifier pushes logits toward correct class")


def test_loss_values_at_extremes():
    """Verify loss values at extremes make sense."""
    # Very positive logit → prover happy (low loss)
    high_logit = torch.tensor([10.0])
    p_loss_high = F.softplus(-high_logit).item()
    assert p_loss_high < 0.001, f"Expected ~0, got {p_loss_high}"
    
    # Very negative logit → prover unhappy (high loss)  
    low_logit = torch.tensor([-10.0])
    p_loss_low = F.softplus(-low_logit).item()
    assert p_loss_low > 9.0, f"Expected ~10, got {p_loss_low}"
    
    print("✓ Prover loss: high logit → low loss, low logit → high loss")


def test_softplus_equivalence():
    """Verify softplus(-x) ≈ -log(sigmoid(x))."""
    x = torch.randn(100)
    
    via_softplus = F.softplus(-x)
    via_log_sigmoid = -torch.log(torch.sigmoid(x) + 1e-10)  # Add eps for stability
    
    assert torch.allclose(via_softplus, via_log_sigmoid, atol=1e-5)
    print("✓ softplus(-x) == -log(sigmoid(x))")


if __name__ == "__main__":
    test_prover_loss_gradient()
    test_verifier_loss_gradient()
    test_loss_values_at_extremes()
    test_softplus_equivalence()
