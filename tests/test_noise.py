import torch
from pvg.noise import inject_label_noise

def test_noise_statistics():
    """Verify noise injection matches expected flip rate."""
    torch.manual_seed(42)
    labels = torch.ones(10000, dtype=torch.long)
    
    for eps in [0.0, 0.1, 0.2, 0.3]:
        noisy = inject_label_noise(labels, eps)
        actual_flip_rate = (noisy != labels).float().mean().item()
        
        # Allow some statistical variance
        assert abs(actual_flip_rate - eps) < 0.02, \
            f"Expected flip rate {eps}, got {actual_flip_rate}"
    
    print("✓ Noise injection statistics correct")

if __name__ == "__main__":
    test_noise_statistics()