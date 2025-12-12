# Noisy Prover-Verifier Games: Step-by-Step Implementation Walkthrough

**For researchers who want to implement this project from scratch**

This guide walks you through implementing the Noisy PVG framework step by step, explaining *why* each decision is made. Follow it sequentially—each section builds on the previous.

---

## Table of Contents

1. [Understanding What We're Building](#step-1-understanding-what-were-building)
2. [Project Setup](#step-2-project-setup)
3. [Implementing the Noise Model](#step-3-implementing-the-noise-model)
4. [Implementing Loss Functions](#step-4-implementing-loss-functions)
5. [Building the Models](#step-5-building-the-models)
6. [Implementing the Training Loop](#step-6-implementing-the-training-loop)
7. [Creating a Synthetic Dataset](#step-7-creating-a-synthetic-dataset)
8. [Running Your First Experiment](#step-8-running-your-first-experiment)
9. [Adding Evaluation and Metrics](#step-9-adding-evaluation-and-metrics)
10. [Running the Noise Sweep](#step-10-running-the-noise-sweep)
11. [Extending to NLP Tasks](#step-11-extending-to-nlp-tasks)
12. [Scaling to LLMs](#step-12-scaling-to-llms)
13. [Debugging and Validation](#step-13-debugging-and-validation)

---

## Step 1: Understanding What We're Building

**Goal:** Internalize the core concepts before writing any code.

### 1.1 The Research Question

We're investigating: **How does label noise affect the equilibrium of Prover-Verifier Games?**

Concretely:
- If we train a PVG system on labels corrupted with flip probability ε, how far does the resulting equilibrium deviate from the clean-label equilibrium?
- At what noise level does the system "break down"?

### 1.2 The Game Setup

```
Input x → [Prover P_φ] → message z → [Verifier V_θ] → prediction ŷ
                ↓                            ↓
            "generate proof"          "judge proof"
```

**Key insight:** The verifier sees BOTH the input x AND the prover's message z. This is different from standard supervised learning where a model only sees x.

### 1.3 Why Stackelberg, Not Nash?

This is the **most important conceptual distinction** in the project.

| Aspect | What it means | Why it matters |
|--------|---------------|----------------|
| **Verifier leads** | Verifier's parameters are "published" first | Matches real deployment: verification systems exist, provers adapt to them |
| **Prover best-responds** | Prover fully optimizes against fixed verifier | This is bilevel optimization, not simultaneous gradient descent |
| **Sequential, not simultaneous** | Inner loop (prover) completes before outer loop (verifier) | Getting this wrong gives you Nash equilibrium, which is a different game |

**Implementation implication:** You cannot just alternate gradient steps. The prover must converge (or take K steps) before the verifier updates.

### 1.4 The Noise Model

We use **symmetric label noise**: both classes flip with probability ε.

```python
# Ground truth y ∈ {0, 1}
# Noisy label ỹ:
P(ỹ = 1 | y = 0) = ε    # false positive
P(ỹ = 0 | y = 1) = ε    # false negative
```

**Why symmetric?**
- Preserves class balance (important for binary classification)
- Admits closed-form analysis: the "effective label" is `y_eff = (1-2ε)y + ε`
- Simpler to reason about theoretically

**Sanity check:** At ε = 0.5, labels are pure noise (50% chance of either class regardless of true label). At ε = 0, labels are clean.

### 1.5 Loss Functions

**Verifier loss:** Standard cross-entropy, but trained on noisy labels ỹ:
```
L_V(θ, φ) = E[-ỹ log p_θ(x,z) - (1-ỹ) log(1-p_θ(x,z))]
```

**Prover loss:** Always tries to maximize probability of label 1:
```
L_P(θ, φ) = E[-log p_θ(ŷ=1|x,z)]
```

**Why does the prover defend label 1?** This is Anil et al.'s setup. It simplifies analysis—there's only one prover strategy to characterize. The prover is adversarial: it tries to convince the verifier to output 1 regardless of the true label.

**✓ Checkpoint:** Before proceeding, make sure you can explain:
- Why we use Stackelberg, not Nash
- What the prover is trying to do (maximize p(ŷ=1))
- What the verifier is trying to do (predict correctly, but only sees noisy labels)

---

## Step 2: Project Setup

**Goal:** Create a clean project structure that will scale.

### 2.1 Create Directory Structure

```bash
mkdir -p prover-verifier-game-with-noise/{pvg,experiments/{synthetic,nlp,math_reasoning},notebooks,tests,checkpoints,results}
cd prover-verifier-game-with-noise
touch pvg/__init__.py
touch pvg/{noise,losses,models,training,metrics,utils,config}.py
```

**Rationale:** 
- `pvg/` is a proper Python package (not `src/`) that can be installed with `pip install -e .`
- Using a package name (`pvg`) instead of generic `src/` allows clean imports like `from pvg import losses`
- `experiments/` contains task-specific scripts (keeps experiments isolated)
- Separating by tier (synthetic → nlp → math) lets you validate on simple tasks first

### 2.2 Create `pyproject.toml`

We use `uv` for fast, reliable dependency management. Create `pyproject.toml`:

```toml
[project]
name = "prover-verifier-game-with-noise"
version = "0.1.0"
description = "Investigating how noisy labels affect Prover-Verifier Game equilibria"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    # Core
    "torch>=2.0.0",
    "numpy>=1.24.0",
    
    # NLP (Tier 2)
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    
    # RL Training (Tier 3)
    "trl>=0.7.0",
    "accelerate>=0.24.0",
    
    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    
    # Utilities
    "tqdm>=4.65.0",
    "scipy>=1.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]
```

**Why uv?**
- 10-100x faster than pip
- Deterministic resolution with lockfile
- Built-in virtual environment management
- Drop-in replacement for pip/pip-tools/virtualenv

### 2.3 Set Up Environment

```bash
# Install uv if you haven't (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in EDITABLE mode - this is critical!
# The -e flag makes 'pvg' importable from anywhere while you develop
uv pip install -e ".[dev]"

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verify pvg package is importable
python -c "from pvg import losses; print('✓ pvg package installed')"
```

**Alternative: Use uv's project management (recommended)**

```bash
# Let uv manage everything - creates .venv automatically
uv sync

# Run commands through uv (auto-activates venv)
uv run python -c "import torch; print(torch.__version__)"
uv run pytest tests/
```

**For GPU support**, install PyTorch with CUDA separately:

```bash
# After uv sync, add CUDA-enabled PyTorch
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Why editable install (`-e`)?** 
- Makes `pvg` importable from anywhere (no `sys.path` hacks needed)
- Changes to code are immediately reflected without reinstalling
- Works seamlessly with notebooks, experiments, and tests

**✓ Checkpoint:** You should have a working Python environment with PyTorch installed and `pvg` importable.

---

## Step 3: Implementing the Noise Model

**Goal:** Create utilities for injecting label noise.

**File:** `pvg/noise.py`

### 3.1 Core Noise Injection Function

```python
"""
pvg/noise.py

Label noise injection utilities.
"""

import torch
from typing import Optional

def inject_label_noise(
    labels: torch.Tensor,
    epsilon: float,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Flip binary labels with probability epsilon (symmetric noise).
    
    Args:
        labels: Binary tensor of shape (batch_size,) with values in {0, 1}
        epsilon: Flip probability in [0, 0.5)
        generator: Optional RNG for reproducibility
    
    Returns:
        Noisy labels, same shape as input
    
    Rationale:
        - epsilon < 0.5 ensures labels still carry signal
        - At epsilon = 0.5, labels become pure noise (uninformative)
        - We use XOR for efficient flipping: label ^ flip_mask
    """
    if not 0 <= epsilon < 0.5:
        raise ValueError(f"epsilon must be in [0, 0.5), got {epsilon}")
    
    if epsilon == 0:
        return labels.clone()
    
    # Generate random mask: True where we flip
    flip_mask = torch.rand(
        labels.shape,
        device=labels.device,
        generator=generator
    ) < epsilon
    
    # XOR flips the label where mask is True
    noisy_labels = labels ^ flip_mask.long()
    
    return noisy_labels
```

**Why XOR?** For binary labels, `label XOR flip_mask` gives:
- `0 XOR 1 = 1` (flip)
- `1 XOR 1 = 0` (flip)
- `0 XOR 0 = 0` (no change)
- `1 XOR 0 = 1` (no change)

This is cleaner than `(1 - label) * flip_mask + label * (1 - flip_mask)`.

### 3.2 Effective Label (for Theoretical Understanding)

```python
def compute_effective_label(y: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Compute the expected label under symmetric noise.
    
    Under symmetric noise:
        E[ỹ | y] = (1 - ε)y + ε(1 - y) = (1 - 2ε)y + ε
    
    This is useful for:
        - Understanding gradient behavior (gradients scale by (1-2ε))
        - Implementing soft-label training variants
        - Verifying theoretical predictions
    """
    return (1 - 2 * epsilon) * y.float() + epsilon
```

**Insight:** The effective signal strength scales as `(1 - 2ε)`. At ε = 0.25, you've lost half your signal.

### 3.3 Test Your Implementation

Create `tests/test_noise.py`:

```python
import torch
from pvg.noise import inject_label_noise, compute_effective_label

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

def test_effective_label():
    """Verify effective label formula."""
    y = torch.tensor([0.0, 1.0])
    
    # At eps=0, effective label equals true label
    assert torch.allclose(compute_effective_label(y, 0.0), y)
    
    # At eps=0.5, effective label is 0.5 for both classes
    assert torch.allclose(compute_effective_label(y, 0.5), torch.tensor([0.5, 0.5]))
    
    print("✓ Effective label formula correct")

if __name__ == "__main__":
    test_noise_statistics()
    test_effective_label()
```

Run: `uv run pytest tests/test_noise.py -v`

**✓ Checkpoint:** Noise injection should flip ~10% of labels when ε=0.1.

---

## Step 4: Understanding and Using Loss Functions

**Goal:** Understand the loss functions and how to use them inline.

### 4.1 The Two Losses

In PVG, there are two distinct losses—one for each player:

| Player | Loss | PyTorch Implementation |
|--------|------|------------------------|
| **Verifier** | Cross-entropy with (noisy) labels | `F.binary_cross_entropy_with_logits(logits, labels.float())` |
| **Prover** | `-log p(ŷ=1)` (always defends 1) | `F.softplus(-logits).mean()` |

**Key insight:** The verifier loss is standard supervised learning. The prover loss is unusual—it doesn't take labels because the prover always tries to maximize the probability of class 1, regardless of the true label.

### 4.2 Verifier Loss (Standard Cross-Entropy)

The verifier minimizes cross-entropy with (potentially noisy) labels:

```python
import torch.nn.functional as F

# In your training loop:
logits = verifier(x, z)  # Shape: (batch_size,)
v_loss = F.binary_cross_entropy_with_logits(logits, noisy_labels.float())
```

**Why BCE with logits?** It's numerically stable—combines sigmoid and cross-entropy in one step, avoiding log(0) issues.

**Rationale:** The verifier trains on noisy labels `ỹ`. The equilibrium deviation we study comes from this noise—the verifier learns a "corrupted" decision boundary.

### 4.3 Prover Loss (Maximize p(ŷ=1))

The prover minimizes `-log p(ŷ=1)`, equivalently maximizing the probability the verifier outputs 1:

```python
# In your training loop:
logits = verifier(x, z)
p_loss = F.softplus(-logits).mean()
```

**Why `softplus(-x)`?** This is mathematically equivalent to `-log(sigmoid(x))`:

```
-log(sigmoid(x)) = -log(1/(1+exp(-x))) = log(1+exp(-x)) = softplus(-x)
```

Using `softplus` is numerically stable for extreme logits (avoids overflow/underflow).

**Rationale:**
- The prover is adversarial: it tries to convince the verifier regardless of truth
- This creates the game-theoretic tension
- The verifier must learn to resist deceptive proofs
- Note: the prover does NOT see the labels!

### 4.4 Optional: Metrics Helper

If you want to bundle loss computation with metrics tracking, you can create a helper (but this is optional—you can compute metrics inline too):

**File:** `pvg/losses.py`

```python
"""
pvg/losses.py

Optional helper for computing losses and metrics together.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_step_metrics(
    logits: torch.Tensor,
    true_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute losses and metrics in one call.
    
    Args:
        logits: Verifier output, shape (batch_size,)
        true_labels: Ground truth labels (for evaluation)
        noisy_labels: Noisy labels (for training)
    
    Returns:
        (verifier_loss, prover_loss, metrics_dict)
    """
    # Losses (using inline PyTorch functions)
    v_loss = F.binary_cross_entropy_with_logits(logits, noisy_labels.float())
    p_loss = F.softplus(-logits).mean()
    v_loss_clean = F.binary_cross_entropy_with_logits(logits, true_labels.float())
    
    # Metrics (no gradient needed)
    with torch.no_grad():
        preds = (logits > 0).long()
        accuracy = (preds == true_labels).float().mean()
        prover_success = preds.float().mean()  # Rate of predicting 1
    
    metrics = {
        "verifier_loss_noisy": v_loss.item(),
        "verifier_loss_clean": v_loss_clean.item(),
        "prover_loss": p_loss.item(),
        "accuracy": accuracy.item(),
        "prover_success_rate": prover_success.item(),
    }
    
    return v_loss, p_loss, metrics
```

### 4.5 Test Your Understanding

Create `tests/test_losses.py` to verify the loss behavior:

```python
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
```

Run: `uv run pytest tests/test_losses.py -v`

### 4.6 Summary: What to Use Where

In your training loop, just use these directly:

```python
# Verifier step (trains on noisy labels)
logits = verifier(x, z)
v_loss = F.binary_cross_entropy_with_logits(logits, noisy_labels.float())
v_loss.backward()
verifier_optimizer.step()

# Prover step (no labels needed!)
logits = verifier(x, z)
p_loss = F.softplus(-logits).mean()
p_loss.backward()
prover_optimizer.step()
```

**✓ Checkpoint:** You should understand why the prover loss doesn't use labels, and why `softplus(-x)` is used instead of `-log(sigmoid(x))`.

---

## Step 5: Building the Models

**Goal:** Create prover and verifier neural networks.

**File:** `pvg/models.py`

### 5.1 Design Philosophy

```
Prover: x → z (generates a "message" or "proof")
Verifier: (x, z) → ŷ (makes decision given input AND proof)
```

**Key architectural decisions:**
1. **Message is continuous:** For MLPs, the message z is a vector in [-1, 1]^d
2. **Verifier sees both x and z:** Concatenate before classification
3. **Models are decoupled:** Different architectures for different tiers

### 5.2 MLP Prover (Tier 1)

```python
"""
pvg/models.py

Prover and Verifier model architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple

class BaseProver(nn.Module, ABC):
    """Abstract base for all provers."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate message z given input x."""
        pass

class BaseVerifier(nn.Module, ABC):
    """Abstract base for all verifiers."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Produce logits given input x and message z."""
        pass


class MLPProver(BaseProver):
    """
    MLP prover for synthetic tasks.
    
    Architecture: x → [MLP] → tanh → z
    
    The tanh bounds messages to [-1, 1], which:
    - Prevents unbounded messages
    - Creates a compact message space
    - Makes optimization easier
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        message_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, message_dim))
        layers.append(nn.Tanh())  # Bound output to [-1, 1]
        
        self.network = nn.Sequential(*layers)
        self.message_dim = message_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed (e.g., for matrix inputs)
        x_flat = x.view(x.shape[0], -1)
        return self.network(x_flat)
```

### 5.3 MLP Verifier (Tier 1)

```python
class MLPVerifier(BaseVerifier):
    """
    MLP verifier for synthetic tasks.
    
    Architecture: concat(x, z) → [MLP] → logit
    
    Critical: The verifier MUST use the message z.
    Concatenation ensures z influences the decision.
    """
    
    def __init__(
        self,
        input_dim: int,
        message_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim + message_dim  # Concatenate x and z
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))  # Single logit for binary
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        combined = torch.cat([x_flat, z], dim=-1)
        return self.network(combined).squeeze(-1)
```

### 5.4 Model Factory

```python
def create_models(config) -> Tuple[BaseProver, BaseVerifier]:
    """
    Factory function to create prover-verifier pair.
    
    Rationale: Centralizes model creation, makes it easy to
    switch between tiers without changing experiment code.
    """
    if config.tier == "synthetic":
        # Calculate input dimension from problem parameters
        input_dim = config.n_equations * (config.n_vars + 1)
        
        prover = MLPProver(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            message_dim=config.message_dim,
            num_layers=config.num_layers
        )
        verifier = MLPVerifier(
            input_dim=input_dim,
            message_dim=config.message_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )
        return prover, verifier
    
    # Add other tiers later
    raise ValueError(f"Unknown tier: {config.tier}")
```

### 5.5 Test Your Models

```python
# tests/test_models.py
import torch
from pvg.models import MLPProver, MLPVerifier

def test_prover_output_shape():
    """Verify prover produces correct message shape."""
    prover = MLPProver(input_dim=100, message_dim=64)
    x = torch.randn(32, 100)
    z = prover(x)
    
    assert z.shape == (32, 64), f"Expected (32, 64), got {z.shape}"
    assert (z >= -1).all() and (z <= 1).all(), "Messages should be in [-1, 1]"
    print("✓ Prover output shape and bounds correct")

def test_verifier_output_shape():
    """Verify verifier produces correct logit shape."""
    verifier = MLPVerifier(input_dim=100, message_dim=64)
    x = torch.randn(32, 100)
    z = torch.randn(32, 64)
    logits = verifier(x, z)
    
    assert logits.shape == (32,), f"Expected (32,), got {logits.shape}"
    print("✓ Verifier output shape correct")

def test_gradients_flow():
    """Verify gradients flow through both models."""
    prover = MLPProver(input_dim=10, message_dim=8)
    verifier = MLPVerifier(input_dim=10, message_dim=8)
    
    x = torch.randn(16, 10)
    z = prover(x)
    logits = verifier(x, z)
    loss = logits.mean()
    loss.backward()
    
    # Check prover has gradients
    prover_has_grad = any(p.grad is not None for p in prover.parameters())
    assert prover_has_grad, "Prover should have gradients"
    
    # Check verifier has gradients
    verifier_has_grad = any(p.grad is not None for p in verifier.parameters())
    assert verifier_has_grad, "Verifier should have gradients"
    
    print("✓ Gradients flow through both models")

if __name__ == "__main__":
    test_prover_output_shape()
    test_verifier_output_shape()
    test_gradients_flow()
```

**✓ Checkpoint:** Models should produce correct shapes and gradients should flow.

---

## Step 6: Implementing the Training Loop

**Goal:** Implement Stackelberg (bilevel) training.

**File:** `pvg/training.py`

### 6.1 The Algorithm

```
for round t = 1, ..., T:
    # INNER LOOP: Prover best-responds to current verifier
    freeze(verifier)
    for k = 1, ..., K_P:
        z = prover(x)
        L_P = -log p_θ(ŷ=1|x,z)
        update prover to minimize L_P
    
    # OUTER LOOP: Verifier optimizes against prover's best response
    freeze(prover)
    for k = 1, ..., K_V:
        z = prover(x)  # Use current (frozen) prover
        L_V = CE(p_θ(x,z), ỹ)  # Train on NOISY labels
        update verifier to minimize L_V
```

**Critical:** The order matters! Prover optimizes first (inner), then verifier (outer).

### 6.2 Implementation

```python
"""
pvg/training.py

Stackelberg training loop for Prover-Verifier Games.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
from tqdm import tqdm

from pvg.noise import inject_label_noise

logger = logging.getLogger(__name__)


class StackelbergTrainer:
    """
    Trainer implementing bilevel Stackelberg optimization.
    
    Key insight: This is NOT alternating gradient descent (Nash).
    The prover fully best-responds before the verifier updates.
    """
    
    def __init__(
        self,
        prover: torch.nn.Module,
        verifier: torch.nn.Module,
        config,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ):
        self.prover = prover
        self.verifier = verifier
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        # Create infinite iterator for sampling batches
        self._train_iter = None
        
        # Separate optimizers (critical for bilevel opt)
        self.prover_optimizer = AdamW(
            prover.parameters(),
            lr=config.prover_lr,
            weight_decay=0.01
        )
        self.verifier_optimizer = AdamW(
            verifier.parameters(),
            lr=config.verifier_lr,
            weight_decay=0.01
        )
        
        # Device setup
        self.device = torch.device(config.device)
        self.prover.to(self.device)
        self.verifier.to(self.device)
        
        # History tracking
        self.history = {"train": [], "eval": []}
    
    def _get_batch(self) -> tuple:
        """Get next batch, cycling through dataset."""
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        
        x, y = batch
        return x.to(self.device), y.to(self.device)
    
    def train(self) -> Dict:
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting Stackelberg training: ε={self.config.epsilon}")
        
        for round_t in tqdm(range(self.config.num_rounds), desc="Training"):
            
            # ═══════════════════════════════════════════════════════════
            # STEP 1: PROVER BEST-RESPONSE (Inner Loop)
            # ═══════════════════════════════════════════════════════════
            # 
            # The prover optimizes against a FIXED verifier.
            # This simulates the prover "adapting" to the deployed verifier.
            #
            self.verifier.eval()
            for param in self.verifier.parameters():
                param.requires_grad = False
            
            self.prover.train()
            for param in self.prover.parameters():
                param.requires_grad = True
            
            prover_losses = []
            for _ in range(self.config.prover_steps_per_round):
                p_loss = self._prover_step()
                prover_losses.append(p_loss)
            
            # ═══════════════════════════════════════════════════════════
            # STEP 2: VERIFIER OPTIMIZATION (Outer Loop)
            # ═══════════════════════════════════════════════════════════
            #
            # The verifier optimizes against the prover's BEST RESPONSE.
            # The prover is now frozen—verifier sees how good the prover got.
            #
            self.prover.eval()
            for param in self.prover.parameters():
                param.requires_grad = False
            
            self.verifier.train()
            for param in self.verifier.parameters():
                param.requires_grad = True
            
            verifier_losses = []
            for _ in range(self.config.verifier_steps_per_round):
                v_loss = self._verifier_step()
                verifier_losses.append(v_loss)
            
            # ═══════════════════════════════════════════════════════════
            # LOGGING AND EVALUATION
            # ═══════════════════════════════════════════════════════════
            
            if round_t % self.config.log_every == 0:
                self._log_round(round_t, prover_losses, verifier_losses)
            
            if round_t % self.config.eval_every == 0 and self.eval_loader:
                eval_metrics = self.evaluate()
                self.history["eval"].append({"round": round_t, **eval_metrics})
                logger.info(f"  Eval: acc={eval_metrics['accuracy']:.3f}, clean_loss={eval_metrics['clean_loss']:.4f}")
        
        return self.history
    
    def _prover_step(self) -> float:
        """
        Single prover gradient step.
        
        Prover minimizes L_P = -log p_θ(ŷ=1|x,z)
        """
        x, _ = self._get_batch()  # Prover doesn't see labels!
        
        # Forward pass: prover generates message
        z = self.prover(x)
        
        # Verifier evaluates (but we don't backprop through it)
        logits = self.verifier(x, z)
        
        # Compute prover loss: -log p(ŷ=1) = softplus(-logits)
        loss = F.softplus(-logits).mean()
        
        # Backward pass (only updates prover)
        self.prover_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.prover.parameters(), 1.0)
        self.prover_optimizer.step()
        
        return loss.item()
    
    def _verifier_step(self) -> float:
        """
        Single verifier gradient step.
        
        Verifier minimizes L_V = CE(p_θ(x,z), ỹ) with noisy labels.
        """
        x, y = self._get_batch()
        
        # Inject noise into labels
        noisy_y = inject_label_noise(y, self.config.epsilon)
        
        # Prover generates message (detached—prover is frozen)
        with torch.no_grad():
            z = self.prover(x)
        
        # Verifier forward pass
        logits = self.verifier(x, z)
        
        # Compute loss on NOISY labels (standard BCE)
        loss = F.binary_cross_entropy_with_logits(logits, noisy_y.float())
        
        # Backward pass (only updates verifier)
        self.verifier_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), 1.0)
        self.verifier_optimizer.step()
        
        return loss.item()
    
    def evaluate(self) -> Dict:
        """
        Evaluate on CLEAN labels (no noise).
        
        This measures true performance of the learned equilibrium.
        """
        self.prover.eval()
        self.verifier.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_prover_wins = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                z = self.prover(x)
                logits = self.verifier(x, z)
                
                # Clean loss (no noise!)
                loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="sum")
                total_loss += loss.item()
                
                # Predictions
                preds = (logits > 0).long()
                total_correct += (preds == y).sum().item()
                total_prover_wins += preds.sum().item()
                total_samples += len(y)
        
        return {
            "clean_loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "prover_success_rate": total_prover_wins / total_samples
        }
    
    def _log_round(self, round_t, prover_losses, verifier_losses):
        """Log training progress."""
        avg_p = sum(prover_losses) / len(prover_losses)
        avg_v = sum(verifier_losses) / len(verifier_losses)
        
        self.history["train"].append({
            "round": round_t,
            "prover_loss": avg_p,
            "verifier_loss": avg_v
        })
        
        logger.info(f"Round {round_t}: P_loss={avg_p:.4f}, V_loss={avg_v:.4f}")
```

### 6.3 Configuration Dataclass

Create `pvg/config.py`:

```python
"""
pvg/config.py

Configuration dataclasses for experiments.
"""

from dataclasses import dataclass, field
from typing import Literal
import torch

@dataclass
class ExperimentConfig:
    """Base configuration for all experiments."""
    
    # Experiment identification
    experiment_name: str = "default"
    seed: int = 42
    
    # Noise level (primary independent variable)
    epsilon: float = 0.0
    
    # Training loop
    num_rounds: int = 100
    prover_steps_per_round: int = 10
    verifier_steps_per_round: int = 5
    
    # Optimization
    prover_lr: float = 1e-4
    verifier_lr: float = 1e-4
    batch_size: int = 64
    
    # Logging
    log_every: int = 10
    eval_every: int = 20
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SyntheticConfig(ExperimentConfig):
    """Config for Tier 1 synthetic experiments."""
    
    tier: Literal["synthetic"] = "synthetic"
    
    # Problem parameters
    n_vars: int = 10
    n_equations: int = 15
    dataset_size: int = 5000
    
    # Model architecture
    hidden_dim: int = 128
    message_dim: int = 64
    num_layers: int = 2
```

**✓ Checkpoint:** You should be able to instantiate a `StackelbergTrainer`. Test with dummy data.

---

## Step 7: Creating a Synthetic Dataset

**Goal:** Create a dataset where we know the ground truth labels.

**File:** `experiments/synthetic/linear_f2.py`

### 7.1 The Problem: Linear Equations over F₂

We use **linear systems over the binary field F₂** (integers mod 2):
- Input: A matrix A and vector b (both binary)
- Label: 1 if Ax = b has a solution, 0 otherwise

**Why this problem?**
- Ground truth is computable (Gaussian elimination)
- Non-trivial: random systems are satisfiable ~50% of the time
- Natural PVG interpretation: prover could send a "witness" (candidate solution)

### 7.2 Dataset Implementation

```python
"""
experiments/synthetic/linear_f2.py

Linear equations over F_2 (binary field).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


def gaussian_elimination_f2(A: np.ndarray, b: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Check if linear system Ax = b has a solution over F_2.
    
    Uses Gaussian elimination with pivoting.
    
    Returns:
        (is_satisfiable, solution_if_exists)
    """
    A = A.copy().astype(np.int64)
    b = b.copy().astype(np.int64)
    n_eq, n_var = A.shape
    
    # Augmented matrix [A | b]
    aug = np.hstack([A, b.reshape(-1, 1)])
    
    pivot_row = 0
    for col in range(n_var):
        # Find a pivot (non-zero entry)
        pivot_found = False
        for row in range(pivot_row, n_eq):
            if aug[row, col] == 1:
                # Swap rows
                aug[[pivot_row, row]] = aug[[row, pivot_row]]
                pivot_found = True
                break
        
        if not pivot_found:
            continue
        
        # Eliminate other rows
        for row in range(n_eq):
            if row != pivot_row and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[pivot_row]) % 2
        
        pivot_row += 1
    
    # Check for contradiction: row of form [0 0 ... 0 | 1]
    for row in range(pivot_row, n_eq):
        if aug[row, -1] == 1:
            return False, None
    
    # System is satisfiable; extract a solution
    solution = np.zeros(n_var, dtype=np.int64)
    for row in range(min(pivot_row, n_eq) - 1, -1, -1):
        col = np.argmax(aug[row, :n_var])
        if aug[row, col] == 1:
            solution[col] = (aug[row, -1] - np.dot(aug[row, :n_var], solution)) % 2
    
    return True, solution


class LinearF2Dataset(Dataset):
    """
    Dataset of linear systems over F_2.
    
    Each sample: (flattened [A, b], satisfiability label)
    """
    
    def __init__(
        self,
        n_samples: int,
        n_vars: int = 10,
        n_equations: int = 15,
        seed: int = 42
    ):
        np.random.seed(seed)
        
        self.samples = []
        self.n_vars = n_vars
        self.n_equations = n_equations
        
        for _ in range(n_samples):
            # Random binary matrix and vector
            A = np.random.randint(0, 2, (n_equations, n_vars))
            b = np.random.randint(0, 2, n_equations)
            
            # Compute ground truth
            is_sat, _ = gaussian_elimination_f2(A, b)
            
            # Flatten for model input: [A_flat, b]
            x = np.concatenate([A.flatten(), b])
            y = int(is_sat)
            
            self.samples.append((
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Create train and eval dataloaders."""
    
    train_dataset = LinearF2Dataset(
        n_samples=config.dataset_size,
        n_vars=config.n_vars,
        n_equations=config.n_equations,
        seed=config.seed
    )
    
    eval_dataset = LinearF2Dataset(
        n_samples=config.dataset_size // 5,
        n_vars=config.n_vars,
        n_equations=config.n_equations,
        seed=config.seed + 1000  # Different seed for eval
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    return train_loader, eval_loader
```

### 7.3 Test the Dataset

```python
# Quick test
if __name__ == "__main__":
    dataset = LinearF2Dataset(n_samples=1000, n_vars=10, n_equations=15)
    
    # Check class balance
    labels = [y.item() for _, y in dataset]
    print(f"Class balance: {sum(labels)/len(labels):.2%} satisfiable")
    # Should be roughly 50%
    
    # Check shape
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")  # Should be (15*10 + 15,) = (165,)
    print(f"Label: {y.item()}")
```

**✓ Checkpoint:** Dataset should have ~50% satisfiable systems and correct shapes.

---

## Step 8: Running Your First Experiment

**Goal:** Train a PVG system and verify it works.

### 8.1 Main Experiment Script

Create `experiments/synthetic/run_experiment.py`:

```python
"""
experiments/synthetic/run_experiment.py

Run a single synthetic experiment.

Note: No sys.path hacks needed! The pvg package is installed via `uv pip install -e .`
"""

import logging
import torch
import json
from pathlib import Path

from pvg.config import SyntheticConfig
from pvg.models import create_models
from pvg.training import StackelbergTrainer
from experiments.synthetic.linear_f2 import create_dataloaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(config: SyntheticConfig):
    """Run a single experiment with given config."""
    
    # Reproducibility
    torch.manual_seed(config.seed)
    
    logger.info(f"Running experiment: {config.experiment_name}")
    logger.info(f"  epsilon = {config.epsilon}")
    logger.info(f"  seed = {config.seed}")
    
    # Create data
    train_loader, eval_loader = create_dataloaders(config)
    logger.info(f"  train samples = {len(train_loader.dataset)}")
    logger.info(f"  eval samples = {len(eval_loader.dataset)}")
    
    # Create models
    prover, verifier = create_models(config)
    logger.info(f"  prover params = {sum(p.numel() for p in prover.parameters()):,}")
    logger.info(f"  verifier params = {sum(p.numel() for p in verifier.parameters()):,}")
    
    # Train
    trainer = StackelbergTrainer(
        prover=prover,
        verifier=verifier,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader
    )
    
    history = trainer.train()
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    logger.info(f"Final results:")
    logger.info(f"  Clean loss: {final_metrics['clean_loss']:.4f}")
    logger.info(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    logger.info(f"  Prover success rate: {final_metrics['prover_success_rate']:.3f}")
    
    # Save results
    output_dir = Path("results/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "config": {
            "epsilon": config.epsilon,
            "seed": config.seed,
            "num_rounds": config.num_rounds,
        },
        "final_metrics": final_metrics,
        "history": history
    }
    
    output_path = output_dir / f"{config.experiment_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_rounds", type=int, default=100)
    args = parser.parse_args()
    
    config = SyntheticConfig(
        experiment_name=f"linear_f2_eps{args.epsilon}_seed{args.seed}",
        epsilon=args.epsilon,
        seed=args.seed,
        num_rounds=args.num_rounds,
    )
    
    run_experiment(config)
```

### 8.2 Run It!

```bash
# Clean labels (baseline)
uv run python -m experiments.synthetic.run_experiment --epsilon 0.0 --seed 42

# With noise
uv run python -m experiments.synthetic.run_experiment --epsilon 0.1 --seed 42
uv run python -m experiments.synthetic.run_experiment --epsilon 0.2 --seed 42
```

### 8.3 What to Expect

**With ε = 0 (clean labels):**
- Accuracy should reach 70-90% (depends on model capacity)
- Prover success rate will be high (prover learns to convince verifier)

**With ε > 0:**
- Accuracy should decrease
- Clean loss should increase
- This is the equilibrium deviation we're measuring!

**✓ Checkpoint:** Training should complete without errors and show improving losses.

---

## Step 9: Adding Evaluation and Metrics

**Goal:** Create utilities for tracking and analyzing experiments.

**File:** `pvg/metrics.py`

```python
"""
pvg/metrics.py

Metrics tracking and analysis utilities.
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.train_history = []
        self.eval_history = []
    
    def log_train(self, round_t: int, metrics: Dict):
        self.train_history.append({"round": round_t, **metrics})
    
    def log_eval(self, round_t: int, metrics: Dict):
        self.eval_history.append({"round": round_t, **metrics})
    
    def get_history(self) -> Dict:
        return {
            "train": self.train_history,
            "eval": self.eval_history
        }
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.get_history(), f, indent=2)


def compute_equilibrium_deviation(
    baseline_loss: float,
    noisy_loss: float
) -> float:
    """
    Compute deviation from clean equilibrium.
    
    This is our primary metric for answering the research question.
    """
    return noisy_loss - baseline_loss


def aggregate_runs(results: List[Dict]) -> Dict:
    """Aggregate results across random seeds."""
    
    metrics = {}
    for key in results[0]["final_metrics"].keys():
        values = [r["final_metrics"][key] for r in results]
        metrics[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values
        }
    
    return metrics
```

---

## Step 10: Running the Noise Sweep

**Goal:** Sweep over noise levels to test the hypothesis.

Create `experiments/sweep_noise.py`:

```python
"""
experiments/sweep_noise.py

Main experiment: sweep noise levels and measure equilibrium deviation.

Note: No sys.path hacks needed! The pvg package is installed via `uv pip install -e .`
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from pvg.config import SyntheticConfig
from experiments.synthetic.run_experiment import run_experiment


def run_noise_sweep(
    noise_levels: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    seeds: List[int] = [42, 123, 456],
    num_rounds: int = 100
):
    """
    Run experiments across noise levels and seeds.
    
    This tests the hypothesis: ||NE(G_ε) - NE(G_0)|| = O(ε)
    """
    results = {}
    
    for eps in noise_levels:
        results[eps] = []
        
        for seed in seeds:
            config = SyntheticConfig(
                experiment_name=f"sweep_eps{eps}_seed{seed}",
                epsilon=eps,
                seed=seed,
                num_rounds=num_rounds,
            )
            
            result = run_experiment(config)
            results[eps].append(result["final_metrics"])
    
    return results


def analyze_results(results: dict) -> dict:
    """Analyze equilibrium deviation vs noise level."""
    
    # Get baseline (clean) loss
    baseline_losses = [r["clean_loss"] for r in results[0.0]]
    baseline = np.mean(baseline_losses)
    
    analysis = {
        "noise_levels": [],
        "deviation_mean": [],
        "deviation_std": [],
        "accuracy_mean": [],
        "accuracy_std": [],
    }
    
    for eps in sorted(results.keys()):
        losses = [r["clean_loss"] for r in results[eps]]
        accuracies = [r["accuracy"] for r in results[eps]]
        
        deviation = np.mean(losses) - baseline
        
        analysis["noise_levels"].append(eps)
        analysis["deviation_mean"].append(deviation)
        analysis["deviation_std"].append(np.std(losses))
        analysis["accuracy_mean"].append(np.mean(accuracies))
        analysis["accuracy_std"].append(np.std(accuracies))
    
    # Fit linear model to test O(ε) hypothesis
    eps_arr = np.array(analysis["noise_levels"])
    dev_arr = np.array(analysis["deviation_mean"])
    
    # Linear fit: deviation = a * epsilon + b
    coeffs = np.polyfit(eps_arr, dev_arr, 1)
    linear_fit = np.poly1d(coeffs)
    
    # R² score
    ss_res = np.sum((dev_arr - linear_fit(eps_arr))**2)
    ss_tot = np.sum((dev_arr - np.mean(dev_arr))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    analysis["linear_fit"] = {
        "slope": coeffs[0],
        "intercept": coeffs[1],
        "r_squared": r2
    }
    
    return analysis


def plot_results(analysis: dict, output_path: str = "results/noise_sweep_plot.png"):
    """Generate publication-quality plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    eps = analysis["noise_levels"]
    
    # Plot 1: Equilibrium deviation
    ax1 = axes[0]
    ax1.errorbar(
        eps,
        analysis["deviation_mean"],
        yerr=analysis["deviation_std"],
        fmt="o-",
        capsize=5,
        label="Empirical",
        color="blue"
    )
    
    # Add linear fit
    eps_fine = np.linspace(0, max(eps), 100)
    linear_pred = analysis["linear_fit"]["slope"] * eps_fine + analysis["linear_fit"]["intercept"]
    ax1.plot(
        eps_fine, linear_pred, "--",
        label=f'Linear fit (R²={analysis["linear_fit"]["r_squared"]:.3f})',
        color="red"
    )
    
    ax1.set_xlabel("Noise level ε", fontsize=12)
    ax1.set_ylabel("Equilibrium deviation", fontsize=12)
    ax1.set_title("Equilibrium Sensitivity to Label Noise", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy degradation
    ax2 = axes[1]
    ax2.errorbar(
        eps,
        analysis["accuracy_mean"],
        yerr=analysis["accuracy_std"],
        fmt="s-",
        capsize=5,
        color="green"
    )
    ax2.axhline(y=0.5, color="red", linestyle="--", label="Random baseline")
    
    ax2.set_xlabel("Noise level ε", fontsize=12)
    ax2.set_ylabel("Verifier accuracy (clean test)", fontsize=12)
    ax2.set_title("Accuracy Degradation Under Noise", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    # Run the sweep
    results = run_noise_sweep(
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4],
        seeds=[42, 123],
        num_rounds=50  # Reduce for quick test
    )
    
    # Analyze
    analysis = analyze_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("NOISE SWEEP RESULTS")
    print("="*60)
    print(f"Linear fit: deviation ≈ {analysis['linear_fit']['slope']:.4f} × ε + {analysis['linear_fit']['intercept']:.4f}")
    print(f"R² = {analysis['linear_fit']['r_squared']:.3f}")
    print("\nIf R² > 0.9, the O(ε) hypothesis is supported.")
    print("="*60)
    
    # Save analysis
    Path("results").mkdir(exist_ok=True)
    with open("results/noise_sweep_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Plot
    plot_results(analysis)
```

### 10.1 Run the Sweep

```bash
uv run python -m experiments.sweep_noise
```

### 10.2 Interpret Results

**If R² > 0.9:** The equilibrium deviation scales linearly with ε (O(ε) hypothesis supported).

**If accuracy drops to ~50% at some ε:** You've found the "breakdown" threshold where noise overwhelms the signal.

**✓ Checkpoint:** You should see a plot showing how equilibrium deviates with noise.

---

## Step 11: Extending to NLP Tasks (Tier 2)

**Goal:** Validate on real NLP data.

### 11.1 Key Differences from Synthetic

| Aspect | Synthetic | NLP |
|--------|-----------|-----|
| Input | Numeric vector | Text (tokens) |
| Models | MLPs | Transformers |
| Message | Continuous vector | Still continuous (encoded representation) |
| Complexity | Controllable | Real-world noise |

### 11.2 Transformer Prover/Verifier

Add to `pvg/models.py`:

```python
class TransformerProver(BaseProver):
    """
    Transformer prover for NLP tasks.
    
    Uses pretrained encoder to map text → dense message vector.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        message_dim: int = 256,
        freeze_encoder: bool = False
    ):
        super().__init__()
        from transformers import AutoModel
        
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, message_dim),
            nn.Tanh()
        )
        self.message_dim = message_dim
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.projection(cls_output)


class TransformerVerifier(BaseVerifier):
    """
    Transformer verifier for NLP tasks.
    
    Takes encoded text and message, produces binary classification.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        message_dim: int = 256
    ):
        super().__init__()
        from transformers import AutoModel
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + message_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_output, message], dim=-1)
        return self.classifier(combined).squeeze(-1)
```

### 11.3 SNLI Dataset Setup

Create `experiments/nlp/snli_binary.py`:

```python
"""
experiments/nlp/snli_binary.py

Binary NLI using SNLI dataset.
"""

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def load_snli_binary(split: str, max_samples: int = None):
    """
    Load SNLI and convert to binary classification.
    
    Entailment (0) → 1 (positive)
    Contradiction/Neutral (1,2) → 0 (negative)
    """
    dataset = load_dataset("stanfordnlp/snli", split=split)
    
    # Filter invalid labels
    dataset = dataset.filter(lambda x: x["label"] != -1)
    
    # Binarize
    def binarize(example):
        example["binary_label"] = 1 if example["label"] == 0 else 0
        return example
    
    dataset = dataset.map(binarize)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset


class SNLICollator:
    """Collate function for SNLI batches."""
    
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        premises = [ex["premise"] for ex in batch]
        hypotheses = [ex["hypothesis"] for ex in batch]
        labels = torch.tensor([ex["binary_label"] for ex in batch])
        
        encodings = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "label": labels
        }
```

### 11.4 Adapting the Trainer

The training loop needs to handle dict batches with `input_ids`, `attention_mask`. Modify `_unpack_batch` in `StackelbergTrainer`:

```python
def _unpack_batch(self, batch):
    """Unpack batch for both synthetic and NLP tasks."""
    if isinstance(batch, dict):
        # NLP batch
        return batch, batch["label"].to(self.device)
    else:
        # Synthetic batch (x, y tuple)
        x, y = batch
        return x.to(self.device), y.to(self.device)
```

---

## Step 12: Scaling to LLMs (Tier 3)

**Goal:** Apply the framework to LLM-scale math reasoning.

### 12.1 Key Differences

| Aspect | Tier 1-2 | Tier 3 |
|--------|----------|--------|
| Prover output | Continuous vector | Generated text |
| Training | Gradient descent | RL (PPO) |
| Reward | Differentiable | Non-differentiable (correctness) |

### 12.2 LLM Prover

```python
class LLMProver(BaseProver):
    """
    LLM prover that generates text proofs.
    """
    
    def __init__(self, model_name: str = "gpt2", max_length: int = 256):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
    
    def forward(self, input_ids, attention_mask):
        """Return hidden states (for value head in PPO)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        return outputs.hidden_states[-1][:, -1, :]
    
    def generate(self, input_ids, attention_mask):
        """Generate text proofs."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id
        )
```

### 12.3 PPO Training

For LLM provers, use the TRL library's PPOTrainer:

```python
from trl import PPOTrainer, PPOConfig

def train_llm_prover(prover, verifier, config, data):
    """
    Train LLM prover with PPO.
    
    Reward = log p_θ(ŷ=1|x,z) from verifier
    """
    ppo_config = PPOConfig(
        model_name=config.prover_model,
        learning_rate=config.prover_lr,
        batch_size=config.batch_size,
        ppo_epochs=4,
    )
    
    ppo_trainer = PPOTrainer(
        model=prover.model,
        config=ppo_config,
        tokenizer=prover.tokenizer
    )
    
    for batch in data:
        # Generate proofs
        proofs = prover.generate(batch["input_ids"], batch["attention_mask"])
        
        # Get rewards from verifier
        with torch.no_grad():
            logits = verifier(batch["problem"], proofs)
            rewards = torch.sigmoid(logits)  # p(ŷ=1)
        
        # PPO update
        ppo_trainer.step(batch["input_ids"], proofs, rewards.tolist())
```

---

## Step 13: Debugging and Validation

### 13.1 Sanity Checks to Run

Before trusting your results, verify:

1. **Gradient flow:**
```python
import torch
import torch.nn.functional as F

# Prover gradients should be negative (increasing logit decreases loss)
logits = torch.zeros(32, requires_grad=True)
loss = F.softplus(-logits).mean()  # prover loss
loss.backward()
assert (logits.grad < 0).all()
```

2. **Noise statistics:**
```python
import torch
from pvg.noise import inject_label_noise

# Flip rate should match epsilon
labels = torch.ones(10000).long()
noisy = inject_label_noise(labels, 0.1)
flip_rate = (noisy != labels).float().mean()
assert abs(flip_rate - 0.1) < 0.02
```

3. **Prover improves:**
```python
# Prover loss should decrease when training against fixed verifier
initial_loss = F.softplus(-verifier(x, prover(x))).mean()
# ... train prover ...
final_loss = F.softplus(-verifier(x, prover(x))).mean()
assert final_loss < initial_loss
```

4. **Bilevel structure:**
```python
# Verify prover is frozen during verifier updates
assert all(not p.requires_grad for p in prover.parameters())
```

### 13.2 Common Bugs and Fixes

| Bug | Symptom | Fix |
|-----|---------|-----|
| Nash instead of Stackelberg | Training unstable | Ensure prover fully optimizes before verifier |
| Noise in eval | Artificially low accuracy | Only inject noise in training |
| Prover collapse | Same message for all inputs | Add entropy regularization |
| Verifier ignores message | Same prediction regardless of z | Check concatenation happens |
| NaN losses | Training crashes | Use numerically stable loss (softplus) |

### 13.3 Validation Checklist

Before publishing results:

- [ ] Clean accuracy at ε=0 is reasonable (>70%)
- [ ] Accuracy degrades with increasing ε
- [ ] Multiple random seeds give consistent results
- [ ] Linear fit R² is reported
- [ ] Comparison to random baseline included

---

## Summary: Implementation Order

1. **Understand the theory** (Stackelberg, not Nash; prover defends 1)
2. **Set up project structure**
3. **Implement noise injection** → test it
4. **Understand the losses** (use inline PyTorch functions)
5. **Build MLP models** → test gradient flow
6. **Implement training loop** → careful about bilevel structure
7. **Create synthetic dataset** → verify class balance
8. **Run single experiment** → verify training works
9. **Run noise sweep** → test hypothesis
10. **Extend to NLP** → validate on real data
11. **Scale to LLMs** → if resources permit

**Total time estimate:** 2-3 days for Tier 1, 1 week for Tier 2, 2 weeks for Tier 3.

---

## Quick Reference

### Loss Functions
```python
# Verifier: cross-entropy with noisy labels
v_loss = F.binary_cross_entropy_with_logits(logits, noisy_labels.float())

# Prover: maximize prob of label 1 (no labels needed!)
p_loss = F.softplus(-logits).mean()  # = -log(sigmoid(logits))
```

### Training Loop
```
for round in rounds:
    freeze(verifier); train prover for K_P steps
    freeze(prover); train verifier for K_V steps
```

### Effective Label
```python
y_eff = (1 - 2*epsilon) * y + epsilon
```

### Primary Metric
```
Equilibrium deviation = L_V^clean(θ*, φ*; ε) - L_V^clean(θ*, φ*; ε=0)
```
