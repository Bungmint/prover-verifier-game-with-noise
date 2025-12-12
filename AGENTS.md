# AGENTS.md — Noisy Prover-Verifier Games

## Project Overview

This is a CS 272 (Game Theory for ML/Alignment) research project investigating how **noisy labels** affect the equilibrium of **Prover-Verifier Games**. The core question: when training data has label noise (with flip probability ε), how does the Stackelberg equilibrium deviate from the clean-label equilibrium?

**Collaborators:** Youngmin Park, Devon Ding  
**Course:** CS 272, Fall 2025 — [Course Page](https://people.eecs.berkeley.edu/~nika/courses/cs272/f25/)

---

## Mathematical Setup

### Decision Problem
- Input space `X`, language/subset `L ⊂ X`
- Ground truth label: `y = 1(x ∈ L) ∈ {0, 1}`
- Dataset: `D = {(x_i, y_i)}` where labels may be noisy

### Noisy Labels
With noise level `ε ∈ [0, 0.5)`:
```
ỹ = y      with prob 1 - ε
ỹ = 1 - y  with prob ε
```

### Players
- **Prover** `P_φ`: maps `x → z` (message/proof), parametrized by `φ`
- **Verifier** `V_θ`: maps `(x, z) → ŷ ∈ {0,1}`, parametrized by `θ`

In Anil et al. setup, prover always defends label 1.

### Loss Functions (Cross-Entropy)
```python
# Verifier loss (under noisy labels)
L_V(θ, φ) = E_{x,ỹ,z} [ -ỹ log p_θ(x,z) - (1-ỹ) log(1 - p_θ(x,z)) ]

# Prover loss (always defends label 1)
L_P(θ, φ) = E_{x,z} [ -log p_θ(x,z) ]
```

Where `p_θ(x,z) = σ(s_θ(x,z))` is verifier's probability of outputting label 1.

### Equilibrium Concept
**Stackelberg equilibrium** (verifier-leading), NOT Nash:
1. Prover best-responds: `φ*(θ) = argmin_φ L_P(θ, φ)`
2. Verifier optimizes reduced objective: `θ* = argmin_θ L_V(θ, φ*(θ))`

---

## Research Questions

1. **Primary Hypothesis:** For small ε, does `||NE(G_ε) - NE(G_0)|| = O(ε)`? (or O(ε²), O(√ε)?)
2. At what noise level does the system "break down"?
3. Can we robustify training against label noise?

---

## Current Goals

### Empirical
- [ ] Design experiments on a reasonable dataset (NOT the contrived Anil setup)
- [ ] Plot verifier loss (on ground-truth labels) vs. noise level ε
- [ ] Identify empirical "breakdown" threshold
- [ ] Consider using RL training approach from OpenAI paper for LLM experiments


## Key References

| Paper | Use For | Link |
|-------|---------|------|
| Anil et al. 2021 | Theoretical framework, BEC instance | https://arxiv.org/abs/2108.12099 |
| Kirchner et al. 2024 | Helpful/Sneaky prover setup, LLM context | https://arxiv.org/abs/2407.13692 |
**Important:** Anil and Kirchner have slightly different reward/loss setups. For THIS project, use the loss functions defined above.

---

## Code Conventions

### Structure
```
├── AGENTS.md
├── experiments/
│   ├── synthetic/      # Toy problems (BEC, linear)
│   └── language/       # LLM experiments if we get there
├── src/
│   ├── models.py       # Prover/Verifier model classes
│   ├── losses.py       # Loss function implementations
│   ├── training.py     # Stackelberg training loop
│   └── utils.py
└── notebooks/          # Exploration and plotting
```

### Preferences
- Python 3.10+, PyTorch preferred
- Use `@dataclass` for configs
- Type hints encouraged
- Experiments should be reproducible (seed everything, log hyperparams)

### Naming
- `eps` or `epsilon` for noise level (not `noise` or `p`)
- `theta` for verifier params, `phi` for prover params
- `L_V`, `L_P` for losses (or `verifier_loss`, `prover_loss`)

---

## Notes for AI Agents

1. **Equilibrium type matters:** We use Stackelberg (bilevel optimization), not simultaneous Nash. The verifier leads.

2. **Don't conflate setups:** Anil uses a single prover defending label 1. Kirchner uses helpful/sneaky provers. We follow Anil's setup with added noise.

3. **Anil's experiments are contrived:** Good for theory, not for realistic empirical work. Design new experiments for empirical validation.

4. **The effective soft label** under symmetric noise is:
   ```
   y_eff(ε) = (1 - 2ε)y + ε
   ```
   This is useful for gradient computations.

5. When implementing training, remember the **bilevel structure**:
   - Inner loop: prover best-responds to current verifier
   - Outer loop: verifier updates accounting for prover's response
