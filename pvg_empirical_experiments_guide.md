# Empirical Experiments for Noisy Prover-Verifier Games

## Overview

This document outlines dataset recommendations and training procedures for empirically investigating the Stackelberg equilibrium of Prover-Verifier Games under label noise, as described in your CS 272 project proposal.

---

## Part 1: Recommended Datasets

Based on your theoretical framework (binary decision problems with noisy labels), I recommend a **tiered approach** from simpler to more complex settings:

### Tier 1: Toy/Synthetic Tasks (For Theoretical Validation)

These align closely with Anil et al.'s contrived settings and are ideal for validating your primary hypothesis before scaling up.

#### 1.1 Synthetic Linear Equations over F₂
- **Description**: Generate random systems of linear equations and labels indicating satisfiability
- **Why it fits**: Directly matches Example 2.1 in your proposal; ground truth is computable
- **Implementation**: Generate using NumPy/SciPy; controllable noise injection
- **Prover/Verifier**: Simple MLPs or linear models

```python
# Example generation
import numpy as np

def generate_linear_system(n_vars=10, n_equations=15):
    A = np.random.randint(0, 2, (n_equations, n_vars))
    b = np.random.randint(0, 2, n_equations)
    # Check satisfiability via Gaussian elimination in F2
    satisfiable = check_satisfiability_f2(A, b)
    return A, b, satisfiable
```

#### 1.2 Binary Erasure Channel (from Anil et al.)
- **Description**: Prover sends bits through a noisy channel; verifier decides correctness
- **Why it fits**: Amenable to closed-form analysis; directly referenced in your backup hypothesis
- **Implementation**: Custom synthetic data generator

### Tier 2: Binary Classification NLP Tasks (Moderate Scale)

#### 2.1 SNLI/MNLI (Natural Language Inference) - Binary Version
- **HuggingFace**: `stanfordnlp/snli` (570k pairs), `nyu-mll/multi_nli` (433k pairs)
- **Adaptation**: Merge to binary (entailment vs. {neutral, contradiction})
- **Why it fits**: 
  - Premise → Hypothesis structure maps to Prover → Verifier communication
  - Large scale allows meaningful statistical analysis
  - Well-understood baselines exist

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/snli")
# Convert to binary: entailment (1) vs not-entailment (0)
def binarize(example):
    example['binary_label'] = 1 if example['label'] == 0 else 0
    return example
dataset = dataset.map(binarize)
```

#### 2.2 FEVER (Fact Verification)
- **HuggingFace**: `fever/fever` (185k claims)
- **Labels**: SUPPORTED, REFUTED, NOT ENOUGH INFO
- **Adaptation**: Binary (SUPPORTED vs REFUTED), filter out NEI
- **Why it fits**:
  - Natural prover-verifier interpretation: evidence (proof) → claim (decision)
  - Realistic noisy label scenario (human annotators disagree)

### Tier 3: Mathematical Reasoning (LLM-Scale)

#### 3.1 GSM8K (Grade School Math)
- **HuggingFace**: `openai/gsm8k` (8.5k problems)
- **Why it fits**: Used in Kirchner et al. (2024) - the OpenAI paper you referenced
- **Adaptation for your setup**:
  - **Label**: Binary (correct/incorrect final answer)
  - **Prover message**: Chain-of-thought solution
  - **Noisy labels**: Flip correctness labels with probability ε

```python
from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main")
# Extract numerical answer and create binary correctness label
def extract_answer(example):
    answer = example['answer'].split('####')[-1].strip()
    example['numerical_answer'] = float(answer)
    return example
```

#### 3.2 GSM8K-Platinum (Cleaner Version)
- **HuggingFace**: `madrylab/gsm8k-platinum`
- **Advantage**: Reduced annotation errors in original GSM8K
- **Use case**: Clean baseline for comparison with noisy settings

---

## Part 2: Model Architectures by Dataset

### For Toy Tasks (Tier 1)

| Component | Architecture | Parameters |
|-----------|--------------|------------|
| Prover P | 2-layer MLP | ~10K params |
| Verifier V | 2-layer MLP | ~10K params |
| Message space M | Binary vector | dim = 32-128 |

### For NLP Tasks (Tier 2)

| Component | Architecture | Parameters |
|-----------|--------------|------------|
| Prover P | BERT-base / DistilBERT | 110M / 66M |
| Verifier V | Smaller transformer (DistilBERT / TinyBERT) | 66M / 14.5M |
| Message space M | Text (natural language) | - |

### For Math Reasoning (Tier 3)

Following Kirchner et al.'s setup:

| Component | Architecture | Parameters |
|-----------|--------------|------------|
| Prover P | GPT-2 / Llama-3.2-1B | 117M-1B |
| Verifier V | Smaller LM (GPT-2-small) | 117M |
| Message space M | Chain-of-thought text | - |

---

## Part 3: Training Procedure

### Loss Functions (From Your Proposal)

Recall from Definition 5.2:

**Noisy Verifier Loss:**
$$L_V^\epsilon(\theta, \phi) = \mathbb{E}_{X \sim P_X, Y=f(X), \tilde{Y}|Y, Z \sim p_\phi(\cdot|X)} \left[ \ell(p_\theta(X, Z), \tilde{Y}) \right]$$

**Noisy Prover Loss (Anil setup - always defends label 1):**
$$L_P^\epsilon(\theta, \phi) = \mathbb{E}_{X \sim P_X, Z \sim p_\phi(\cdot|X)} \left[ \ell(p_\theta(X, Z), 1) \right]$$

where $\ell(p, y) = -[y \log p + (1-y) \log(1-p)]$ is cross-entropy.

### Training Algorithm: Stackelberg Learning

Your proposal uses **Verifier-leading Stackelberg equilibrium**. The training procedure alternates:

```
Algorithm: Stackelberg Prover-Verifier Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Dataset D, noise level ε, learning rates η_V, η_P
Initialize: θ (verifier), φ (prover)

for round t = 1, ..., T:
    # Step 1: Prover best-responds to current verifier
    for step k = 1, ..., K_P:
        Sample batch {(x_i, y_i)}
        Inject noise: ỹ_i = flip(y_i) with prob ε
        Generate messages: z_i ~ p_φ(·|x_i)
        Compute L_P^ε(θ, φ)  # Prover wants V to output 1
        φ ← φ - η_P · ∇_φ L_P^ε(θ, φ)
    
    # Step 2: Verifier optimizes against prover's best response
    for step k = 1, ..., K_V:
        Sample batch {(x_i, y_i)}
        Inject noise: ỹ_i = flip(y_i) with prob ε
        Generate messages: z_i ~ p_φ(·|x_i)
        Compute L_V^ε(θ, φ)
        θ ← θ - η_V · ∇_θ L_V^ε(θ, φ)

Output: (θ*, φ*)
```

### Implementation with HuggingFace TRL

For LLM-scale experiments, use the **TRL library** (Transformer Reinforcement Learning):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch

# ═══════════════════════════════════════════════════════════
# Setup: Prover and Verifier Models
# ═══════════════════════════════════════════════════════════

# Prover: Generates "proof" messages
prover_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
prover_tokenizer = AutoTokenizer.from_pretrained("gpt2")
prover_tokenizer.pad_token = prover_tokenizer.eos_token

# Verifier: Binary classifier on (x, proof) pairs
from transformers import AutoModelForSequenceClassification
verifier_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
verifier_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ═══════════════════════════════════════════════════════════
# Noise Injection
# ═══════════════════════════════════════════════════════════

def inject_label_noise(labels, epsilon):
    """Flip labels with probability epsilon"""
    noise_mask = torch.rand_like(labels.float()) < epsilon
    noisy_labels = torch.where(noise_mask, 1 - labels, labels)
    return noisy_labels

# ═══════════════════════════════════════════════════════════
# Prover Loss: Convince verifier to output 1
# ═══════════════════════════════════════════════════════════

def compute_prover_reward(verifier, x_batch, proof_batch):
    """
    Prover reward = -L_P = log p_θ(ŷ=1 | x, z)
    In Anil's setup, prover always defends label 1
    """
    inputs = verifier_tokenizer(
        [f"{x} [SEP] {proof}" for x, proof in zip(x_batch, proof_batch)],
        padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = verifier(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        reward = probs[:, 1]  # Probability of label 1
    return reward

# ═══════════════════════════════════════════════════════════
# Verifier Loss: Standard cross-entropy with noisy labels
# ═══════════════════════════════════════════════════════════

def compute_verifier_loss(verifier, x_batch, proof_batch, noisy_labels):
    """
    Verifier loss = L_V^ε = CE(p_θ(x, z), ỹ)
    """
    inputs = verifier_tokenizer(
        [f"{x} [SEP] {proof}" for x, proof in zip(x_batch, proof_batch)],
        padding=True, truncation=True, return_tensors="pt"
    )
    outputs = verifier(**inputs, labels=noisy_labels)
    return outputs.loss

# ═══════════════════════════════════════════════════════════
# Full Training Loop (Stackelberg Style)
# ═══════════════════════════════════════════════════════════

ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
)

ppo_trainer = PPOTrainer(
    model=prover_model,
    config=ppo_config,
    tokenizer=prover_tokenizer,
)

def train_stackelberg(dataset, epsilon, num_rounds=100):
    verifier_optimizer = torch.optim.AdamW(verifier_model.parameters(), lr=2e-5)
    
    for round_t in range(num_rounds):
        for batch in dataset:
            x_batch = batch['input']
            true_labels = batch['label']
            noisy_labels = inject_label_noise(true_labels, epsilon)
            
            # Step 1: Prover generates proofs
            query_tensors = prover_tokenizer(x_batch, return_tensors="pt", padding=True)
            response_tensors = ppo_trainer.generate(query_tensors['input_ids'])
            proofs = prover_tokenizer.batch_decode(response_tensors)
            
            # Step 2: Compute prover reward (from frozen verifier)
            rewards = compute_prover_reward(verifier_model, x_batch, proofs)
            
            # Step 3: PPO update for prover
            ppo_trainer.step(query_tensors['input_ids'], response_tensors, rewards)
            
            # Step 4: Update verifier
            verifier_loss = compute_verifier_loss(
                verifier_model, x_batch, proofs, noisy_labels
            )
            verifier_optimizer.zero_grad()
            verifier_loss.backward()
            verifier_optimizer.step()
        
        # Log metrics for analysis
        print(f"Round {round_t}: V_loss={verifier_loss:.4f}, P_reward={rewards.mean():.4f}")
```

### Alternative: Using DPO/GRPO (More Stable)

For better training stability, consider **Direct Preference Optimization** or **GRPO**:

```python
from trl import DPOTrainer, DPOConfig

# Create preference pairs: (x, winning_proof, losing_proof)
# winning_proof = proof that led to correct classification
# losing_proof = proof that led to incorrect classification

dpo_config = DPOConfig(
    model_name_or_path="gpt2",
    learning_rate=1e-5,
    beta=0.1,  # KL penalty
)

dpo_trainer = DPOTrainer(
    model=prover_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=prover_tokenizer,
)
```

---

## Part 4: Experimental Design

### Primary Experiment: Equilibrium Sensitivity to Noise

**Goal**: Verify Hypothesis 3.1 - that ||NE(G_ε) - NE(G_0)||₂ = O(ε)

```python
# Experimental sweep
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

results = {}
for epsilon in noise_levels:
    # Train to convergence
    theta_star, phi_star = train_stackelberg(dataset, epsilon, num_rounds=100)
    
    # Evaluate on CLEAN test set (ground truth labels)
    clean_verifier_loss = evaluate_verifier(theta_star, phi_star, test_set, epsilon=0)
    clean_accuracy = evaluate_accuracy(theta_star, phi_star, test_set)
    
    results[epsilon] = {
        'verifier_loss': clean_verifier_loss,
        'accuracy': clean_accuracy,
        'theta_norm': torch.norm(theta_star).item(),
        'phi_norm': torch.norm(phi_star).item(),
    }

# Plot: Equilibrium deviation vs noise level
import matplotlib.pyplot as plt

epsilons = list(results.keys())
deviations = [results[e]['verifier_loss'] - results[0]['verifier_loss'] for e in epsilons]

plt.plot(epsilons, deviations, 'o-')
plt.xlabel('Noise level ε')
plt.ylabel('||NE(G_ε) - NE(G_0)||')
plt.title('Equilibrium Sensitivity to Label Noise')
plt.savefig('equilibrium_sensitivity.png')
```

### Secondary Experiment: Breakdown Threshold

**Goal**: Find the critical ε* where training dynamics "break down"

Indicators of breakdown:
1. Verifier accuracy drops below random (50%)
2. Prover success rate saturates
3. Training loss becomes non-convergent

```python
def detect_breakdown(training_history, window=10):
    """Detect if training has broken down"""
    recent_losses = training_history[-window:]
    
    # Check for non-convergence (high variance)
    if np.std(recent_losses) > 0.5 * np.mean(recent_losses):
        return True
    
    # Check for degenerate solution
    if recent_losses[-1] > 0.9 * np.log(2):  # Close to random
        return True
    
    return False
```

---

## Part 5: Metrics to Track

| Metric | Description | Purpose |
|--------|-------------|---------|
| `L_V^ε(θ*, φ*)` | Noisy verifier loss at equilibrium | Training objective |
| `L_V^0(θ*, φ*)` | Clean verifier loss at equilibrium | **Primary metric**: measures true performance |
| `Acc_V` | Verifier accuracy on clean test set | Interpretable performance |
| `||θ* - θ*_0||` | Parameter deviation from clean equilibrium | Tests Hypothesis 3.1 |
| `Prover success rate` | How often verifier outputs 1 | Monitors prover's influence |
| `KL(π_φ || π_φ0)` | KL divergence from clean prover | Alternative deviation metric |

---

## Part 6: Recommended Experimental Progression

1. Implement and validate on synthetic linear equations (Tier 1)
   - Verify training converges for ε=0
   - Sweep over ε ∈ [0, 0.5]
   - Plot equilibrium deviation curve

2. Scale to SNLI/FEVER binary classification (Tier 2)
   - Use DistilBERT for both prover/verifier
   - Compare with supervised baseline
   - Identify breakdown threshold

3. GSM8K experiments (Tier 3)
   - Follow Kirchner et al. setup
   - Add noise injection layer
   - Compare helpful vs sneaky provers under noise

---

## References

- Anil, C., Zhang, G., Wu, Y., & Grosse, R. (2021). Learning to give checkable answers with prover-verifier games. arXiv:2108.12099
- Kirchner, J. H., et al. (2024). Prover-verifier games improve legibility of LLM outputs. arXiv:2407.13692
- HuggingFace TRL: https://huggingface.co/docs/trl
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- SNLI: https://huggingface.co/datasets/stanfordnlp/snli
- FEVER: https://huggingface.co/datasets/fever/fever
