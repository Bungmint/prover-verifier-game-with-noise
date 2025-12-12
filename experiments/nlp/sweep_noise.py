"""Noise sweep experiment for NLP (SNLI) task."""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from pvg.config import NLPConfig
from experiments.nlp.run_experiment import run_experiment


def run_noise_sweep(
    noise_levels: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    seeds: List[int] = [42, 123, 456],
    num_rounds: int = 30,
    train_samples: int = 20000,
    eval_samples: int = 2000,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 16,
) -> dict:
    """Run NLP experiments across noise levels and seeds."""
    results = {}
    total_experiments = len(noise_levels) * len(seeds)
    completed = 0

    for eps in noise_levels:
        results[eps] = []

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Running: ε={eps}, seed={seed} ({completed+1}/{total_experiments})")
            print(f"{'='*60}\n")

            config = NLPConfig(
                experiment_name=f"sweep_snli_eps{eps}_seed{seed}",
                epsilon=eps,
                seed=seed,
                model_name=model_name,
                train_samples=train_samples,
                eval_samples=eval_samples,
                num_rounds=num_rounds,
                batch_size=batch_size,
                log_every=10,
                eval_every=10,
            )

            result = run_experiment(config)
            results[eps].append(result["final_metrics"])
            completed += 1

    return results


def analyze_results(results: dict) -> dict:
    """Analyze equilibrium deviation vs noise level."""
    baseline_losses = [r["clean_loss"] for r in results[0.0]]
    baseline = np.mean(baseline_losses)

    analysis = {
        "noise_levels": [],
        "deviation_mean": [],
        "deviation_std": [],
        "accuracy_mean": [],
        "accuracy_std": [],
        "prover_success_mean": [],
        "prover_success_std": [],
    }

    for eps in sorted(results.keys()):
        losses = [r["clean_loss"] for r in results[eps]]
        accuracies = [r["accuracy"] for r in results[eps]]
        prover_success = [r["prover_success_rate"] for r in results[eps]]

        deviation = np.mean(losses) - baseline

        analysis["noise_levels"].append(eps)
        analysis["deviation_mean"].append(deviation)
        analysis["deviation_std"].append(np.std(losses))
        analysis["accuracy_mean"].append(np.mean(accuracies))
        analysis["accuracy_std"].append(np.std(accuracies))
        analysis["prover_success_mean"].append(np.mean(prover_success))
        analysis["prover_success_std"].append(np.std(prover_success))

    eps_arr = np.array(analysis["noise_levels"])
    dev_arr = np.array(analysis["deviation_mean"])

    if len(eps_arr) > 1:
        coeffs = np.polyfit(eps_arr, dev_arr, 1)
        linear_fit = np.poly1d(coeffs)

        ss_res = np.sum((dev_arr - linear_fit(eps_arr)) ** 2)
        ss_tot = np.sum((dev_arr - np.mean(dev_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        analysis["linear_fit"] = {
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "r_squared": float(r2),
        }
    else:
        analysis["linear_fit"] = {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

    analysis["baseline_loss"] = float(baseline)

    return analysis


def plot_results(analysis: dict, output_dir: str = "results/nlp_sweep") -> None:
    """Generate plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    eps = analysis["noise_levels"]

    ax1 = axes[0]
    ax1.errorbar(
        eps, analysis["deviation_mean"], yerr=analysis["deviation_std"],
        fmt="o-", capsize=5, label="Empirical", color="blue", markersize=8,
    )

    if "linear_fit" in analysis and analysis["linear_fit"]["r_squared"] > 0:
        eps_fine = np.linspace(0, max(eps), 100)
        linear_pred = analysis["linear_fit"]["slope"] * eps_fine + analysis["linear_fit"]["intercept"]
        ax1.plot(
            eps_fine, linear_pred, "--",
            label=f'Linear fit (R²={analysis["linear_fit"]["r_squared"]:.3f})',
            color="red", linewidth=2,
        )

    ax1.set_xlabel("Noise level ε", fontsize=12)
    ax1.set_ylabel("Equilibrium deviation (loss increase)", fontsize=12)
    ax1.set_title("NLP: Equilibrium Sensitivity to Label Noise", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.errorbar(
        eps, analysis["accuracy_mean"], yerr=analysis["accuracy_std"],
        fmt="s-", capsize=5, color="green", markersize=8,
    )
    ax2.axhline(y=0.5, color="red", linestyle="--", label="Random baseline", linewidth=2)
    ax2.set_xlabel("Noise level ε", fontsize=12)
    ax2.set_ylabel("Verifier accuracy (clean test)", fontsize=12)
    ax2.set_title("NLP: Accuracy Degradation Under Noise", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.errorbar(
        eps, analysis["prover_success_mean"], yerr=analysis["prover_success_std"],
        fmt="^-", capsize=5, color="purple", markersize=8,
    )
    ax3.axhline(y=0.5, color="gray", linestyle=":", label="Balanced", linewidth=2)
    ax3.set_xlabel("Noise level ε", fontsize=12)
    ax3.set_ylabel("Prover success rate", fontsize=12)
    ax3.set_title("NLP: Prover Persuasion Rate", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "nlp_noise_sweep_plot.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "nlp_noise_sweep_plot.pdf", bbox_inches="tight")
    print(f"Plots saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="NLP noise sweep experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path("results/nlp_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        print("Running in QUICK TEST mode (reduced samples/rounds)")
        results = run_noise_sweep(
            noise_levels=[0.0, 0.1, 0.2],
            seeds=[42],
            num_rounds=10,
            train_samples=2000,
            eval_samples=500,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    else:
        results = run_noise_sweep(
            noise_levels=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            seeds=[42, 123, 456],
            num_rounds=30,
            train_samples=20000,
            eval_samples=2000,
            model_name=args.model,
            batch_size=args.batch_size,
        )

    raw_results_path = output_dir / "nlp_noise_sweep_raw_results.json"
    serializable_results = {str(k): v for k, v in results.items()}
    with open(raw_results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Raw results saved to {raw_results_path}")

    analysis = analyze_results(results)

    print("\n" + "=" * 60)
    print("NLP NOISE SWEEP RESULTS")
    print("=" * 60)
    print(f"Baseline (ε=0) loss: {analysis['baseline_loss']:.4f}")
    print(
        f"Linear fit: deviation ≈ {analysis['linear_fit']['slope']:.4f} × ε "
        f"+ {analysis['linear_fit']['intercept']:.4f}"
    )
    print(f"R² = {analysis['linear_fit']['r_squared']:.3f}")
    print("\nNoise level | Deviation | Accuracy")
    print("-" * 40)
    for i, eps in enumerate(analysis["noise_levels"]):
        dev = analysis["deviation_mean"][i]
        acc = analysis["accuracy_mean"][i]
        print(f"    {eps:.2f}    |   {dev:+.4f}  |  {acc:.3f}")
    print("=" * 60)
    print("\nIf R² > 0.9, the O(ε) hypothesis is supported.")

    analysis_path = output_dir / "nlp_noise_sweep_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to {analysis_path}")

    plot_results(analysis, str(output_dir))


if __name__ == "__main__":
    main()
