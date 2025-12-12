"""
Noise sweep experiment: measure equilibrium deviation vs. label noise level.

This is the main experiment for testing the hypothesis:
    ||NE(G_ε) - NE(G_0)|| = O(ε)

Usage:
    uv run python -m experiments.sweep_noise
    uv run python -m experiments.sweep_noise --noise_levels 0.0 0.1 0.2 0.3 --seeds 42 123 456
    uv run python -m experiments.sweep_noise --num_rounds 50 --quick  # Quick test
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from pvg.config import SyntheticConfig
from pvg.metrics import analyze_noise_sweep, save_analysis
from experiments.synthetic.run_experiment import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_noise_sweep(
    noise_levels: List[float],
    seeds: List[int],
    num_rounds: int = 100,
    n_vars: int = 12,
    n_equations: int = 10,
    dataset_size: int = 5000,
    **kwargs,
) -> Dict[float, List[Dict]]:
    """
    Run experiments across noise levels and seeds.

    Args:
        noise_levels: List of epsilon values to test
        seeds: List of random seeds for replication
        num_rounds: Training rounds per experiment
        n_vars: Number of variables in linear system
        n_equations: Number of equations in linear system
        dataset_size: Training dataset size
        **kwargs: Additional config overrides

    Returns:
        Dictionary mapping epsilon -> list of result dicts
    """
    total_runs = len(noise_levels) * len(seeds)
    logger.info(f"Starting noise sweep: {len(noise_levels)} noise levels × {len(seeds)} seeds = {total_runs} runs")
    logger.info(f"Noise levels: {noise_levels}")
    logger.info(f"Seeds: {seeds}")

    results: Dict[float, List[Dict]] = {}

    run_count = 0
    for eps in noise_levels:
        results[eps] = []

        for seed in seeds:
            run_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Run {run_count}/{total_runs}: ε={eps}, seed={seed}")
            logger.info(f"{'='*60}")

            config = SyntheticConfig(
                experiment_name=f"sweep_eps{eps}_seed{seed}",
                epsilon=eps,
                seed=seed,
                num_rounds=num_rounds,
                n_vars=n_vars,
                n_equations=n_equations,
                dataset_size=dataset_size,
                **kwargs,
            )

            result = run_experiment(config)
            results[eps].append(result)

    return results


def plot_results(analysis: Dict, output_dir: Path) -> None:
    """Generate plots for the noise sweep results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    eps = analysis["noise_levels"]

    ax1 = axes[0]
    ax1.errorbar(
        eps,
        analysis["deviation_mean"],
        yerr=analysis["deviation_std"],
        fmt="o-",
        capsize=5,
        label="Empirical",
        color="#2563eb",
        linewidth=2,
        markersize=8,
    )

    if "linear_fit" in analysis:
        eps_fine = np.linspace(0, max(eps), 100)
        linear_pred = analysis["linear_fit"]["slope"] * eps_fine + analysis["linear_fit"]["intercept"]
        r2 = analysis["linear_fit"]["r_squared"]
        ax1.plot(
            eps_fine,
            linear_pred,
            "--",
            label=f"Linear fit (R²={r2:.3f})",
            color="#dc2626",
            linewidth=2,
        )

    ax1.set_xlabel("Noise level ε", fontsize=12)
    ax1.set_ylabel("Equilibrium deviation (Δ clean loss)", fontsize=12)
    ax1.set_title("Equilibrium Sensitivity to Label Noise", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, max(eps) + 0.02)

    ax2 = axes[1]
    ax2.errorbar(
        eps,
        analysis["accuracy_mean"],
        yerr=analysis["accuracy_std"],
        fmt="s-",
        capsize=5,
        color="#16a34a",
        linewidth=2,
        markersize=8,
        label="Verifier accuracy",
    )
    ax2.axhline(y=0.5, color="#dc2626", linestyle="--", linewidth=1.5, label="Random baseline")

    ax2.set_xlabel("Noise level ε", fontsize=12)
    ax2.set_ylabel("Accuracy (on clean test set)", fontsize=12)
    ax2.set_title("Accuracy Degradation Under Noise", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.02, max(eps) + 0.02)
    ax2.set_ylim(0.4, 1.0)

    plt.tight_layout()

    plot_path = output_dir / "noise_sweep_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")

    pdf_path = output_dir / "noise_sweep_plot.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    logger.info(f"PDF saved to {pdf_path}")

    plt.close()


def print_summary(analysis: Dict) -> None:
    """Print a summary of the noise sweep results."""
    print("\n" + "=" * 70)
    print("NOISE SWEEP RESULTS SUMMARY")
    print("=" * 70)

    print("\n📊 Results by noise level:")
    print("-" * 50)
    print(f"{'ε':>6} | {'Deviation':>12} | {'Accuracy':>12}")
    print("-" * 50)
    for i, eps in enumerate(analysis["noise_levels"]):
        dev = analysis["deviation_mean"][i]
        dev_std = analysis["deviation_std"][i]
        acc = analysis["accuracy_mean"][i]
        acc_std = analysis["accuracy_std"][i]
        print(f"{eps:>6.2f} | {dev:>6.4f}±{dev_std:.4f} | {acc:>6.3f}±{acc_std:.3f}")
    print("-" * 50)

    if "linear_fit" in analysis:
        fit = analysis["linear_fit"]
        print(f"\n📈 Linear fit: deviation ≈ {fit['slope']:.4f} × ε + {fit['intercept']:.4f}")
        print(f"   R² = {fit['r_squared']:.4f}")
        
        if fit["r_squared"] > 0.9:
            print("\n✅ Strong linear relationship (R² > 0.9)")
            print("   → O(ε) hypothesis is SUPPORTED")
        elif fit["r_squared"] > 0.7:
            print("\n⚠️  Moderate linear relationship (0.7 < R² < 0.9)")
            print("   → O(ε) hypothesis has partial support")
        else:
            print("\n❌ Weak linear relationship (R² < 0.7)")
            print("   → O(ε) hypothesis NOT supported; consider O(√ε) or O(ε²)")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run noise sweep experiment for PVG equilibrium analysis"
    )

    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        help="Noise levels (epsilon) to test",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for replication",
    )
    parser.add_argument("--num_rounds", type=int, default=100, help="Training rounds per run")
    parser.add_argument("--n_vars", type=int, default=12, help="Variables in linear system")
    parser.add_argument("--n_equations", type=int, default=10, help="Equations in linear system")
    parser.add_argument("--dataset_size", type=int, default=5000, help="Training dataset size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced settings")
    parser.add_argument("--output_dir", type=str, default="results/sweep", help="Output directory")

    args = parser.parse_args()

    if args.quick:
        args.noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
        args.seeds = [42, 123]
        args.num_rounds = 50
        args.dataset_size = 2000
        logger.info("Quick mode enabled")

    for eps in args.noise_levels:
        if not 0 <= eps < 0.5:
            parser.error(f"All noise levels must be in [0, 0.5), got {eps}")

    if 0.0 not in args.noise_levels:
        logger.warning("Adding ε=0.0 to noise_levels (required for baseline)")
        args.noise_levels = [0.0] + list(args.noise_levels)

    args.noise_levels = sorted(set(args.noise_levels))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_noise_sweep(
        noise_levels=args.noise_levels,
        seeds=args.seeds,
        num_rounds=args.num_rounds,
        n_vars=args.n_vars,
        n_equations=args.n_equations,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
    )

    analysis = analyze_noise_sweep(results)
    analysis["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "num_seeds": len(args.seeds),
        "num_noise_levels": len(args.noise_levels),
        "num_rounds": args.num_rounds,
        "n_vars": args.n_vars,
        "n_equations": args.n_equations,
        "dataset_size": args.dataset_size,
    }

    analysis_path = output_dir / "noise_sweep_analysis.json"
    save_analysis(analysis, str(analysis_path))
    logger.info(f"Analysis saved to {analysis_path}")

    results_path = output_dir / "noise_sweep_raw_results.json"
    results_json = {str(k): v for k, v in results.items()}
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Raw results saved to {results_path}")

    plot_results(analysis, output_dir)
    print_summary(analysis)


if __name__ == "__main__":
    main()
