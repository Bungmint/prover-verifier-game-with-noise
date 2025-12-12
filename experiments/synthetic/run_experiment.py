"""
Run a single synthetic PVG experiment.

Usage:
    uv run python -m experiments.synthetic.run_experiment --epsilon 0.0 --seed 42
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from pvg.config import SyntheticConfig
from pvg.models import create_mlp_models
from pvg.training import StackelbergTrainer
from experiments.synthetic.linear_f2 import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(config: SyntheticConfig) -> dict:
    """Run a single PVG experiment."""
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info(f"Running experiment: {config.experiment_name}")
    logger.info(f"  epsilon={config.epsilon}, seed={config.seed}, device={config.device}")

    train_loader, eval_loader = create_dataloaders(
        n_vars=config.n_vars,
        n_equations=config.n_equations,
        train_size=config.dataset_size,
        eval_size=config.eval_size,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    logger.info(f"  train={len(train_loader.dataset)}, eval={len(eval_loader.dataset)}")

    prover, verifier = create_mlp_models(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        message_dim=config.message_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    trainer = StackelbergTrainer(
        prover=prover,
        verifier=verifier,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    history = trainer.train()
    final_metrics = trainer.evaluate()

    logger.info(f"Final: loss={final_metrics['clean_loss']:.4f}, acc={final_metrics['accuracy']:.3f}")

    results = {
        "config": {
            "experiment_name": config.experiment_name,
            "epsilon": config.epsilon,
            "seed": config.seed,
            "num_rounds": config.num_rounds,
            "n_vars": config.n_vars,
            "n_equations": config.n_equations,
            "hidden_dim": config.hidden_dim,
            "message_dim": config.message_dim,
        },
        "final_metrics": final_metrics,
        "history": history,
    }

    output_dir = Path("results/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.experiment_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic PVG experiment")

    parser.add_argument("--epsilon", type=float, default=0.0, help="Noise level [0, 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--n_vars", type=int, default=10)
    parser.add_argument("--n_equations", type=int, default=15)
    parser.add_argument("--dataset_size", type=int, default=5000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--message_dim", type=int, default=64)
    parser.add_argument("--prover_lr", type=float, default=1e-4)
    parser.add_argument("--verifier_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=20)

    args = parser.parse_args()

    if not 0 <= args.epsilon < 0.5:
        parser.error(f"epsilon must be in [0, 0.5), got {args.epsilon}")

    config = SyntheticConfig(
        experiment_name=f"linear_f2_eps{args.epsilon}_seed{args.seed}",
        epsilon=args.epsilon,
        seed=args.seed,
        num_rounds=args.num_rounds,
        n_vars=args.n_vars,
        n_equations=args.n_equations,
        dataset_size=args.dataset_size,
        hidden_dim=args.hidden_dim,
        message_dim=args.message_dim,
        prover_lr=args.prover_lr,
        verifier_lr=args.verifier_lr,
        batch_size=args.batch_size,
        log_every=args.log_every,
        eval_every=args.eval_every,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
