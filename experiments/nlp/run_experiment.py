"""Run a single NLP experiment on binary SNLI."""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from pvg.config import NLPConfig
from pvg.models import create_transformer_models
from experiments.nlp.snli_binary import create_snli_dataloaders
from experiments.nlp.training import NLPStackelbergTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment(config: NLPConfig) -> dict:
    """Run a single NLP experiment with given config."""
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info(f"Running NLP experiment: {config.experiment_name}")
    logger.info(f"  epsilon = {config.epsilon}")
    logger.info(f"  seed = {config.seed}")
    logger.info(f"  model = {config.model_name}")
    logger.info(f"  device = {config.device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_loader, eval_loader = create_snli_dataloaders(
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        train_samples=config.train_samples,
        eval_samples=config.eval_samples,
        seed=config.seed,
    )
    logger.info(f"  train batches = {len(train_loader)}")
    logger.info(f"  eval batches = {len(eval_loader)}")

    prover, verifier = create_transformer_models(
        model_name=config.model_name,
        message_dim=config.message_dim,
        freeze_encoder=config.freeze_encoder,
    )

    n_prover_params = sum(p.numel() for p in prover.parameters())
    n_verifier_params = sum(p.numel() for p in verifier.parameters())
    n_trainable_prover = sum(p.numel() for p in prover.parameters() if p.requires_grad)
    n_trainable_verifier = sum(p.numel() for p in verifier.parameters() if p.requires_grad)

    logger.info(f"  prover params = {n_prover_params:,} ({n_trainable_prover:,} trainable)")
    logger.info(f"  verifier params = {n_verifier_params:,} ({n_trainable_verifier:,} trainable)")

    trainer = NLPStackelbergTrainer(
        prover=prover,
        verifier=verifier,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    history = trainer.train()

    final_metrics = trainer.evaluate()
    logger.info("Final results:")
    logger.info(f"  Clean loss: {final_metrics['clean_loss']:.4f}")
    logger.info(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    logger.info(f"  Prover success rate: {final_metrics['prover_success_rate']:.3f}")

    output_dir = Path("results/nlp")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "experiment_name": config.experiment_name,
            "epsilon": config.epsilon,
            "seed": config.seed,
            "model_name": config.model_name,
            "message_dim": config.message_dim,
            "freeze_encoder": config.freeze_encoder,
            "train_samples": config.train_samples,
            "eval_samples": config.eval_samples,
            "num_rounds": config.num_rounds,
            "batch_size": config.batch_size,
            "prover_lr": config.prover_lr,
            "verifier_lr": config.verifier_lr,
        },
        "final_metrics": final_metrics,
        "history": history,
        "model_info": {
            "prover_params": n_prover_params,
            "verifier_params": n_verifier_params,
            "trainable_prover_params": n_trainable_prover,
            "trainable_verifier_params": n_trainable_verifier,
        },
    }

    output_path = output_dir / f"{config.experiment_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    checkpoint_dir = Path("checkpoints/nlp")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.experiment_name}.pt"
    trainer.save_checkpoint(str(checkpoint_path))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run NLP PVG experiment")

    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--message_dim", type=int, default=256)
    parser.add_argument("--freeze_encoder", action="store_true")

    parser.add_argument("--train_samples", type=int, default=50000)
    parser.add_argument("--eval_samples", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prover_lr", type=float, default=2e-5)
    parser.add_argument("--verifier_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--prover_steps", type=int, default=5)
    parser.add_argument("--verifier_steps", type=int, default=3)

    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=10)

    args = parser.parse_args()

    config = NLPConfig(
        experiment_name=f"snli_eps{args.epsilon}_seed{args.seed}",
        epsilon=args.epsilon,
        seed=args.seed,
        model_name=args.model_name,
        message_dim=args.message_dim,
        freeze_encoder=args.freeze_encoder,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        max_length=args.max_length,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        prover_lr=args.prover_lr,
        verifier_lr=args.verifier_lr,
        warmup_steps=args.warmup_steps,
        prover_steps_per_round=args.prover_steps,
        verifier_steps_per_round=args.verifier_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
