"""Stackelberg (bilevel) training loop for Prover-Verifier Games."""

import logging
from typing import Dict, Optional, Iterator, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from pvg.noise import inject_label_noise

logger = logging.getLogger(__name__)


class StackelbergTrainer:
    """Bilevel Stackelberg trainer: prover best-responds, then verifier optimizes."""

    def __init__(
        self,
        prover: nn.Module,
        verifier: nn.Module,
        config: Any,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ) -> None:
        self.prover = prover
        self.verifier = verifier
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self._train_iter: Optional[Iterator] = None

        self.prover_optimizer = AdamW(
            prover.parameters(), lr=config.prover_lr, weight_decay=0.01
        )
        self.verifier_optimizer = AdamW(
            verifier.parameters(), lr=config.verifier_lr, weight_decay=0.01
        )

        self.device = torch.device(config.device)
        self.prover.to(self.device)
        self.verifier.to(self.device)

        self.history: Dict[str, list] = {"train": [], "eval": []}
        self.use_wandb = getattr(config, "use_wandb", False)

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
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

    def train(self) -> Dict[str, list]:
        """Run Stackelberg training loop."""
        logger.info(f"Starting Stackelberg training: ε={self.config.epsilon}")
        logger.info(f"  num_rounds={self.config.num_rounds}")
        logger.info(f"  prover_steps={self.config.prover_steps_per_round}")
        logger.info(f"  verifier_steps={self.config.verifier_steps_per_round}")

        for round_t in tqdm(range(self.config.num_rounds), desc="Training"):
            # Inner loop: prover best-responds to fixed verifier
            self._freeze_verifier()
            self._unfreeze_prover()

            prover_losses = []
            for _ in range(self.config.prover_steps_per_round):
                prover_losses.append(self._prover_step())

            # Outer loop: verifier optimizes against prover's best response
            self._freeze_prover()
            self._unfreeze_verifier()

            verifier_losses = []
            for _ in range(self.config.verifier_steps_per_round):
                verifier_losses.append(self._verifier_step())

            if round_t % self.config.log_every == 0:
                self._log_round(round_t, prover_losses, verifier_losses)

            if round_t % self.config.eval_every == 0 and self.eval_loader is not None:
                eval_metrics = self.evaluate()
                self.history["eval"].append({"round": round_t, **eval_metrics})
                logger.info(
                    f"  Eval: acc={eval_metrics['accuracy']:.3f}, "
                    f"clean_loss={eval_metrics['clean_loss']:.4f}"
                )
                
                if self.use_wandb:
                    wandb.log({
                        "eval/clean_loss": eval_metrics["clean_loss"],
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval/prover_success_rate": eval_metrics["prover_success_rate"],
                        "round": round_t,
                    })

        return self.history

    def _freeze_verifier(self) -> None:
        self.verifier.eval()
        for param in self.verifier.parameters():
            param.requires_grad = False

    def _unfreeze_verifier(self) -> None:
        self.verifier.train()
        for param in self.verifier.parameters():
            param.requires_grad = True

    def _freeze_prover(self) -> None:
        self.prover.eval()
        for param in self.prover.parameters():
            param.requires_grad = False

    def _unfreeze_prover(self) -> None:
        self.prover.train()
        for param in self.prover.parameters():
            param.requires_grad = True

    def _prover_step(self) -> float:
        """Prover minimizes -log p(y=1) to convince verifier."""
        x, _ = self._get_batch()

        z = self.prover(x)
        logits = self.verifier(x, z)
        loss = F.softplus(-logits).mean()

        self.prover_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.prover.parameters(), max_norm=1.0)
        self.prover_optimizer.step()

        return loss.item()

    def _verifier_step(self) -> float:
        """Verifier minimizes BCE with (noisy) labels."""
        x, y = self._get_batch()
        noisy_y = inject_label_noise(y, self.config.epsilon)

        with torch.no_grad():
            z = self.prover(x)

        logits = self.verifier(x, z)
        loss = F.binary_cross_entropy_with_logits(logits, noisy_y.float())

        self.verifier_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=1.0)
        self.verifier_optimizer.step()

        return loss.item()

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on clean labels."""
        self.prover.eval()
        self.verifier.eval()

        total_loss = 0.0
        total_correct = 0
        total_prover_wins = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)

                z = self.prover(x)
                logits = self.verifier(x, z)

                loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="sum")
                total_loss += loss.item()

                preds = (logits > 0).long()
                total_correct += (preds == y).sum().item()
                total_prover_wins += preds.sum().item()
                total_samples += len(y)

        return {
            "clean_loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "prover_success_rate": total_prover_wins / total_samples,
        }

    def _log_round(
        self, round_t: int, prover_losses: list[float], verifier_losses: list[float]
    ) -> None:
        avg_p = sum(prover_losses) / len(prover_losses)
        avg_v = sum(verifier_losses) / len(verifier_losses)

        self.history["train"].append({
            "round": round_t,
            "prover_loss": avg_p,
            "verifier_loss": avg_v,
        })

        logger.info(f"Round {round_t}: P_loss={avg_p:.4f}, V_loss={avg_v:.4f}")
        
        if self.use_wandb:
            wandb.log({
                "train/prover_loss": avg_p,
                "train/verifier_loss": avg_v,
                "round": round_t,
            })

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "prover_state_dict": self.prover.state_dict(),
            "verifier_state_dict": self.verifier.state_dict(),
            "prover_optimizer_state_dict": self.prover_optimizer.state_dict(),
            "verifier_optimizer_state_dict": self.verifier_optimizer.state_dict(),
            "history": self.history,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.prover.load_state_dict(checkpoint["prover_state_dict"])
        self.verifier.load_state_dict(checkpoint["verifier_state_dict"])
        self.prover_optimizer.load_state_dict(checkpoint["prover_optimizer_state_dict"])
        self.verifier_optimizer.load_state_dict(checkpoint["verifier_optimizer_state_dict"])
        self.history = checkpoint["history"]
        logger.info(f"Checkpoint loaded from {path}")
