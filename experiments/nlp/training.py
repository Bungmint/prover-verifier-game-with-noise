"""NLP-adapted Stackelberg trainer for Transformer models."""

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
from experiments.nlp.snli_binary import SNLIBatch

logger = logging.getLogger(__name__)


class NLPStackelbergTrainer:
    """Stackelberg trainer for NLP tasks with Transformer models."""

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
        self._global_step = 0

        self.prover_optimizer = AdamW(
            prover.parameters(), lr=config.prover_lr, weight_decay=0.01
        )
        self.verifier_optimizer = AdamW(
            verifier.parameters(), lr=config.verifier_lr, weight_decay=0.01
        )

        warmup_steps = getattr(config, "warmup_steps", 500)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        self.prover_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.prover_optimizer, lr_lambda
        )
        self.verifier_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.verifier_optimizer, lr_lambda
        )

        self.device = torch.device(config.device)
        self.prover.to(self.device)
        self.verifier.to(self.device)

        self.history: Dict[str, list] = {"train": [], "eval": []}
        self.use_wandb = getattr(config, "use_wandb", False)

    def _get_batch(self) -> SNLIBatch:
        """Get next batch, cycling through dataset."""
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)

        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)

        return batch.to(self.device)

    def train(self) -> Dict[str, list]:
        """Run Stackelberg training loop."""
        logger.info(f"Starting NLP Stackelberg training: ε={self.config.epsilon}")
        logger.info(f"  model={self.config.model_name}")
        logger.info(f"  num_rounds={self.config.num_rounds}")
        logger.info(f"  prover_steps={self.config.prover_steps_per_round}")
        logger.info(f"  verifier_steps={self.config.verifier_steps_per_round}")

        for round_t in tqdm(range(self.config.num_rounds), desc="Training"):
            self._freeze_verifier()
            self._unfreeze_prover()

            prover_losses = []
            for _ in range(self.config.prover_steps_per_round):
                prover_losses.append(self._prover_step())
                self._global_step += 1

            self._freeze_prover()
            self._unfreeze_verifier()

            verifier_losses = []
            for _ in range(self.config.verifier_steps_per_round):
                verifier_losses.append(self._verifier_step())
                self._global_step += 1

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
        batch = self._get_batch()

        message = self.prover(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )

        logits = self.verifier(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            message=message,
        )

        loss = F.softplus(-logits).mean()

        self.prover_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.prover.parameters(), max_norm=1.0)
        self.prover_optimizer.step()
        self.prover_scheduler.step()

        return loss.item()

    def _verifier_step(self) -> float:
        """Verifier minimizes BCE with (noisy) labels."""
        batch = self._get_batch()
        noisy_labels = inject_label_noise(batch.labels, self.config.epsilon)

        with torch.no_grad():
            message = self.prover(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )

        logits = self.verifier(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            message=message,
        )

        loss = F.binary_cross_entropy_with_logits(logits, noisy_labels.float())

        self.verifier_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=1.0)
        self.verifier_optimizer.step()
        self.verifier_scheduler.step()

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
            for batch in self.eval_loader:
                batch = batch.to(self.device)

                message = self.prover(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )

                logits = self.verifier(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    message=message,
                )

                loss = F.binary_cross_entropy_with_logits(
                    logits, batch.labels.float(), reduction="sum"
                )
                total_loss += loss.item()

                preds = (logits > 0).long()
                total_correct += (preds == batch.labels).sum().item()
                total_prover_wins += preds.sum().item()
                total_samples += len(batch.labels)

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

        p_lr = self.prover_scheduler.get_last_lr()[0]
        v_lr = self.verifier_scheduler.get_last_lr()[0]

        self.history["train"].append({
            "round": round_t,
            "prover_loss": avg_p,
            "verifier_loss": avg_v,
            "prover_lr": p_lr,
            "verifier_lr": v_lr,
        })

        logger.info(
            f"Round {round_t}: P_loss={avg_p:.4f}, V_loss={avg_v:.4f}, "
            f"P_lr={p_lr:.2e}, V_lr={v_lr:.2e}"
        )
        
        if self.use_wandb:
            wandb.log({
                "train/prover_loss": avg_p,
                "train/verifier_loss": avg_v,
                "train/prover_lr": p_lr,
                "train/verifier_lr": v_lr,
                "round": round_t,
            })

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "prover_state_dict": self.prover.state_dict(),
            "verifier_state_dict": self.verifier.state_dict(),
            "prover_optimizer_state_dict": self.prover_optimizer.state_dict(),
            "verifier_optimizer_state_dict": self.verifier_optimizer.state_dict(),
            "prover_scheduler_state_dict": self.prover_scheduler.state_dict(),
            "verifier_scheduler_state_dict": self.verifier_scheduler.state_dict(),
            "global_step": self._global_step,
            "history": self.history,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.prover.load_state_dict(checkpoint["prover_state_dict"])
        self.verifier.load_state_dict(checkpoint["verifier_state_dict"])
        self.prover_optimizer.load_state_dict(checkpoint["prover_optimizer_state_dict"])
        self.verifier_optimizer.load_state_dict(checkpoint["verifier_optimizer_state_dict"])
        self.prover_scheduler.load_state_dict(checkpoint["prover_scheduler_state_dict"])
        self.verifier_scheduler.load_state_dict(checkpoint["verifier_scheduler_state_dict"])
        self._global_step = checkpoint["global_step"]
        self.history = checkpoint["history"]
        logger.info(f"Checkpoint loaded from {path}")
