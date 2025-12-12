"""Metrics tracking and analysis utilities."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class MetricsTracker:
    """Track metrics during training."""

    def __init__(self) -> None:
        self.train_history: List[Dict] = []
        self.eval_history: List[Dict] = []

    def log_train(self, round_t: int, metrics: Dict) -> None:
        self.train_history.append({"round": round_t, **metrics})

    def log_eval(self, round_t: int, metrics: Dict) -> None:
        self.eval_history.append({"round": round_t, **metrics})

    def get_history(self) -> Dict:
        return {"train": self.train_history, "eval": self.eval_history}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.get_history(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        tracker = cls()
        with open(path) as f:
            data = json.load(f)
        tracker.train_history = data.get("train", [])
        tracker.eval_history = data.get("eval", [])
        return tracker


def compute_equilibrium_deviation(baseline_loss: float, noisy_loss: float) -> float:
    """Deviation from clean equilibrium: noisy_loss - baseline_loss."""
    return noisy_loss - baseline_loss


def aggregate_runs(results: List[Dict]) -> Dict:
    """Aggregate results across random seeds."""
    if not results:
        return {}

    metrics: Dict = {}
    metric_keys = results[0]["final_metrics"].keys()

    for key in metric_keys:
        values = [r["final_metrics"][key] for r in results]
        metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    return metrics


def analyze_noise_sweep(results_by_epsilon: Dict[float, List[Dict]]) -> Dict:
    """Analyze results from noise sweep; fits linear model for O(ε) hypothesis."""
    if 0.0 not in results_by_epsilon:
        raise ValueError("Baseline (ε=0) results required")

    baseline_losses = [r["final_metrics"]["clean_loss"] for r in results_by_epsilon[0.0]]
    baseline = float(np.mean(baseline_losses))

    analysis = {
        "noise_levels": [],
        "deviation_mean": [],
        "deviation_std": [],
        "accuracy_mean": [],
        "accuracy_std": [],
        "baseline_loss": baseline,
    }

    for eps in sorted(results_by_epsilon.keys()):
        losses = [r["final_metrics"]["clean_loss"] for r in results_by_epsilon[eps]]
        accuracies = [r["final_metrics"]["accuracy"] for r in results_by_epsilon[eps]]

        analysis["noise_levels"].append(eps)
        analysis["deviation_mean"].append(float(np.mean(losses)) - baseline)
        analysis["deviation_std"].append(float(np.std(losses)))
        analysis["accuracy_mean"].append(float(np.mean(accuracies)))
        analysis["accuracy_std"].append(float(np.std(accuracies)))

    eps_arr = np.array(analysis["noise_levels"])
    dev_arr = np.array(analysis["deviation_mean"])

    if len(eps_arr) >= 2:
        coeffs = np.polyfit(eps_arr, dev_arr, 1)
        linear_fit = np.poly1d(coeffs)

        ss_res = np.sum((dev_arr - linear_fit(eps_arr)) ** 2)
        ss_tot = np.sum((dev_arr - np.mean(dev_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        analysis["linear_fit"] = {
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "r_squared": float(r2),
        }

    return analysis


def save_analysis(analysis: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
