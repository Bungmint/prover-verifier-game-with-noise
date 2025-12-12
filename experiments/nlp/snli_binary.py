"""Binary SNLI dataset: entailment → 1, contradiction/neutral → 0."""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


@dataclass
class SNLIBatch:
    """Batch container for SNLI data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> "SNLIBatch":
        return SNLIBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


class SNLIBinaryDataset(Dataset):
    """SNLI converted to binary: entailment → 1, else → 0."""

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        dataset = load_dataset("stanfordnlp/snli", split=split)
        dataset = dataset.filter(lambda x: x["label"] != -1)

        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(max_samples))

        self.premises: List[str] = dataset["premise"]
        self.hypotheses: List[str] = dataset["hypothesis"]
        self.labels: List[int] = [1 if label == 0 else 0 for label in dataset["label"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "premise": self.premises[idx],
            "hypothesis": self.hypotheses[idx],
            "label": self.labels[idx],
        }

    def get_class_distribution(self) -> Dict[str, float]:
        """Return class distribution statistics."""
        n_positive = sum(self.labels)
        n_total = len(self.labels)
        return {
            "positive_rate": n_positive / n_total,
            "negative_rate": (n_total - n_positive) / n_total,
            "n_positive": n_positive,
            "n_negative": n_total - n_positive,
            "n_total": n_total,
        }


class SNLICollator:
    """Collate and tokenize SNLI batches."""

    def __init__(self, tokenizer, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> SNLIBatch:
        premises = [ex["premise"] for ex in batch]
        hypotheses = [ex["hypothesis"] for ex in batch]
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)

        encodings = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return SNLIBatch(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            labels=labels,
        )


def create_snli_dataloaders(
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    train_samples: Optional[int] = None,
    eval_samples: Optional[int] = None,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for binary SNLI."""
    train_dataset = SNLIBinaryDataset(split="train", max_samples=train_samples, seed=seed)
    eval_dataset = SNLIBinaryDataset(split="validation", max_samples=eval_samples, seed=seed + 1000)
    collator = SNLICollator(tokenizer=tokenizer, max_length=max_length)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, eval_loader


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("Loading SNLI dataset...")
    dataset = SNLIBinaryDataset(split="train", max_samples=1000)
    dist = dataset.get_class_distribution()

    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dist['positive_rate']:.2%} positive (entailment)")

    print("\nTesting with DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader, eval_loader = create_snli_dataloaders(
        tokenizer=tokenizer,
        batch_size=8,
        train_samples=100,
        eval_samples=50,
    )

    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch.input_ids.shape}")
    print(f"Batch attention_mask shape: {batch.attention_mask.shape}")
    print(f"Batch labels shape: {batch.labels.shape}")
