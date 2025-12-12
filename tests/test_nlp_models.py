"""Tests for NLP (Transformer) models."""

import pytest
import torch


class TestTransformerModels:
    """Tests for Transformer prover/verifier models."""

    @pytest.fixture
    def models(self):
        from pvg.models import create_transformer_models
        return create_transformer_models(
            model_name="distilbert-base-uncased",
            message_dim=64,
            freeze_encoder=False,
        )

    @pytest.fixture
    def sample_batch(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        texts = [
            ("A man is playing a guitar.", "Someone is making music."),
            ("A dog runs in the park.", "An animal is outside."),
        ]
        encodings = tokenizer(
            [t[0] for t in texts],
            [t[1] for t in texts],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        return encodings["input_ids"], encodings["attention_mask"]

    def test_prover_output_shape(self, models, sample_batch):
        prover, _ = models
        input_ids, attention_mask = sample_batch
        message = prover(input_ids=input_ids, attention_mask=attention_mask)

        assert message.shape == (2, 64)
        assert (message >= -1).all() and (message <= 1).all()

    def test_verifier_output_shape(self, models, sample_batch):
        prover, verifier = models
        input_ids, attention_mask = sample_batch

        message = prover(input_ids=input_ids, attention_mask=attention_mask)
        logits = verifier(input_ids=input_ids, attention_mask=attention_mask, message=message)

        assert logits.shape == (2,)

    def test_gradients_flow(self, models, sample_batch):
        prover, verifier = models
        input_ids, attention_mask = sample_batch

        message = prover(input_ids=input_ids, attention_mask=attention_mask)
        logits = verifier(input_ids=input_ids, attention_mask=attention_mask, message=message)
        loss = logits.mean()
        loss.backward()

        prover_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in prover.parameters()
        )
        verifier_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in verifier.parameters()
        )

        assert prover_has_grad
        assert verifier_has_grad

    def test_frozen_encoder(self):
        from pvg.models import create_transformer_models

        prover, _ = create_transformer_models(
            model_name="distilbert-base-uncased",
            message_dim=64,
            freeze_encoder=True,
        )

        for param in prover.encoder.parameters():
            assert not param.requires_grad

        for param in prover.projection.parameters():
            assert param.requires_grad


class TestSNLIDataset:
    """Tests for SNLI dataset."""

    def test_dataset_loading(self):
        from experiments.nlp.snli_binary import SNLIBinaryDataset

        dataset = SNLIBinaryDataset(split="validation", max_samples=100, seed=42)

        assert len(dataset) == 100
        sample = dataset[0]
        assert "premise" in sample
        assert "hypothesis" in sample
        assert "label" in sample
        assert sample["label"] in [0, 1]

    def test_class_distribution(self):
        from experiments.nlp.snli_binary import SNLIBinaryDataset

        dataset = SNLIBinaryDataset(split="validation", max_samples=1000, seed=42)
        dist = dataset.get_class_distribution()

        assert 0.2 < dist["positive_rate"] < 0.8

    def test_dataloader_creation(self):
        from transformers import AutoTokenizer
        from experiments.nlp.snli_binary import create_snli_dataloaders

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        train_loader, _ = create_snli_dataloaders(
            tokenizer=tokenizer,
            batch_size=4,
            train_samples=16,
            eval_samples=8,
        )

        batch = next(iter(train_loader))
        assert batch.input_ids.shape[0] == 4
        assert batch.attention_mask.shape[0] == 4
        assert batch.labels.shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
