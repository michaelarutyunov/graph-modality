"""Tests for class-weighted loss in s5_classification/train_loop.py."""

from __future__ import annotations

import numpy as np

from s5_classification.classifiers import build_classifier
from s5_classification.train_loop import Trainer, TrainingConfig


def _imbalanced_data(seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic single-modality data with a rare class 2 (10 of 210)."""
    rng = np.random.default_rng(seed)
    counts = {0: 100, 1: 100, 2: 10}
    feats, labels = [], []
    for cls, n in counts.items():
        feats.append(rng.normal(loc=cls, scale=0.5, size=(n, 8)))
        labels.extend([cls] * n)
    return {
        "text_emb": np.concatenate(feats).astype(np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def _trainer(class_weight: str | None) -> Trainer:
    model = build_classifier(
        architecture="single", modality_dims={"text": 8}, n_classes=3, hidden=16
    )
    cfg = TrainingConfig(n_classes=3, max_epochs=2, class_weight=class_weight, seed=0)
    return Trainer(model, cfg)


def test_balanced_weights_favour_minority_class():
    """class_weight='balanced' sets inverse-frequency weights from train labels."""
    data = _imbalanced_data()
    trainer = _trainer("balanced")
    trainer.fit(data, data)

    weight = trainer.criterion.weight
    assert weight is not None
    # Rare class 2 (n=10) must get a larger weight than common classes 0/1 (n=100).
    assert float(weight[2]) > float(weight[0])
    assert float(weight[2]) > float(weight[1])
    # sklearn "balanced" formula: n_samples / (n_classes * count).
    assert float(weight[2]) == 210 / (3 * 10)


def test_default_loss_is_unweighted():
    """Without class_weight, the criterion stays unweighted (back-compat)."""
    trainer = _trainer(None)
    assert trainer.criterion.weight is None
