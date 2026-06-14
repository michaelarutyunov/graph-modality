"""Single-modality ablation probe (P2.4 — Method-Review Phase 2).

Standalone logistic-regression probe over a single frozen embedding array,
used to compare five graph-vs-labels variants (text, label-bag, structure-only
GIN, full GIN, masked GIN) without registering them in
``train_config.MODALITY_DIMS`` or ``build_sweep`` (per the bead's pinned
resolution — keeps the Phase-1 production sweep untouched).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from s4_encoding.build_dataset import _load_ai_adoption_labels, _load_ambivalence_labels
from s5_classification.repeated_eval import make_split
from s5_classification.split import load_transcript_ids_with_labels


@cache
def _load_embeddings(emb_path: str, ids_path: str) -> tuple[np.ndarray, dict[str, int]]:
    """Load an embedding array + id->index lookup, cached across calls."""
    embeddings = np.load(emb_path)
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    lookup = {tid: i for i, tid in enumerate(ids)}
    return embeddings, lookup


def _load_labels(target: str) -> dict[str, int]:
    if target == "cohort":
        return load_transcript_ids_with_labels()
    if target == "ai_adoption":
        return _load_ai_adoption_labels()
    if target == "stance_ambivalence":
        return _load_ambivalence_labels()
    raise ValueError(f"Unknown target: {target!r}")


def probe_variant(emb_path: str, ids_path: str, target: str, seed: int) -> dict:
    """Fit a single-modality logistic-regression probe for one variant/target/seed.

    Args:
        emb_path: Path to a ``.npy`` file of shape ``(n_graphs, dim)``.
        ids_path: Path to the sibling ``*_ids.json`` (transcript_ids, aligned
            to ``emb_path`` row order).
        target: ``"cohort"`` or ``"ai_adoption"``.
        seed: Protocol seed (0-9), used both for the split and the classifier.

    Returns:
        Dict with ``val_macro_f1``, ``test_macro_f1``, ``predictions``,
        ``labels`` (test set, for paired McNemar-style comparisons).
    """
    embeddings, lookup = _load_embeddings(emb_path, ids_path)
    labels_dict = _load_labels(target)

    train_ids, val_ids, test_ids = make_split(target, seed)

    def _slice(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        missing = [tid for tid in ids if tid not in lookup]
        if missing:
            raise ValueError(f"Missing ids in {emb_path}: {missing}")
        idx = [lookup[tid] for tid in ids]
        x = embeddings[idx]
        y = np.array([labels_dict[tid] for tid in ids], dtype=np.int64)
        return x, y

    x_train, y_train = _slice(train_ids)
    x_val, y_val = _slice(val_ids)
    x_test, y_test = _slice(test_ids)

    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(x_train, y_train)

    val_preds = model.predict(x_val)
    val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))

    test_preds = model.predict(x_test)
    test_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

    return {
        "val_macro_f1": val_f1,
        "test_macro_f1": test_f1,
        "predictions": test_preds.tolist(),
        "labels": y_test.tolist(),
    }
