"""Route 2b — graph statistics ONLY (no text).

Classifies professional cohort from 30 hand-crafted graph statistics
without any text features.  Answers: do concept graph topologies alone
carry discriminative signal across cohorts?

Usage:
    uv run python classification/route2b.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from classification.split import load_split
from encoding.graph_stats import FEATURE_DIM, compute_all_stats

CACHE_DIR = Path("cache")
MODEL_PATH = CACHE_DIR / "route2b_model.joblib"

LABEL_MAP = {0: "workforce", 1: "creatives", 2: "scientists"}
LABEL_NAMES = [LABEL_MAP[i] for i in range(3)]


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load graph stats and split, returning train/val arrays.

    Returns:
        (X_train, y_train, X_val, y_val)
    """
    stats, stat_ids = compute_all_stats(graph_dir=Path("data/graphs/canonical"))
    train_ids, val_ids, _test_ids, labels_dict = load_split()

    id_to_idx = {tid: i for i, tid in enumerate(stat_ids)}

    train_idx = [id_to_idx[tid] for tid in train_ids if tid in id_to_idx]
    val_idx = [id_to_idx[tid] for tid in val_ids if tid in id_to_idx]

    X_train = stats[train_idx]
    X_val = stats[val_idx]
    y_train = np.array(
        [labels_dict[tid] for tid in train_ids if tid in id_to_idx], dtype=np.int32
    )
    y_val = np.array(
        [labels_dict[tid] for tid in val_ids if tid in id_to_idx], dtype=np.int32
    )

    return X_train, y_train, X_val, y_val


def train_and_evaluate() -> dict:
    """Train graph-only classifier and evaluate on validation set.

    Returns:
        Dictionary with macro_f1, per_class_f1, confusion_matrix, report.
    """
    X_train, y_train, X_val, y_val = load_data()
    print(f"Train: {X_train.shape[0]} samples, {FEATURE_DIM} features")
    print(f"Val:   {X_val.shape[0]} samples")

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        random_state=42,
    )
    print("\nTraining logistic regression (graph stats only)...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    per_class = f1_score(y_val, y_pred, average=None)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES)

    print("=" * 60)
    print("ROUTE 2b — GRAPH STATISTICS ONLY (Validation Set)")
    print("=" * 60)
    print(f"\nMacro F1: {macro_f1:.4f}  (chance = 0.3333)")
    print("\nPer-class F1:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:>12s}: {per_class[i]:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")
    print("=" * 60)

    # Save model
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return {
        "route": "route2b_graph_stats_only",
        "macro_f1": macro_f1,
        "per_class_f1": {LABEL_MAP[i]: float(per_class[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_dim": FEATURE_DIM,
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    print(f"\nDone. Validation macro-F1: {results['macro_f1']:.4f}")
