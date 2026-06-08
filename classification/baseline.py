"""Route 1 -- text-only baseline classifier.

Logistic regression on 768-dim sentence-transformer embeddings.
This is the control condition: Routes 2 and 3 must beat this to
demonstrate that concept graphs add predictive value.

Usage:
    uv run python classification/baseline.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from classification.split import load_split
from encoding.text_encoder import encode_transcripts

# Paths (relative to repo root)
CACHE_DIR = Path("cache")
MODEL_PATH = CACHE_DIR / "baseline_model.joblib"
VAL_PREDS_PATH = CACHE_DIR / "baseline_val_preds.npy"

# Reverse label map for pretty printing
LABEL_NAMES = {0: "workforce", 1: "creatives", 2: "scientists"}


def load_data() -> tuple:
    """Load text embeddings and split data, returning aligned arrays.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, val_ids, label_names).
    """
    # Load embeddings (cached -- will not re-encode)
    embeddings, embedding_ids = encode_transcripts()

    # Load split
    train_ids, val_ids, test_ids, labels_dict = load_split()

    # Build id -> index mapping for O(1) lookup
    id_to_idx = {tid: i for i, tid in enumerate(embedding_ids)}

    # Extract train/val subsets aligned with embeddings
    train_idx = [id_to_idx[tid] for tid in train_ids]
    val_idx = [id_to_idx[tid] for tid in val_ids]

    X_train = embeddings[train_idx]
    X_val = embeddings[val_idx]
    y_train = np.array([labels_dict[tid] for tid in train_ids], dtype=np.int32)
    y_val = np.array([labels_dict[tid] for tid in val_ids], dtype=np.int32)

    return X_train, y_train, X_val, y_val, val_ids, LABEL_NAMES


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Train logistic regression baseline.

    Args:
        X_train: (N, 768) embedding matrix.
        y_train: (N,) integer labels.

    Returns:
        Trained LogisticRegression model.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: LogisticRegression,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_names: dict[int, str],
) -> dict:
    """Evaluate model on validation set.

    Args:
        model: Trained classifier.
        X_val: (N, 768) validation embeddings.
        y_val: (N,) true labels.
        label_names: Mapping int label -> string name.

    Returns:
        Dictionary with macro_f1, per_class_f1, confusion_matrix, report.
    """
    y_pred = model.predict(X_val)

    macro_f1 = f1_score(y_val, y_pred, average="macro")
    per_class_f1 = f1_score(y_val, y_pred, average=None)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=list(label_names.values()))

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": {label_names[i]: float(f) for i, f in enumerate(per_class_f1)},
        "confusion_matrix": cm,
        "predictions": y_pred,
        "report": report,
    }


def print_results(results: dict) -> None:
    """Print formatted evaluation summary.

    Args:
        results: Dictionary from evaluate().
    """
    print("=" * 60)
    print("ROUTE 1 -- TEXT-ONLY BASELINE (Validation Set)")
    print("=" * 60)
    print(f"\nMacro F1: {results['macro_f1']:.4f}")
    print("\nPer-class F1:")
    for cls, score in results["per_class_f1"].items():
        print(f"  {cls:>12s}: {score:.4f}")
    print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
    print(f"\nClassification Report:\n{results['report']}")
    print("=" * 60)


def save_results(
    model: LogisticRegression,
    val_preds: np.ndarray,
) -> None:
    """Save trained model and validation predictions to cache.

    Args:
        model: Trained LogisticRegression.
        val_preds: (N,) predicted labels on validation set.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    np.save(VAL_PREDS_PATH, val_preds)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Validation predictions saved to {VAL_PREDS_PATH}")


def run_baseline() -> dict:
    """End-to-end baseline: load, train, evaluate, save.

    Returns:
        Evaluation results dictionary.
    """
    print("Loading data...")
    X_train, y_train, X_val, y_val, _val_ids, label_names = load_data()
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")

    print("Training logistic regression...")
    model = train_baseline(X_train, y_train)

    print("Evaluating on validation set...")
    results = evaluate(model, X_val, y_val, label_names)

    print_results(results)

    save_results(model, results["predictions"])

    return results


if __name__ == "__main__":
    run_baseline()
