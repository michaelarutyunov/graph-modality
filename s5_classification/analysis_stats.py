"""
Loads a pre-trained sklearn model (stats-only, trained by ``classification/run.py``
with backend=sklearn, modalities=[\"stats\"]) and reports per-class metrics.

Training is now handled by the unified experiment runner:
    uv run python classification/run.py --sweep sklearn --target cohort

Usage:
    uv run python classification/route2b.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from s4_encoding.graph_stats_encoder import compute_all_stats
from s5_classification.split import load_split

CACHE_DIR = Path("cache")
DEFAULT_MODEL_PATH = CACHE_DIR / "route2b_model.joblib"
LABEL_MAP = {0: "workforce", 1: "creatives", 2: "scientists"}


def analyze(model_path: Path | None = None) -> dict:
    """Load a trained stats-only model and report metrics.

    Args:
        model_path: Path to a joblib-saved sklearn model.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Train it first with: uv run python classification/run.py --sweep sklearn"
        )

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Load stats aligned to val split
    stats, stat_ids = compute_all_stats(graph_dir=Path("s1_data/graphs/canonical"))
    s_map = {tid: i for i, tid in enumerate(stat_ids)}

    _train_ids, val_ids, _test_ids, labels_dict = load_split()

    idx = [s_map[t] for t in val_ids if t in s_map]
    X_val = stats[idx]
    y_val = np.array([labels_dict[t] for t in val_ids if t in s_map], dtype=np.int32)

    preds = model.predict(X_val)
    f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
    per_class = f1_score(y_val, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_val, preds)

    print(f"\n{'=' * 60}")
    print("ROUTE 2b: Graph Statistics ONLY (30-dim)")
    print(f"{'=' * 60}")
    print(f"Val macro-F1: {f1:.4f}")
    print("\nPer-class F1:")
    for i, name in LABEL_MAP.items():
        print(f"  {name}: {per_class[i]:.4f}")
    print("\nClassification report:")
    print(
        classification_report(y_val, preds, target_names=list(LABEL_MAP.values()), zero_division=0)
    )
    print(f"Confusion matrix:\n{cm}")

    return {
        "macro_f1": f1,
        "per_class_f1": {LABEL_MAP[i]: float(per_class[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    results = analyze()
    print(f"\nDone. Val macro-F1: {results['macro_f1']:.4f}")
