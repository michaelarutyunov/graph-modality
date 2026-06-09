"""Route 2 analysis: permutation importance of graph features.

Loads a pre-trained sklearn model (trained by ``classification/run.py`` with
backend=sklearn) and computes permutation importance to identify which
graph-statistic features drive classification.

Training is now handled by the unified experiment runner:
    uv run python classification/run.py --sweep sklearn --target cohort

Usage:
    uv run python classification/route2.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from s5_classification.split import load_split
from s4_encoding.graph_stats_encoder import compute_all_stats
from s4_encoding.text_encoder import encode_transcripts

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
DEFAULT_MODEL_PATH = CACHE_DIR / "route2_model.joblib"
LABEL_MAP = {0: "workforce", 1: "creatives", 2: "scientists"}
TEXT_DIM = 768
GRAPH_DIM = 30

FEATURE_NAMES: list[str] = [
    "node_count_norm", "edge_count_norm", "density", "component_ratio",
    "avg_degree_norm", "max_degree_norm", "diameter_norm",
    "construct_ratio", "value_ratio", "stance_ratio", "csm_ratio",
    "construct_value_ratio", "stance_construct_ratio",
    "bipolarity_score", "mean_construct_degree", "max_construct_degree",
    "positive_stance_frac", "negative_stance_frac", "mixed_stance_frac",
    "ambivalent_stance_frac",
    "valence_positive", "valence_negative", "valence_mixed",
    "valence_ambivalent", "valence_absent",
    "max_betweenness", "mean_betweenness", "max_value_betweenness",
    "csm_present", "csm_count_clipped",
]


def load_aligned_features():
    """Load text + graph stats features aligned by transcript ID.

    Returns:
        (features, labels, val_features, val_labels) — features are
        (N, 798) concatenated [text|stats] matrices.
    """
    text_emb, text_ids = encode_transcripts()
    graph_stats, graph_ids = compute_all_stats(graph_dir=Path("s1_data/graphs/canonical"))

    t_map = {tid: i for i, tid in enumerate(text_ids)}
    s_map = {tid: i for i, tid in enumerate(graph_ids)}
    common = sorted(set(text_ids) & set(graph_ids))

    text_aligned = np.array([text_emb[t_map[t]] for t in common], dtype=np.float32)
    graph_aligned = np.array([graph_stats[s_map[t]] for t in common], dtype=np.float32)
    features = np.concatenate([text_aligned, graph_aligned], axis=1)

    train_ids, val_ids, _test_ids, labels_dict = load_split()
    train_set = set(train_ids)
    val_set = set(val_ids)

    train_idx = [i for i, t in enumerate(common) if t in train_set]
    val_idx = [i for i, t in enumerate(common) if t in val_set]

    labels = np.array([labels_dict[t] for t in common], dtype=np.int32)

    return (
        features[train_idx], labels[train_idx],
        features[val_idx], labels[val_idx],
    )


def analyze(model_path: Path | None = None) -> dict:
    """Load a trained model and compute permutation importance.

    Args:
        model_path: Path to a joblib-saved sklearn model. Defaults to
            ``cache/route2_model.joblib``.
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

    X_train, y_train, X_val, y_val = load_aligned_features()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Baseline F1
    val_preds = model.predict(X_val)
    baseline_f1 = f1_score(y_val, val_preds, average="macro")
    print(f"Val macro-F1: {baseline_f1:.4f}")

    # Permutation importance (graph features only: cols TEXT_DIM:)
    print("\nComputing permutation importance (graph features, n_repeats=10)...")
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=10, random_state=42, scoring="f1_macro",
    )

    graph_imp = result.importances_mean[TEXT_DIM:]
    graph_std = result.importances_std[TEXT_DIM:]

    # Top-10
    top_idx = np.argsort(graph_imp)[::-1][:10]
    print(f"\n{'='*80}")
    print("Top-10 graph features by permutation importance")
    print(f"{'='*80}")
    print(f"{'rank':>4}  {'feature':<30}  {'importance':>10}  {'std':>8}")
    print("-" * 80)
    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank:>4}  {FEATURE_NAMES[idx]:<30}  {graph_imp[idx]:>10.6f}  {graph_std[idx]:>8.6f}")

    return {
        "baseline_f1": float(baseline_f1),
        "top_features": [
            {"rank": r, "name": FEATURE_NAMES[i],
             "importance": float(graph_imp[i]), "std": float(graph_std[i])}
            for r, i in enumerate(top_idx, start=1)
        ],
    }


if __name__ == "__main__":
    results = analyze()
    print(f"\nDone. Val F1: {results['baseline_f1']:.4f}")
