"""Route 2 classifier: text embeddings + hand-crafted graph statistics.

Concatenates 768-dim sentence-transformer embeddings with 30-dim graph
statistic features for a 798-dim input vector, then trains a balanced
logistic regression to classify professional cohort.

Usage:
    uv run python classification/route2.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from classification.split import load_split
from encoding.graph_stats import compute_all_stats
from encoding.text_encoder import encode_transcripts

# ── Feature metadata ────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # structural (7)
    "node_count_norm",
    "edge_count_norm",
    "density",
    "component_ratio",
    "avg_degree_norm",
    "max_degree_norm",
    "diameter_norm",
    # node type distribution (6)
    "construct_ratio",
    "value_ratio",
    "stance_ratio",
    "csm_ratio",
    "construct_value_ratio",
    "stance_construct_ratio",
    # construct quality (3)
    "bipolarity_score",
    "mean_construct_degree",
    "max_construct_degree",
    # stance valence (8)
    "positive_stance_frac",
    "negative_stance_frac",
    "mixed_stance_frac",
    "ambivalent_stance_frac",
    "valence_positive",
    "valence_negative",
    "valence_mixed",
    "valence_ambivalent",
    "valence_absent",
    # centrality (3)
    "max_betweenness",
    "mean_betweenness",
    "max_value_betweenness",
    # cognitive style (2)
    "csm_present",
    "csm_count_clipped",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "node_count_norm": "normalized total node count",
    "edge_count_norm": "normalized total edge count",
    "density": "graph edge density",
    "component_ratio": "ratio of connected components to nodes",
    "avg_degree_norm": "normalized average node degree",
    "max_degree_norm": "normalized maximum node degree",
    "diameter_norm": "normalized graph diameter",
    "construct_ratio": "fraction of Construct nodes",
    "value_ratio": "fraction of Value nodes",
    "stance_ratio": "fraction of Stance nodes",
    "csm_ratio": "fraction of CognitiveStyleMarker nodes",
    "construct_value_ratio": "ratio of Constructs to Values",
    "stance_construct_ratio": "ratio of Stances to Constructs",
    "bipolarity_score": "completeness of construct bipolarity across graph",
    "mean_construct_degree": "average degree of Construct nodes",
    "max_construct_degree": "maximum degree of any Construct node",
    "positive_stance_frac": "fraction of Stances with positive valence",
    "negative_stance_frac": "fraction of Stances with negative valence",
    "mixed_stance_frac": "fraction of Stances with mixed valence",
    "ambivalent_stance_frac": "fraction of Stances with ambivalent valence",
    "valence_positive": "dominant valence is positive (binary)",
    "valence_negative": "dominant valence is negative (binary)",
    "valence_mixed": "dominant valence is mixed (binary)",
    "valence_ambivalent": "dominant valence is ambivalent (binary)",
    "valence_absent": "no dominant valence (no stances present) (binary)",
    "max_betweenness": "maximum betweenness centrality",
    "mean_betweenness": "mean betweenness centrality",
    "max_value_betweenness": "max betweenness centrality among Value nodes",
    "csm_present": "at least one CognitiveStyleMarker present",
    "csm_count_clipped": "CognitiveStyleMarker count (clipped to 2)",
}

TEXT_DIM = 768
GRAPH_DIM = 30
COMBINED_DIM = TEXT_DIM + GRAPH_DIM  # 798

CACHE_DIR = Path("cache")
MODEL_PATH = CACHE_DIR / "route2_model.joblib"
BASELINE_MODEL_PATH = CACHE_DIR / "baseline_model.joblib"
LABEL_MAP = {0: "workforce", 1: "creatives", 2: "scientists"}


def load_and_align_features(
    graph_dir: Path = Path("data/graphs/canonical"),
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Load text embeddings and graph stats, align by transcript ID.

    Args:
        graph_dir: Directory of canonicalised graph JSON files.

    Returns:
        (features, labels, ids, train_indices) where:
        - features: (N, 798) concatenated feature matrix
        - labels: (N,) integer label array
        - ids: ordered list of transcript IDs
        - train_indices: indices into features/labels that belong to train split
    """
    # Load embeddings
    text_emb, text_ids = encode_transcripts()
    print(f"text embeddings: {text_emb.shape}")

    # Load graph stats from canonical graphs
    graph_stats, graph_ids = compute_all_stats(graph_dir=graph_dir)
    print(f"graph stats: {graph_stats.shape}")

    # Align both by transcript ID
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    graph_id_to_idx = {tid: i for i, tid in enumerate(graph_ids)}

    # Find common IDs
    common_ids = sorted(set(text_ids) & set(graph_ids))
    if len(common_ids) == 0:
        raise ValueError(
            f"No common transcript IDs between text embeddings ({len(text_ids)}) "
            f"and graph stats ({len(graph_ids)})"
        )
    print(f"common transcripts: {len(common_ids)}")

    # Build aligned arrays
    text_aligned = np.array([text_emb[text_id_to_idx[tid]] for tid in common_ids], dtype=np.float32)
    graph_aligned = np.array(
        [graph_stats[graph_id_to_idx[tid]] for tid in common_ids], dtype=np.float32
    )

    # Concatenate: [text (768) | graph (30)]
    features = np.concatenate([text_aligned, graph_aligned], axis=1)
    assert features.shape[1] == COMBINED_DIM, (
        f"Expected {COMBINED_DIM}-dim features, got {features.shape[1]}"
    )

    # Load split and labels
    train_ids, val_ids, test_ids, labels_dict = load_split()

    # Build label array aligned to features
    labels = np.array([labels_dict[tid] for tid in common_ids], dtype=np.int32)

    # Create index masks for splits
    train_indices = [i for i, tid in enumerate(common_ids) if tid in set(train_ids)]
    val_indices = [i for i, tid in enumerate(common_ids) if tid in set(val_ids)]
    # test_indices not needed for validation-only evaluation, but compute for reference
    test_indices = [i for i, tid in enumerate(common_ids) if tid in set(test_ids)]

    print(f"train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}")

    return features, labels, common_ids, train_indices, val_indices, test_indices


def train_and_evaluate() -> dict:
    """Train Route 2 classifier and evaluate on validation set.

    Returns:
        Dictionary with model, metrics, and importances.
    """
    features, labels, _ids, train_idx, val_idx, _test_idx = load_and_align_features()

    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_val = features[val_idx]
    y_val = labels[val_idx]

    # Train logistic regression
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    print("\ntraining logistic regression...")
    clf.fit(X_train, y_train)
    print("training complete.")

    # Predictions
    y_pred = clf.predict(X_val)

    # Metrics
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    per_class_f1 = f1_score(y_val, y_pred, average=None)
    cm = confusion_matrix(y_val, y_pred)

    print(f"\n{'=' * 60}")
    print("ROUTE 2: Text + Graph Statistics (798-dim)")
    print(f"{'=' * 60}")
    print(f"\nValidation macro-F1: {macro_f1:.4f}")
    print("\nPer-class F1:")
    for label_idx, name in LABEL_MAP.items():
        print(f"  {name}: {per_class_f1[label_idx]:.4f}")

    print("\nClassification report:")
    target_names = [LABEL_MAP[i] for i in range(3)]
    print(classification_report(y_val, y_pred, target_names=target_names))

    print("Confusion matrix:")
    print(cm)

    # Compare with baseline if available
    if BASELINE_MODEL_PATH.exists():
        baseline_clf = joblib.load(BASELINE_MODEL_PATH)
        baseline_pred = baseline_clf.predict(X_val[:, :TEXT_DIM])
        baseline_macro_f1 = f1_score(y_val, baseline_pred, average="macro")
        delta = macro_f1 - baseline_macro_f1
        print(f"\nBaseline macro-F1: {baseline_macro_f1:.4f}")
        print(f"Delta (route2 - baseline): {delta:+.4f}")
    else:
        baseline_macro_f1 = None
        print("\n(baseline model not found at cache/baseline_model.joblib)")

    # Permutation importance on graph features only (indices TEXT_DIM:COMBINED_DIM)
    print("\ncomputing permutation importance (graph features, n_repeats=10)...")
    perm_result = permutation_importance(
        clf,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        scoring="f1_macro",
    )

    # Extract graph feature importances (indices TEXT_DIM through COMBINED_DIM)
    graph_importances = perm_result.importances_mean[TEXT_DIM:COMBINED_DIM]
    graph_importances_std = perm_result.importances_std[TEXT_DIM:COMBINED_DIM]

    # Top-10 table
    top_indices = np.argsort(graph_importances)[::-1][:10]
    print("\nTop-10 graph features by permutation importance:")
    print(f"{'rank':>4}  {'feature':<28}  {'importance':>10}  {'description'}")
    print("-" * 80)
    for rank, idx in enumerate(top_indices, start=1):
        fname = FEATURE_NAMES[idx]
        imp = graph_importances[idx]
        desc = FEATURE_DESCRIPTIONS[fname]
        print(f"{rank:>4}  {fname:<28}  {imp:>10.6f}  {desc}")

    # Save model
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nmodel saved to {MODEL_PATH}")

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "baseline_macro_f1": float(baseline_macro_f1) if baseline_macro_f1 is not None else None,
        "delta": float(delta) if baseline_macro_f1 is not None else None,
        "top_features": [
            {
                "rank": rank,
                "name": FEATURE_NAMES[idx],
                "importance": float(graph_importances[idx]),
                "std": float(graph_importances_std[idx]),
            }
            for rank, idx in enumerate(top_indices, start=1)
        ],
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    print(f"\nDone. Validation macro-F1: {results['macro_f1']:.4f}")
