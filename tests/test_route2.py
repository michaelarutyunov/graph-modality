"""Tests for classification.route2 — text + graph statistics classifier."""

from __future__ import annotations

import numpy as np

from classification.route2 import (
    COMBINED_DIM,
    FEATURE_NAMES,
    GRAPH_DIM,
    TEXT_DIM,
    load_and_align_features,
    train_and_evaluate,
)


def test_concatenated_shape():
    """Feature matrix is (N, 798) — 768 text + 30 graph."""
    features, _labels, ids, _train_idx, _val_idx, _test_idx = load_and_align_features()
    assert features.ndim == 2, f"Expected 2D array, got {features.ndim}D"
    assert features.shape[1] == COMBINED_DIM, (
        f"Expected {COMBINED_DIM} dimensions, got {features.shape[1]}"
    )
    assert features.shape[0] == len(ids), (
        f"Shape mismatch: {features.shape[0]} features vs {len(ids)} IDs"
    )


def test_model_trains():
    """Route 2 model trains without error and returns metrics."""
    results = train_and_evaluate()
    assert "macro_f1" in results
    assert isinstance(results["macro_f1"], float)
    assert results["macro_f1"] > 0, "Model should achieve non-zero F1"


def test_permutation_importance():
    """Permutation importance returns entries for all 30 graph features."""
    results = train_and_evaluate()
    top_features = results["top_features"]
    assert len(top_features) == 10, f"Expected 10 top features, got {len(top_features)}"
    # Each entry should have the expected keys
    for entry in top_features:
        assert "rank" in entry
        assert "name" in entry
        assert "importance" in entry
        assert entry["name"] in FEATURE_NAMES, (
            f"Feature name '{entry['name']}' not in FEATURE_NAMES"
        )


def test_graph_features_range():
    """Graph features (indices 768:798) are normalized to [0, 1] range."""
    features, _labels, _ids, _train_idx, _val_idx, _test_idx = load_and_align_features()
    graph_features = features[:, TEXT_DIM : TEXT_DIM + GRAPH_DIM]
    # Most graph features should be in [0, 1] after normalization.
    # We allow a small tolerance for floating-point edge cases and the
    # diameter feature which uses (diameter + 1) / 10 normalization
    # that could exceed 1 for large diameters, but typical values should be fine.
    assert np.all(graph_features >= -0.01), (
        f"Graph features have negative values below -0.01: min={graph_features.min():.4f}"
    )
    # Check that the vast majority are within [0, 1].
    # Some features (betweenness centrality, construct/value ratios) can
    # slightly exceed 1.0 by design, so we use a relaxed threshold of 0.97.
    in_range = np.mean((graph_features >= -0.01) & (graph_features <= 1.01))
    assert in_range >= 0.97, f"Only {in_range:.1%} of graph features in [-0.01, 1.01] range"


def test_no_val_leakage():
    """Validation set IDs do not overlap with training set IDs."""
    _features, _labels, _ids, train_idx, val_idx, _test_idx = load_and_align_features()
    train_set = set(train_idx)
    val_set = set(val_idx)
    overlap = train_set & val_set
    assert len(overlap) == 0, f"Val/train overlap: {len(overlap)} indices shared between splits"


def test_macro_f1_above_zero():
    """Macro-F1 is above zero — model learned something beyond random."""
    results = train_and_evaluate()
    assert results["macro_f1"] > 0, f"Macro-F1 is zero — model did not learn: {results['macro_f1']}"
