"""Tests for s4_encoding/graph_dataset.py feature_mode variants (P2.1)."""

from pathlib import Path

from s4_encoding.graph_dataset import GraphDataset

SAMPLE_GRAPH = Path("s1_data/graphs/canonical/creativity_0000.json")


def test_full_feature_mode_unchanged():
    dataset = GraphDataset([SAMPLE_GRAPH], [-1], feature_mode="full")
    data = dataset[0]
    assert data.x.shape[1] == 388


def test_structure_only_feature_mode():
    dataset = GraphDataset([SAMPLE_GRAPH], [-1], feature_mode="structure_only")
    assert dataset._label_encoder is None
    data = dataset[0]
    assert data.x.shape[1] == 5
    # First 4 dims are a one-hot type encoding.
    assert ((data.x[:, :4].sum(dim=1) == 1.0) | (data.x[:, :4].sum(dim=1) == 0.0)).all()
    # Last dim is non-negative degree.
    assert (data.x[:, 4] >= 0).all()
