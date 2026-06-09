"""Tests for classification.route3 — text + GIN graph embedding classifier."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

os.environ["BEADS_DB"] = "/tmp/test_beads.db"

from s4_encoding.gnn.model import GraphEncoder

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_synthetic_batch(batch_size: int = 4, n_nodes: int = 3) -> Batch:
    """Create a synthetic PyG batch of graph data for testing the encoder.

    Each graph has ``n_nodes`` nodes with 388-dim features and 2 edges.
    """
    graphs = []
    for _ in range(batch_size):
        x = torch.randn(n_nodes, 388)
        # Create a simple edge list: 0->1, 1->2
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_gin_produces_128d_embeddings():
    """GraphEncoder.encode_graph() returns (batch_size, 128) embeddings."""
    model = GraphEncoder(in_channels=388, hidden=256, out_channels=128, n_classes=3)
    model.eval()

    batch = _make_synthetic_batch(batch_size=4, n_nodes=3)
    graph_emb = model.encode_graph(batch.x, batch.edge_index, batch.batch)

    assert graph_emb.shape == (4, 128), f"Expected shape (4, 128), got {graph_emb.shape}"
    assert graph_emb.dtype == torch.float32


def test_gin_forward_3_classes():
    """GraphEncoder.forward() returns (batch_size, 3) logits."""
    model = GraphEncoder(in_channels=388, hidden=256, out_channels=128, n_classes=3)
    model.eval()

    batch = _make_synthetic_batch(batch_size=4, n_nodes=3)
    text_embs = torch.randn(4, 768)

    with torch.no_grad():
        logits = model(batch, text_embs)

    assert logits.shape == (4, 3), f"Expected shape (4, 3), got {logits.shape}"


def test_best_model_exists():
    """cache/best_gin.pt should exist after GNN training (already run)."""
    best_model_path = Path("cache/best_gin.pt")
    if not best_model_path.exists():
        pytest.skip(
            f"GNN model not found at {best_model_path}; "
            "run 'uv run python encoding/gnn/train.py' first"
        )
    assert best_model_path.stat().st_size > 0, "Model file should not be empty"


def test_curves_exist():
    """cache/gnn_curves.png should exist after GNN training (already run)."""
    curves_path = Path("cache/gnn_curves.png")
    if not curves_path.exists():
        pytest.skip(
            f"Training curves not found at {curves_path}; "
            "run 'uv run python encoding/gnn/train.py' first"
        )
    assert curves_path.stat().st_size > 0, "Plot file should not be empty"


def test_val_macro_f1_recorded():
    """route3 results dict has macro_f1 as a float in [0, 1]."""
    results_path = Path("cache/route3_results.json")
    if not results_path.exists():
        pytest.skip("route3_results.json not found; run classification/route3.py first")

    with open(results_path) as f:
        results = json.load(f)

    assert "val_macro_f1" in results, "Results missing 'val_macro_f1' key"
    macro_f1 = results["val_macro_f1"]
    assert isinstance(macro_f1, float), f"macro_f1 should be float, got {type(macro_f1)}"
    assert 0.0 <= macro_f1 <= 1.0, f"macro_f1={macro_f1} outside [0, 1] range"


def test_delta_computed():
    """route3 comparison includes delta vs baseline when baseline model exists."""
    results_path = Path("cache/route3_results.json")
    if not results_path.exists():
        pytest.skip("route3_results.json not found; run classification/route3.py first")

    with open(results_path) as f:
        results = json.load(f)

    # delta_vs_baseline should exist if the baseline model was available
    if Path("cache/baseline_model.joblib").exists():
        assert "delta_vs_baseline" in results, "Results missing 'delta_vs_baseline' key"
        delta = results["delta_vs_baseline"]
        assert delta is not None, "delta_vs_baseline should not be None"
        assert isinstance(delta, float), f"delta should be float, got {type(delta)}"


def test_per_class_f1_recorded():
    """route3 results include per-class F1 for all three cohorts."""
    results_path = Path("cache/route3_results.json")
    if not results_path.exists():
        pytest.skip("route3_results.json not found; run classification/route3.py first")

    with open(results_path) as f:
        results = json.load(f)

    assert "per_class_f1" in results, "Results missing 'per_class_f1' key"
    per_class = results["per_class_f1"]
    for cohort in ["workforce", "creatives", "scientists"]:
        assert cohort in per_class, f"Missing per-class F1 for {cohort}"
        assert 0.0 <= per_class[cohort] <= 1.0, (
            f"F1 for {cohort}={per_class[cohort]} outside [0, 1]"
        )
