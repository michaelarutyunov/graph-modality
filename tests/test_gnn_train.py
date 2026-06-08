"""Tests for encoding/gnn/train.py module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

os.environ["BEADS_DB"] = "/tmp/test_beads.db"

from encoding.gnn.train import (
    compute_class_weights,
    gather_text_embeddings,
    run_training_loop,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_fake_graph_data(n: int, label: int, tid_prefix: str = "test") -> list[Data]:
    """Create n synthetic PyG Data objects for testing.

    Uses random features of the correct dimensionality (388) so the model
    can run a forward pass without errors.
    """
    graphs = []
    for i in range(n):
        n_nodes = 5  # small fixed graph size
        x = torch.randn(n_nodes, 388)
        edge_index = torch.randint(0, n_nodes, (2, 6))
        graphs.append(
            Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(label),
                transcript_id=f"{tid_prefix}_{i}",
            )
        )
    return graphs


def _make_fake_setup(n_train=9, n_val=3):
    """Create fake train/val loaders, text embeddings, and labels.

    Returns (train_loader, val_loader, text_emb_dict, train_labels).
    Each class gets n_train/3 train graphs and n_val/3 val graphs.
    """
    train_graphs = []
    train_labels = []
    train_tids = []
    for label in range(3):
        graphs = _make_fake_graph_data(n_train // 3, label, tid_prefix=f"train_{label}")
        train_graphs.extend(graphs)
        train_labels.extend([label] * len(graphs))
        train_tids.extend([g.transcript_id for g in graphs])

    val_graphs = []
    for label in range(3):
        graphs = _make_fake_graph_data(n_val // 3, label, tid_prefix=f"val_{label}")
        val_graphs.extend(graphs)

    all_tids = train_tids + [g.transcript_id for g in val_graphs]
    text_emb_dict = {tid: torch.randn(768) for tid in all_tids}

    train_loader = DataLoader(train_graphs, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=3, shuffle=False)

    return train_loader, val_loader, text_emb_dict, train_labels


# ── Pure function tests (no I/O) ────────────────────────────────────────────


def test_class_weights():
    """Class weights are computed from training labels only; minority class gets higher weight."""
    labels = [0] * 100 + [1] * 50 + [2] * 10
    weights = compute_class_weights(labels)

    assert weights.shape == (3,), f"Expected shape (3,), got {weights.shape}"

    # Class 2 (minority, count=10) should have highest weight
    assert weights[2] > weights[1], (
        f"Minority class (2) should have higher weight than class 1: {weights}"
    )
    assert weights[2] > weights[0], (
        f"Minority class (2) should have higher weight than class 0: {weights}"
    )

    # Class 0 (majority, count=100) should have lowest weight
    assert weights[0] < weights[1], f"Majority class (0) should have lowest weight: {weights}"

    # Verify the weights are inverse frequency: w_i = N / (C * count_i)
    n = len(labels)
    for i, count in enumerate([100, 50, 10]):
        expected = n / (3 * count)
        assert abs(weights[i].item() - expected) < 1e-6, (
            f"Weight for class {i}: expected {expected:.4f}, got {weights[i].item():.4f}"
        )


def test_class_weights_balanced():
    """Balanced labels produce equal weights."""
    labels = [0] * 50 + [1] * 50 + [2] * 50
    weights = compute_class_weights(labels)

    assert abs(weights[0].item() - weights[1].item()) < 1e-6
    assert abs(weights[1].item() - weights[2].item()) < 1e-6


def test_gather_text_embeddings():
    """gather_text_embeddings correctly aligns embeddings to batch IDs."""
    emb_dict = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0, 4.0]),
        "c": torch.tensor([5.0, 6.0]),
    }
    result = gather_text_embeddings(["c", "a", "b"], emb_dict)
    expected = torch.tensor([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


# ── Training loop tests (use synthetic data, mock _prepare_data) ─────────────


def test_training_loop_runs():
    """Training loop runs without error on small synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        best_model = Path(tmpdir) / "best.pt"
        curves = Path(tmpdir) / "curves.png"
        with (
            patch("encoding.gnn.train.BEST_MODEL_PATH", best_model),
            patch("encoding.gnn.train.CURVES_PATH", curves),
        ):
            train_loader, val_loader, emb_dict, labels = _make_fake_setup()
            results = run_training_loop(
                train_loader,
                val_loader,
                emb_dict,
                labels,
                max_epochs=2,
                early_stopping_patience=5,
            )

    assert "best_val_f1" in results
    assert "epochs_run" in results
    assert results["epochs_run"] >= 1
    assert results["best_val_f1"] >= 0.0


def test_best_model_saved():
    """Best model checkpoint saved after training."""
    best_model = Path("cache/best_gin_test_check.pt")
    curves = Path("cache/gnn_curves_test_check.png")

    try:
        with (
            patch("encoding.gnn.train.BEST_MODEL_PATH", best_model),
            patch("encoding.gnn.train.CURVES_PATH", curves),
        ):
            train_loader, val_loader, emb_dict, labels = _make_fake_setup()
            run_training_loop(
                train_loader,
                val_loader,
                emb_dict,
                labels,
                max_epochs=2,
                early_stopping_patience=5,
            )

        assert best_model.exists(), "Best model checkpoint should be saved"

        # Verify it's a valid state dict
        state_dict = torch.load(best_model, weights_only=True)
        assert "conv1" in state_dict or "classifier.0.weight" in state_dict, (
            "Saved file should contain model state dict keys"
        )
    finally:
        best_model.unlink(missing_ok=True)
        curves.unlink(missing_ok=True)


def test_curves_saved():
    """Train/val curves plot file exists after training."""
    curves = Path("cache/gnn_curves_test_check2.png")
    best_model = Path("cache/best_gin_test_check2.pt")

    try:
        with (
            patch("encoding.gnn.train.BEST_MODEL_PATH", best_model),
            patch("encoding.gnn.train.CURVES_PATH", curves),
        ):
            train_loader, val_loader, emb_dict, labels = _make_fake_setup()
            run_training_loop(
                train_loader,
                val_loader,
                emb_dict,
                labels,
                max_epochs=2,
                early_stopping_patience=5,
            )

        assert curves.exists(), "Training curves plot should be saved"
        assert curves.stat().st_size > 0, "Plot file should not be empty"
    finally:
        curves.unlink(missing_ok=True)
        best_model.unlink(missing_ok=True)


def test_early_stopping():
    """Early stopping triggers before max epochs on very small data."""
    best_model = Path("cache/best_gin_test_es.pt")
    curves = Path("cache/gnn_curves_test_es.png")

    try:
        with (
            patch("encoding.gnn.train.BEST_MODEL_PATH", best_model),
            patch("encoding.gnn.train.CURVES_PATH", curves),
        ):
            # Use extremely small data so model overfits fast
            train_loader, val_loader, emb_dict, labels = _make_fake_setup(
                n_train=3,
                n_val=3,
            )
            results = run_training_loop(
                train_loader,
                val_loader,
                emb_dict,
                labels,
                max_epochs=30,
                early_stopping_patience=3,
            )

        # With 1 graph per class, val F1 cannot keep improving
        assert results["epochs_run"] < 30, (
            f"Early stopping should trigger before max epochs. "
            f"Ran {results['epochs_run']} epochs out of 30."
        )
    finally:
        best_model.unlink(missing_ok=True)
        curves.unlink(missing_ok=True)
