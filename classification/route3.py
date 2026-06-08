"""Route 3 classifier: text embeddings + GIN graph embedding.

Fuses 128-dim GIN graph embeddings (from a trained GraphEncoder) with
768-dim text embeddings and classifies professional cohort through an MLP head.

If the GIN has already been trained (``cache/best_gin.pt`` exists), this module
skips re-training and loads the saved weights. Otherwise it calls the training
loop from ``encoding/gnn/train.py``.

Usage:
    uv run python classification/route3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from classification.split import load_split
from encoding.gnn.model import GraphEncoder
from encoding.gnn.train import (
    BEST_MODEL_PATH,
    GRAPH_EMB_DIM,
    IN_CHANNELS,
    N_CLASSES,
    _prepare_data,
    compute_class_weights,
    gather_text_embeddings,
)
from encoding.gnn.train import (
    evaluate as gnn_evaluate,
)
from encoding.gnn.train import (
    train as gnn_train,
)
from encoding.text_encoder import encode_transcripts

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
RESULTS_PATH = CACHE_DIR / "route3_results.json"
BASELINE_MODEL_PATH = CACHE_DIR / "baseline_model.joblib"
ROUTE2_MODEL_PATH = CACHE_DIR / "route2_model.joblib"

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "workforce", 1: "creatives", 2: "scientists"}
LABEL_NAMES = [LABEL_MAP[i] for i in range(3)]


def run_route3() -> dict:
    """End-to-end Route 3: train (if needed), evaluate, compare, save.

    Returns:
        Dictionary with val macro-F1, per-class F1, comparison deltas, and
        confusion matrix.
    """
    # ── Step 1: Train GIN if no cached model ─────────────────────────────────
    if not BEST_MODEL_PATH.exists():
        print("No cached GIN model found. Training from scratch...")
        train_results = gnn_train()
        print(
            f"GNN training complete. Best val macro-F1: {train_results['best_val_f1']:.4f} "
            f"at epoch {train_results['best_epoch']}"
        )
    else:
        print(f"Loading cached GIN model from {BEST_MODEL_PATH}")

    # ── Step 2: Load val data and model for evaluation ────────────────────────
    print("\nPreparing validation data...")
    _train_loader, val_loader, text_emb_dict, train_labels = _prepare_data()

    device: torch.device = torch.device("cpu")  # pyright: ignore[reportPrivateImportUsage]

    # Recreate model and load best weights
    model = GraphEncoder(
        in_channels=IN_CHANNELS,
        out_channels=GRAPH_EMB_DIM,
        n_classes=N_CLASSES,
    ).to(device)

    state_dict = torch.load(BEST_MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Step 3: Evaluate on validation set ───────────────────────────────────
    class_weights = compute_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print("\nEvaluating Route 3 on validation set...")
    val_loss, val_macro_f1 = gnn_evaluate(model, val_loader, text_emb_dict, criterion, device)

    # Collect predictions for per-class metrics
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            transcript_ids = batch.transcript_id
            text_embs = gather_text_embeddings(transcript_ids, text_emb_dict).to(device)
            logits = model(batch, text_embs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    per_class_f1: npt.NDArray[np.float64] = np.asarray(
        f1_score(all_labels, all_preds, average=None)  # pyright: ignore[reportArgumentType]
    )
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES)

    # ── Step 4: Compare against Route 1 and Route 2 ─────────────────────────
    print("\n" + "=" * 60)
    print("ROUTE 3 -- TEXT + GIN GRAPH EMBEDDING (Validation Set)")
    print("=" * 60)
    print(f"\nMacro F1: {val_macro_f1:.4f}")
    print("\nPer-class F1:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:>12s}: {per_class_f1[i]:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")

    # Load baseline for comparison
    baseline_f1 = None
    delta_baseline = None
    if BASELINE_MODEL_PATH.exists():
        _X_train, _y_train, X_val, y_val, _val_ids, _label_names = _load_baseline_data()
        baseline_clf = joblib.load(BASELINE_MODEL_PATH)
        baseline_preds = baseline_clf.predict(X_val)
        baseline_f1 = float(f1_score(y_val, baseline_preds, average="macro"))
        delta_baseline = val_macro_f1 - baseline_f1

    route2_f1 = None
    delta_route2 = None
    if ROUTE2_MODEL_PATH.exists():
        from classification.route2 import load_and_align_features

        features, labels_arr, _ids, _train_idx, val_idx, _test_idx = load_and_align_features()  # pyright: ignore[reportAssignmentType]
        X_val_r2 = features[val_idx]
        y_val_r2 = labels_arr[val_idx]
        route2_clf = joblib.load(ROUTE2_MODEL_PATH)
        route2_preds = route2_clf.predict(X_val_r2)
        route2_f1 = float(f1_score(y_val_r2, route2_preds, average="macro"))
        delta_route2 = val_macro_f1 - route2_f1

    # ── Print comparison table ───────────────────────────────────────────────
    _print_comparison(baseline_f1, delta_baseline, route2_f1, delta_route2, val_macro_f1)

    # ── Step 5: Save results ─────────────────────────────────────────────────
    results = {
        "route": "route3",
        "description": "text + GIN graph embedding",
        "val_macro_f1": val_macro_f1,
        "per_class_f1": {LABEL_MAP[i]: float(per_class_f1[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "val_loss": val_loss,
        "baseline_macro_f1": baseline_f1,
        "delta_vs_baseline": delta_baseline,
        "route2_macro_f1": route2_f1,
        "delta_vs_route2": delta_route2,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


def _load_baseline_data() -> tuple:
    """Load text embeddings and split data for baseline evaluation.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, val_ids, label_names).
    """
    embeddings, embedding_ids = encode_transcripts()
    train_ids, val_ids, _test_ids, labels_dict = load_split()

    id_to_idx = {tid: i for i, tid in enumerate(embedding_ids)}
    train_idx = [id_to_idx[tid] for tid in train_ids]
    val_idx = [id_to_idx[tid] for tid in val_ids]

    X_train = embeddings[train_idx]
    X_val = embeddings[val_idx]
    y_train = np.array([labels_dict[tid] for tid in train_ids], dtype=np.int32)
    y_val = np.array([labels_dict[tid] for tid in val_ids], dtype=np.int32)

    label_names = {0: "workforce", 1: "creatives", 2: "scientists"}
    return X_train, y_train, X_val, y_val, val_ids, label_names


def _print_comparison(
    baseline_f1: float | None,
    delta_baseline: float | None,
    route2_f1: float | None,
    delta_route2: float | None,
    route3_f1: float,
) -> None:
    """Print formatted route comparison table."""
    print("\n" + "=" * 60)
    print("ROUTE COMPARISON (Validation Set)")
    print("=" * 60)

    if baseline_f1 is not None:
        print(f"Route 1 (Text-only):     macro-F1 = {baseline_f1:.4f}")
    else:
        print("Route 1 (Text-only):     (model not found)")

    if route2_f1 is not None and baseline_f1 is not None:
        print(
            f"Route 2 (Text + Stats):  macro-F1 = {route2_f1:.4f}  "
            f"(Δ = {delta_route2:+.4f} vs baseline)"
        )
    elif route2_f1 is not None:
        print(f"Route 2 (Text + Stats):  macro-F1 = {route2_f1:.4f}")
    else:
        print("Route 2 (Text + Stats):  (model not found)")

    if delta_baseline is not None:
        print(
            f"Route 3 (Text + GIN):    macro-F1 = {route3_f1:.4f}  "
            f"(Δ = {delta_baseline:+.4f} vs baseline)"
        )
    else:
        print(f"Route 3 (Text + GIN):    macro-F1 = {route3_f1:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    results = run_route3()
    print(f"\nDone. Validation macro-F1: {results['val_macro_f1']:.4f}")
