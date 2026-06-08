"""Final evaluation on the held-out test set — run exactly once.

Evaluates all three trained routes on the test split and produces:
- Per-route macro-F1, per-class F1, confusion matrix
- Comparison table (baseline vs route 2 vs route 3 with Δ)
- Confusion matrix grid figure (1x3)
- Timestamped JSON results (never overwritten)

Usage:
    uv run python classification/test_evaluation.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch_geometric.loader import DataLoader

from classification.split import load_split
from encoding.gnn.model import GraphEncoder
from encoding.gnn.train import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CANONICAL_DIR,
    COHORT_TO_LABEL,
    GRAPH_EMB_DIM,
    IN_CHANNELS,
    N_CLASSES,
    PREFIX_TO_COHORT,
    build_prefix_to_transcript_id,
    compute_class_weights,
    gather_text_embeddings,
    load_text_embedding_dict,
    precompute_graph_data,
)
from encoding.graph_stats import (
    FEATURE_DIM as GRAPH_STATS_DIM,
)
from encoding.graph_stats import (
    compute_all_stats,
)
from encoding.text_encoder import encode_transcripts

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
BASELINE_MODEL_PATH = CACHE_DIR / "baseline_model.joblib"
ROUTE2_MODEL_PATH = CACHE_DIR / "route2_model.joblib"
RESULTS_LOG_PATH = Path(".claude/context/results-log.md")

TEXT_DIM = 768
COMBINED_DIM = TEXT_DIM + GRAPH_STATS_DIM  # 798

LABEL_MAP = {0: "workforce", 1: "creatives", 2: "scientists"}
LABEL_NAMES = [LABEL_MAP[i] for i in range(3)]


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _get_timestamp() -> str:
    """ISO 8601 UTC timestamp for result filenames."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def _build_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    route_name: str,
    extra: dict | None = None,
) -> dict:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        route_name: Human-readable route label.
        extra: Optional extra fields to merge into result.

    Returns:
        Dictionary with macro_f1, per_class_f1, confusion_matrix, report.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)

    result: dict = {
        "route": route_name,
        "macro_f1": macro_f1,
        "per_class_f1": {LABEL_MAP[i]: float(per_class[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_samples": len(y_true),
    }
    if extra:
        result.update(extra)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Route 1 — Text-only baseline
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_route1(
    test_ids: list[str],
    labels_dict: dict[str, int],
    text_embeddings: np.ndarray,
    text_embedding_ids: list[str],
) -> dict:
    """Evaluate text-only logistic regression baseline on test set.

    Args:
        test_ids: Transcript IDs in the test split.
        labels_dict: Mapping from transcript_id → integer label.
        text_embeddings: (N, 768) matrix of sentence-transformer embeddings.
        text_embedding_ids: Ordered list of transcript IDs for the embedding rows.

    Returns:
        Metrics dictionary.
    """
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found at {BASELINE_MODEL_PATH}. "
            "Run classification/baseline.py first."
        )

    print(f"\nLoading baseline model from {BASELINE_MODEL_PATH}...")
    model = joblib.load(BASELINE_MODEL_PATH)

    # Align test IDs with embedding indices
    id_to_idx = {tid: i for i, tid in enumerate(text_embedding_ids)}
    test_idx = [id_to_idx[tid] for tid in test_ids if tid in id_to_idx]

    if len(test_idx) < len(test_ids):
        missing = len(test_ids) - len(test_idx)
        print(f"  Warning: {missing} test IDs have no text embedding (skipped)")

    X_test = text_embeddings[test_idx]
    y_test = np.array([labels_dict[tid] for tid in test_ids if tid in id_to_idx], dtype=np.int32)

    print(f"  Test samples: {X_test.shape[0]}")

    y_pred = model.predict(X_test)

    return _build_metrics_dict(y_test, y_pred, "route1_text_only")


# ═══════════════════════════════════════════════════════════════════════════════
# Route 2 — Text + graph statistics
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_route2(
    test_ids: list[str],
    labels_dict: dict[str, int],
) -> dict:
    """Evaluate text + graph statistics logistic regression on test set.

    Loads text embeddings and graph stats, aligns by transcript ID,
    extracts test split, and evaluates the trained Route 2 model.

    Args:
        test_ids: Transcript IDs in the test split.
        labels_dict: Mapping from transcript_id → integer label.

    Returns:
        Metrics dictionary.
    """
    if not ROUTE2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Route 2 model not found at {ROUTE2_MODEL_PATH}. Run classification/route2.py first."
        )

    print(f"\nLoading Route 2 model from {ROUTE2_MODEL_PATH}...")
    model = joblib.load(ROUTE2_MODEL_PATH)

    # Load and align features
    text_emb, text_ids = encode_transcripts()
    graph_stats, graph_stats_ids = compute_all_stats(
        graph_dir=Path("data/graphs/canonical"),
    )

    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    graph_id_to_idx = {tid: i for i, tid in enumerate(graph_stats_ids)}

    common_ids = sorted(set(text_ids) & set(graph_stats_ids))
    if len(common_ids) == 0:
        raise ValueError("No common IDs between text embeddings and graph stats.")

    # Build aligned feature matrix
    text_aligned = np.array([text_emb[text_id_to_idx[tid]] for tid in common_ids], dtype=np.float32)
    graph_aligned = np.array(
        [graph_stats[graph_id_to_idx[tid]] for tid in common_ids], dtype=np.float32
    )
    features = np.concatenate([text_aligned, graph_aligned], axis=1)

    # Extract test subset
    test_id_set = set(test_ids)
    test_indices = [i for i, tid in enumerate(common_ids) if tid in test_id_set]
    y_test = np.array([labels_dict[common_ids[i]] for i in test_indices], dtype=np.int32)

    X_test = features[test_indices]
    print(f"  Test samples: {X_test.shape[0]} (feature dim: {X_test.shape[1]})")

    y_pred = model.predict(X_test)

    return _build_metrics_dict(y_test, y_pred, "route2_text_stats")


# ═══════════════════════════════════════════════════════════════════════════════
# Route 3 — Text + GIN graph embedding
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare_test_loader() -> tuple[DataLoader, dict[str, torch.Tensor], list[int]]:
    """Prepare test DataLoader and associated data for GNN evaluation.

    Mirrors ``_prepare_data`` from ``encoding/gnn/train.py`` but for the
    held-out test split instead of train/val.

    Returns:
        (test_loader, text_emb_dict, test_labels)
    """
    print("\nPreparing test data for Route 3...")

    # Load split
    _train_ids, _val_ids, test_ids, _labels = load_split()

    # Map graph files to transcript IDs
    prefix_to_tid = build_prefix_to_transcript_id(CANONICAL_DIR)
    tid_to_graph_path: dict[str, Path] = {}
    for stem, tid in prefix_to_tid.items():
        tid_to_graph_path[tid] = CANONICAL_DIR / f"{stem}.json"

    # Build test file/label lists
    test_paths = [tid_to_graph_path[tid] for tid in test_ids if tid in tid_to_graph_path]
    test_labels = [
        COHORT_TO_LABEL[PREFIX_TO_COHORT[tid.rsplit("_", 1)[0]]]
        for tid in test_ids
        if tid in tid_to_graph_path
    ]

    missing = len(test_ids) - len(test_paths)
    if missing:
        print(f"  Warning: {missing} test IDs have no graph file (skipped)")

    print(f"  Test graphs: {len(test_paths)}")

    # Load text embeddings
    text_emb_dict = load_text_embedding_dict()

    # Precompute graph data (encodes node labels via sentence-transformers)
    test_data = precompute_graph_data(test_paths, test_labels)
    print("  Test graphs loaded.")

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return test_loader, text_emb_dict, test_labels


def evaluate_route3() -> dict:
    """Evaluate text + GIN classifier on test set.

    Loads the best GIN checkpoint, runs inference on the test DataLoader,
    and computes all metrics.

    Returns:
        Metrics dictionary.
    """
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GIN model not found at {BEST_MODEL_PATH}. "
            "Run encoding/gnn/train.py or classification/route3.py first."
        )

    print(f"\nLoading GIN model from {BEST_MODEL_PATH}...")

    # Prepare test data
    test_loader, text_emb_dict, test_labels = _prepare_test_loader()

    device = torch.device("cpu")

    # Recreate model architecture and load best weights
    model = GraphEncoder(
        in_channels=IN_CHANNELS,
        out_channels=GRAPH_EMB_DIM,
        n_classes=N_CLASSES,
    ).to(device)

    state_dict = torch.load(BEST_MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Compute class weights from test labels for the loss (reporting only)
    class_weights = compute_class_weights(test_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Run evaluation
    print("  Evaluating on test set...")
    test_loss, test_macro_f1 = _gnn_evaluate_on_loader(
        model, test_loader, text_emb_dict, criterion, device
    )

    # Collect per-class predictions for full metrics
    all_preds, all_labels = _collect_gnn_predictions(model, test_loader, text_emb_dict, device)

    per_class = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES)

    result: dict = {
        "route": "route3_text_gin",
        "macro_f1": float(test_macro_f1),
        "per_class_f1": {LABEL_MAP[i]: float(per_class[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_samples": len(all_labels),
        "test_loss": float(test_loss),
    }

    print(f"  Test samples: {len(all_labels)}")
    print(f"  Test macro-F1: {test_macro_f1:.4f}")
    print(f"  Test loss: {test_loss:.4f}")

    return result


@torch.no_grad()
def _gnn_evaluate_on_loader(
    model: GraphEncoder,
    loader: DataLoader,
    text_emb_dict: dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Compute loss and macro-F1 on a DataLoader.

    Returns:
        (average_loss, macro_f1)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        transcript_ids = batch.transcript_id
        text_embs = gather_text_embeddings(transcript_ids, text_emb_dict).to(device)

        logits = model(batch, text_embs)
        loss = criterion(logits, batch.y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, macro_f1


@torch.no_grad()
def _collect_gnn_predictions(
    model: GraphEncoder,
    loader: DataLoader,
    text_emb_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Collect all predictions and labels from a DataLoader.

    Returns:
        (all_preds, all_labels) as plain Python lists.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        batch = batch.to(device)
        transcript_ids = batch.transcript_id
        text_embs = gather_text_embeddings(transcript_ids, text_emb_dict).to(device)

        logits = model(batch, text_embs)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())

    return all_preds, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison & Output
# ═══════════════════════════════════════════════════════════════════════════════


def print_comparison_table(
    r1: dict,
    r2: dict,
    r3: dict,
) -> None:
    """Print formatted route comparison on test set."""
    delta_r2 = r2["macro_f1"] - r1["macro_f1"]
    delta_r3 = r3["macro_f1"] - r1["macro_f1"]

    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION — ROUTE COMPARISON")
    print("=" * 70)
    print(f"{'Route':<30} {'macro-F1':>10} {'Δ vs baseline':>14}")
    print("-" * 70)
    print(f"{'Route 1 (Text-only)':<30} {r1['macro_f1']:>10.4f} {'—':>14}")
    print(f"{'Route 2 (Text + Stats)':<30} {r2['macro_f1']:>10.4f} {delta_r2:>+14.4f}")
    print(f"{'Route 3 (Text + GIN)':<30} {r3['macro_f1']:>10.4f} {delta_r3:>+14.4f}")
    print("-" * 70)

    # Per-class breakdown
    print("\nPer-class F1:")
    print(f"{'Class':<14} {'Route 1':>10} {'Route 2':>10} {'Route 3':>10}")
    print("-" * 50)
    for cls_name in LABEL_NAMES:
        print(
            f"{cls_name:<14} "
            f"{r1['per_class_f1'][cls_name]:>10.4f} "
            f"{r2['per_class_f1'][cls_name]:>10.4f} "
            f"{r3['per_class_f1'][cls_name]:>10.4f}"
        )

    print("\nConfusion matrices:")
    for route_name, results in [
        ("Route 1 (Text-only)", r1),
        ("Route 2 (Text + Stats)", r2),
        ("Route 3 (Text + GIN)", r3),
    ]:
        print(f"\n  {route_name}:")
        cm = np.array(results["confusion_matrix"])
        print(f"    {LABEL_NAMES[0]:>12} {LABEL_NAMES[1]:>12} {LABEL_NAMES[2]:>12}")
        for i, name in enumerate(LABEL_NAMES):
            print(f"    {name:<12} {cm[i, 0]:>4} {cm[i, 1]:>4} {cm[i, 2]:>4}")

    print("=" * 70)


def plot_confusion_matrices(
    r1: dict,
    r2: dict,
    r3: dict,
    save_path: Path,
) -> None:
    """Render 1x3 confusion matrix grid and save as PNG.

    Args:
        r1, r2, r3: Result dictionaries with 'confusion_matrix' key.
        save_path: Output path for the PNG figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    route_data = [
        ("Route 1: Text-only", r1["confusion_matrix"], r1["macro_f1"]),
        ("Route 2: Text + Stats", r2["confusion_matrix"], r2["macro_f1"]),
        ("Route 3: Text + GIN", r3["confusion_matrix"], r3["macro_f1"]),
    ]

    for ax, (title, cm_list, f1) in zip(axes, route_data, strict=True):
        cm = np.array(cm_list)
        ax.imshow(cm, cmap="Blues", vmin=0)

        # Annotate cells
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        ax.set_xticks(range(3))
        ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels(LABEL_NAMES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(f"{title}\nmacro-F1 = {f1:.4f}", fontsize=12, fontweight="bold")

    # Use the last axes' image for shared colorbar (all share same vmin/vmax scale)
    fig.colorbar(axes[-1].images[0], ax=axes, fraction=0.02, pad=0.04, label="Count")
    fig.suptitle(
        "Test Set Confusion Matrices — Professional Cohort Classification",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix grid saved to {save_path}")


def build_comparison_dict(
    r1: dict,
    r2: dict,
    r3: dict,
    timestamp: str,
) -> dict:
    """Build combined comparison summary for JSON export."""
    return {
        "evaluation": "test_set",
        "timestamp": timestamp,
        "split_info": {
            "train": 875,
            "val": 187,
            "test": 188,
            "seed": 42,
        },
        "routes": {
            "route1_text_only": {
                "macro_f1": r1["macro_f1"],
                "per_class_f1": r1["per_class_f1"],
                "n_samples": r1["n_samples"],
            },
            "route2_text_stats": {
                "macro_f1": r2["macro_f1"],
                "per_class_f1": r2["per_class_f1"],
                "n_samples": r2["n_samples"],
                "delta_vs_baseline": r2["macro_f1"] - r1["macro_f1"],
            },
            "route3_text_gin": {
                "macro_f1": r3["macro_f1"],
                "per_class_f1": r3["per_class_f1"],
                "n_samples": r3["n_samples"],
                "delta_vs_baseline": r3["macro_f1"] - r1["macro_f1"],
            },
        },
        "comparison_table": {
            "route": ["Route 1 (Text-only)", "Route 2 (Text + Stats)", "Route 3 (Text + GIN)"],
            "test_macro_f1": [r1["macro_f1"], r2["macro_f1"], r3["macro_f1"]],
            "delta_vs_baseline": [
                0.0,
                r2["macro_f1"] - r1["macro_f1"],
                r3["macro_f1"] - r1["macro_f1"],
            ],
        },
    }


def update_results_log(
    r1: dict,
    r2: dict,
    r3: dict,
    timestamp: str,
) -> None:
    """Update results-log.md with test set metrics.

    Idempotent: safely re-runnable — tracks route sections by header
    to avoid the identical-placeholder bug that would assign all three
    routes the same value.
    """
    log_path = RESULTS_LOG_PATH
    lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)

    delta_r2 = r2["macro_f1"] - r1["macro_f1"]
    delta_r3 = r3["macro_f1"] - r1["macro_f1"]

    # Values keyed by the route section header they belong under
    route_values = {
        "Route 1": f"| test macro-F1 | **{r1['macro_f1']:.4f}** |\n",
        "Route 2": (
            f"| test macro-F1 | **{r2['macro_f1']:.4f}** (Δ = {delta_r2:+.4f} vs baseline) |\n"
        ),
        "Route 3": (
            f"| test macro-F1 | **{r3['macro_f1']:.4f}** (Δ = {delta_r3:+.4f} vs baseline) |\n"
        ),
    }

    # ── Pass 1: replace per-route test macro-F1 lines ───────────────────
    current_route: str | None = None
    new_lines: list[str] = []
    for line in lines:
        # Track which route section we're inside
        if "### Route 1" in line:
            current_route = "Route 1"
        elif "### Route 2" in line:
            current_route = "Route 2"
        elif "### Route 3" in line:
            current_route = "Route 3"
        elif line.startswith("### Route comparison") or line.startswith("## Phase 4"):
            current_route = None

        if current_route and "| test macro-F1 |" in line:
            new_lines.append(route_values[current_route])
        else:
            new_lines.append(line)

    updated = "".join(new_lines)

    # ── Pass 2: update the route comparison table ───────────────────────
    old_comparison = (
        "| route | val macro-F1 | test macro-F1 | Δ vs baseline (test) |\n"
        "|---|---|---|---|\n"
        f"| Baseline (text-only) | 1.0000 | {r1['macro_f1']:.4f} | — |\n"
        f"| Route 2 (text + stats) | 1.0000 | {r2['macro_f1']:.4f} | {delta_r2:+.4f} |\n"
        f"| Route 3 (text + GIN) | 0.9797 | {r3['macro_f1']:.4f} | {delta_r3:+.4f} |"
    )

    # Try the old-format table first (pre-update), then the new-format one
    old_table_v1 = (
        "| route | val macro-F1 | test macro-F1 | Δ vs baseline |\n"
        "|---|---|---|---|\n"
        "| Baseline (text-only) | 1.0000 | — | — |\n"
        "| Route 2 (text + stats) | 1.0000 | — | +0.0000 |\n"
        "| Route 3 (text + GIN) | 0.9797 | — | -0.0203 |"
    )

    if old_table_v1 in updated:
        updated = updated.replace(old_table_v1, old_comparison)
    else:
        # Already updated in a prior run — replace the previous values
        import re

        # Match the comparison table rows and replace inline
        table_pattern = re.compile(
            r"(\| route \| val macro-F1 \| test macro-F1 \| Δ vs baseline \(test\) \|\n"
            r"\|[-\| ]+\|\n"
            r"\| Baseline \(text-only\) \| 1\.0000 \| )[\d.]+( \| — \|\n"
            r"\| Route 2 \(text \+ stats\) \| 1\.0000 \| )[\d.]+( \| )[+\-\d.]+( \|\n"
            r"\| Route 3 \(text \+ GIN\) \| 0\.9797 \| )[\d.]+( \| )[+\-\d.]+( \|)"
        )
        match = table_pattern.search(updated)
        if match:
            # Replace with current values
            updated = updated.replace(match.group(0), old_comparison)

    # ── Pass 3: add timestamp note (idempotent — skip if already present) ─
    if "### Test set evaluation" not in updated:
        timestamp_note = (
            f"\n### Test set evaluation — {timestamp}\n\n"
            f"Test set evaluated once on {timestamp}. "
            f"188 transcripts (150 workforce / 19 creatives / 19 scientists).\n"
        )
        updated = updated.replace(
            "---\n\n## Phase 4 — Analysis",
            f"---\n\n{timestamp_note}---\n\n## Phase 4 — Analysis",
        )

    log_path.write_text(updated, encoding="utf-8")
    print(f"Updated {log_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run final test set evaluation for all three routes."""
    timestamp = _get_timestamp()
    print(f"Test set evaluation — {timestamp}")
    print(f"Results will be saved to {RESULTS_DIR}/")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load split ────────────────────────────────────────────────────────
    print("\nLoading test split...")
    _train_ids, _val_ids, test_ids, labels_dict = load_split()
    print(f"  Test set: {len(test_ids)} transcripts")

    # ── Load text embeddings (shared) ─────────────────────────────────────
    print("\nLoading text embeddings...")
    text_embeddings, text_embedding_ids = encode_transcripts()

    # ── Route 1: Text-only baseline ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROUTE 1 — Text-only baseline")
    print("=" * 60)
    r1_results = evaluate_route1(test_ids, labels_dict, text_embeddings, text_embedding_ids)
    print(f"  Test macro-F1: {r1_results['macro_f1']:.4f}")

    # ── Route 2: Text + graph statistics ──────────────────────────────────
    print("\n" + "=" * 60)
    print("ROUTE 2 — Text + graph statistics")
    print("=" * 60)
    r2_results = evaluate_route2(test_ids, labels_dict)
    print(f"  Test macro-F1: {r2_results['macro_f1']:.4f}")

    # ── Route 3: Text + GIN ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROUTE 3 — Text + GIN graph embedding")
    print("=" * 60)
    r3_results = evaluate_route3()
    print(f"  Test macro-F1: {r3_results['macro_f1']:.4f}")

    # ── Print comparison table ────────────────────────────────────────────
    print_comparison_table(r1_results, r2_results, r3_results)

    # ── Save individual route results ─────────────────────────────────────
    for route_tag, results in [
        ("route1", r1_results),
        ("route2", r2_results),
        ("route3", r3_results),
    ]:
        path = RESULTS_DIR / f"{route_tag}_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {path}")

    # ── Save combined comparison ─────────────────────────────────────────
    comparison = build_comparison_dict(r1_results, r2_results, r3_results, timestamp)
    comp_path = RESULTS_DIR / f"comparison_{timestamp}.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved {comp_path}")

    # ── Render confusion matrix grid ─────────────────────────────────────
    cm_path = RESULTS_DIR / f"confusion_matrices_{timestamp}.png"
    plot_confusion_matrices(r1_results, r2_results, r3_results, cm_path)

    # ── Update results log ───────────────────────────────────────────────
    update_results_log(r1_results, r2_results, r3_results, timestamp)

    print("\n" + "=" * 70)
    print("Test set evaluation complete.")
    print(f"Timestamp: {timestamp}")
    print(f"Results directory: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
