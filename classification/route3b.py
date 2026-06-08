"""Route 3b — GIN graph embedding ONLY (no text).

Trains a GIN encoder to produce 128-dim graph embeddings and classifies
professional cohort from graph structure alone — no text features fused.
Answers: can learned graph topology alone discriminate between cohorts?

Usage:
    uv run python classification/route3b.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from classification.split import load_split
from encoding.gnn.train import (
    BATCH_SIZE,
    CANONICAL_DIR,
    COHORT_TO_LABEL,
    GRAPH_EMB_DIM,
    IN_CHANNELS,
    N_CLASSES,
    PREFIX_TO_COHORT,
    build_prefix_to_transcript_id,
    compute_class_weights,
    precompute_graph_data,
)

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
MODEL_PATH = CACHE_DIR / "best_gin_graph_only.pt"
CURVES_PATH = CACHE_DIR / "gnn_graph_only_curves.png"

# ── Training config (same as Route 3 for fair comparison) ────────────────────
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

LABEL_MAP = {0: "workforce", 1: "creatives", 2: "scientists"}
LABEL_NAMES = [LABEL_MAP[i] for i in range(3)]


# ═══════════════════════════════════════════════════════════════════════════════
# Model — same GIN encoder, graph-only classifier head
# ═══════════════════════════════════════════════════════════════════════════════


class GraphOnlyClassifier(nn.Module):
    """GIN encoder → graph-only MLP classifier (no text fusion).

    Uses the same GIN architecture as GraphEncoder but replaces the
    text-fusion classifier head with a graph-only head.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden: int = 256,
        out_channels: int = GRAPH_EMB_DIM,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()

        # GIN layer 1 (same as GraphEncoder)
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.conv1 = GINConv(mlp1)
        self.bn1 = nn.BatchNorm1d(hidden)

        # GIN layer 2 (same as GraphEncoder)
        mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channels),
        )
        self.conv2 = GINConv(mlp2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Graph-only classifier head (128 → 256 → 3, no text)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass: encode graph → classify from graph embedding only."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.nn.functional.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x, edge_index)))
        graph_emb = global_mean_pool(x, batch)  # (B, 128)

        return self.classifier(graph_emb)


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation — same as Route 3 but without text embeddings
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare_graph_data(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, list[int]]:
    """Prepare train/val DataLoaders for graph-only training.

    Returns:
        (train_loader, val_loader, train_labels)
    """
    print("Loading split...")
    train_ids, val_ids, _test_ids, _labels_dict = load_split()

    print("Mapping graph files to transcript IDs...")
    prefix_to_tid = build_prefix_to_transcript_id(CANONICAL_DIR)
    tid_to_graph_path: dict[str, Path] = {}
    for stem, tid in prefix_to_tid.items():
        tid_to_graph_path[tid] = CANONICAL_DIR / f"{stem}.json"

    # Build file/label lists
    def build(paths, ids_list):
        result_paths = []
        result_labels = []
        for tid in ids_list:
            if tid in tid_to_graph_path:
                result_paths.append(tid_to_graph_path[tid])
                prefix = tid.rsplit("_", 1)[0]
                result_labels.append(
                    COHORT_TO_LABEL[PREFIX_TO_COHORT[prefix]]
                )
        return result_paths, result_labels

    train_paths, train_labels = build([], train_ids)
    val_paths, val_labels = build([], val_ids)

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    # Precompute graph data
    train_data = precompute_graph_data(train_paths, train_labels)
    print("Train graphs loaded.")
    val_data = precompute_graph_data(val_paths, val_labels)
    print("Val graphs loaded.")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model: GraphOnlyClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: GraphOnlyClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate on a dataset.

    Returns:
        (average_loss, macro_f1, all_preds, all_labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, macro_f1, all_preds, all_labels


def train() -> dict:
    """Run the full GIN-only training loop.

    Returns:
        Dictionary with training results and validation metrics.
    """
    device = torch.device("cpu")

    # Prepare data
    train_loader, val_loader, train_labels = _prepare_graph_data()

    # Model
    model = GraphOnlyClassifier().to(device)

    # Loss and optimizer
    class_weights = compute_class_weights(train_labels)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
    )

    # Training loop
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1s: list[float] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, _preds, _labels = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_f1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} * best"
            )
        else:
            epochs_no_improve += 1
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
                f"(no improve {epochs_no_improve}/{EARLY_STOPPING_PATIENCE})"
            )

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
            )
            break

    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(train_losses) + 1)
    axes[0].plot(epochs_range, train_losses, label="Train loss", marker="o", markersize=3)
    axes[0].plot(epochs_range, val_losses, label="Val loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss (Graph-Only GIN)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        epochs_range, val_f1s, label="Val macro-F1",
        color="green", marker="o", markersize=3,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
    axes[1].set_title("Validation Macro-F1 (Graph-Only)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(CURVES_PATH, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {CURVES_PATH}")

    # Final evaluation with per-class metrics
    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Epochs run: {len(train_losses)}")

    # Load best model and compute full metrics
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    _, val_f1, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

    per_class = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES)

    print("\n" + "=" * 60)
    print("ROUTE 3b — GIN GRAPH EMBEDDING ONLY (Validation Set)")
    print("=" * 60)
    print(f"\nMacro F1: {val_f1:.4f}  (chance = 0.3333)")
    print("\nPer-class F1:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:>12s}: {per_class[i]:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")
    print("=" * 60)

    return {
        "route": "route3b_gin_graph_only",
        "macro_f1": float(val_f1),
        "per_class_f1": {LABEL_MAP[i]: float(per_class[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "best_epoch": best_epoch,
        "epochs_run": len(train_losses),
        "best_val_f1": float(best_val_f1),
    }


if __name__ == "__main__":
    results = train()
    print(f"\nDone. Validation macro-F1: {results['macro_f1']:.4f}")
