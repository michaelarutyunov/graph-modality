"""GIN autoencoder — self-supervised, target-agnostic graph encoder.

Architecture: 2-layer GIN encoder (388→256→128) → global mean pool →
graph embedding. A node type decoder head (Linear(128, 4)) reconstructs
entity types from per-node embeddings before pooling.

Loss: Cross-entropy on 4-class node type prediction. Trained on ALL 1,250
graphs — no classification labels, no train/val/test split.

Why node-type-only (not edges): Edge prediction on 15-node graphs has 87%
negative class, allowing trivial accuracy without learning structure.
Node types require neighbourhood information — a node's type depends on
what it connects to.

Usage:
    uv run python encoding/gnn/autoencoder.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from encoding.gnn.dataset import GraphDataset

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".")
CANONICAL_DIR = PROJECT_ROOT / "data" / "graphs" / "canonical"
CACHE_DIR = PROJECT_ROOT / "cache"
ENCODER_PATH = CACHE_DIR / "gin_encoder.pt"
CURVES_PATH = CACHE_DIR / "gin_autoencoder_curves.png"

# ── Training configuration ────────────────────────────────────────────────────
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5
IN_CHANNELS = 388  # 4 type one-hot + 384 label embedding
HIDDEN_DIM = 256
OUT_CHANNELS = 128
N_NODE_TYPES = 4  # Construct, Value, Stance, CognitiveStyleMarker


class GINEncoder(nn.Module):
    """Target-agnostic GIN encoder producing 128-dim graph embeddings.

    Two GINConv layers with batch norm, followed by global mean pool.
    Trained with self-supervised reconstruction — NEVER sees classification labels.

    Args:
        in_channels: Dimension of input node features (default 388).
        hidden: Hidden dimension for GIN MLPs.
        out_channels: Output graph embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden: int = HIDDEN_DIM,
        out_channels: int = OUT_CHANNELS,
    ):
        super().__init__()

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.conv1 = GINConv(mlp1)
        self.bn1 = nn.BatchNorm1d(hidden)

        mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channels),
        )
        self.conv2 = GINConv(mlp2)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both per-node and graph-level embeddings.

        Args:
            x: Node features of shape ``(total_nodes, in_channels)``.
            edge_index: Adjacency list of shape ``(2, total_edges)``.
            batch: Batch assignment of shape ``(total_nodes,)``.

        Returns:
            (graph_emb, node_emb) — graph-level ``(batch_size, 128)`` and
            per-node ``(total_nodes, 128)`` embeddings.
        """
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        node_emb = F.relu(self.bn2(self.conv2(h, edge_index)))
        graph_emb = global_mean_pool(node_emb, batch)
        return graph_emb, node_emb


class GINAutoencoder(nn.Module):
    """Self-supervised GIN: encode graph structure, reconstruct node types.

    The encoder produces 128-dim per-node embeddings. A linear decoder head
    predicts the 4-class entity type from each node embedding. The encoder
    is forced to learn structural patterns because a node's type depends on
    its neighbourhood context.

    After training, `self.encoder` is extracted, frozen, and saved as the
    target-agnostic graph modality encoder.
    """

    def __init__(self):
        super().__init__()
        self.encoder = GINEncoder()
        self.node_type_head = nn.Linear(OUT_CHANNELS, N_NODE_TYPES)

    def forward(self, data) -> torch.Tensor:
        """Produce node type logits for all nodes in the batch.

        Args:
            data: A batch of ``torch_geometric.data.Data`` objects.

        Returns:
            Logits of shape ``(total_nodes, 4)`` — one per node.
        """
        _graph_emb, node_emb = self.encoder(data.x, data.edge_index, data.batch)
        return self.node_type_head(node_emb)


def _get_node_type_labels(data) -> torch.Tensor:
    """Extract ground-truth node type indices from node features.

    The first 4 dimensions of ``data.x`` are the type one-hot encoding.
    Returns integer labels of shape ``(total_nodes,)``.
    """
    return data.x[:, :N_NODE_TYPES].argmax(dim=1).long()


def _node_type_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute node type prediction accuracy.

    Args:
        logits: Predicted logits of shape ``(N_nodes, 4)``.
        labels: Ground truth node type indices of shape ``(N_nodes,)``.

    Returns:
        Accuracy as a float in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _plot_training_curves(
    train_losses: list[float],
    train_accs: list[float],
    save_path: Path,
) -> None:
    """Plot and save training curves (loss + accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training Loss (Node Type Reconstruction)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="Train accuracy", color="green", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Node Type Prediction Accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def _load_all_graph_paths(graph_dir: Path | None = None) -> list[Path]:
    """Return sorted list of all canonical graph JSON paths."""
    if graph_dir is None:
        graph_dir = CANONICAL_DIR
    paths = sorted(graph_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No graph files found in {graph_dir}")
    return paths


def _precompute_graph_data(
    graph_paths: list[Path],
) -> list:
    """Pre-load all graph Data objects to avoid re-encoding labels on each access.

    Uses the existing GraphDataset which encodes node labels with
    sentence-transformers. Pre-loading avoids repeated encoding during training.
    """
    # Use dummy labels (all -1) since we don't need classification labels
    dummy_labels = [-1] * len(graph_paths)
    dataset = GraphDataset(graph_paths, dummy_labels)
    print(f"Pre-loading {len(dataset)} graphs (sentence-transformer node label encoding)...")
    data_list = [dataset[i] for i in range(len(dataset))]
    return data_list


def train_autoencoder(
    graph_dir: Path | None = None,
    max_epochs: int = MAX_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train the GIN autoencoder on all graphs.

    Args:
        graph_dir: Directory containing canonical graph JSON files.
        max_epochs: Maximum number of training epochs.
        early_stopping_patience: Stop if loss doesn't improve.
        batch_size: DataLoader batch size.

    Returns:
        Dictionary with training results.
    """
    device = torch.device("cpu")
    print(f"Training on device: {device}")

    # ── Load all graphs ────────────────────────────────────────────────────
    graph_paths = _load_all_graph_paths(graph_dir)
    print(f"Loading {len(graph_paths)} graphs for self-supervised training...")
    data_list = _precompute_graph_data(graph_paths)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with {len(loader)} batches (batch_size={batch_size})")

    # ── Model, optimizer, scheduler, loss ──────────────────────────────────
    model = GINAutoencoder().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    train_losses: list[float] = []
    train_accs: list[float] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_nodes = 0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            node_labels = _get_node_type_labels(batch)

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, node_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += _node_type_accuracy(logits, node_labels) * batch.num_nodes
            total_nodes += int(batch.num_nodes)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(total_nodes, 1)

        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        scheduler.step(avg_loss)

        improved = avg_loss < best_loss
        if improved:
            best_loss = avg_loss
            best_epoch = epoch
            epochs_no_improve = 0
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Save only the encoder (not the decoder head)
            torch.save(model.encoder.state_dict(), ENCODER_PATH)
            print(
                f"  Epoch {epoch}: loss={avg_loss:.4f} acc={avg_acc:.4f} * best"
            )
        else:
            epochs_no_improve += 1
            print(
                f"  Epoch {epoch}: loss={avg_loss:.4f} acc={avg_acc:.4f} "
                f"(no improve {epochs_no_improve}/{early_stopping_patience})"
            )

        if epochs_no_improve >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {early_stopping_patience} epochs)"
            )
            break

    # ── Save training curves ───────────────────────────────────────────────
    _plot_training_curves(train_losses, train_accs, CURVES_PATH)

    print("\nAutoencoder training complete.")
    print(f"  Best loss: {best_loss:.4f} at epoch {best_epoch}")
    print(f"  Final node type accuracy: {train_accs[-1]:.4f}")
    print(f"  Epochs run: {len(train_losses)}")
    print(f"  Encoder saved to: {ENCODER_PATH}")

    return {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "final_accuracy": train_accs[-1],
        "epochs_run": len(train_losses),
        "train_losses": train_losses,
        "train_accs": train_accs,
    }


if __name__ == "__main__":
    results = train_autoencoder()
    print(f"\nFinal node type reconstruction accuracy: {results['final_accuracy']:.4f}")
