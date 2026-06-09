"""Graph GNN encoder — self-supervised GIN for target-agnostic graph embeddings.

Two-stage pipeline:
  1. Train: Self-supervised autoencoder (node type reconstruction) on ALL graphs.
  2. Encode: Frozen encoder inference → 128-dim embeddings (cache-first, like text_encoder).

Architecture: 2-layer GIN encoder (388→256→128) → global mean pool → graph embedding.
A node type decoder head (Linear(128, 4)) reconstructs entity types from per-node
embeddings before pooling. After training, only the encoder is saved.

Why node-type-only (not edges): Edge prediction on 15-node graphs has 87%
negative class, allowing trivial accuracy without learning structure.
Node types require neighbourhood information — a node's type depends on
what it connects to.

Usage:
    uv run python encoding/graph_gnn_encoder.py          # train autoencoder
    uv run python encoding/graph_gnn_encoder.py --encode # frozen inference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from s4_encoding.graph_dataset import GraphDataset

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".")
CANONICAL_DIR = PROJECT_ROOT / "s1_data" / "graphs" / "canonical"
FREE_TEXT_DIR = PROJECT_ROOT / "s1_data" / "graphs" / "free_text"
CACHE_DIR = PROJECT_ROOT / "cache"
ENCODER_PATH = CACHE_DIR / "gin_encoder_canonical.pt"
CURVES_PATH = CACHE_DIR / "gin_autoencoder_curves_canonical.png"
EMBEDDING_CACHE = CACHE_DIR / "gin_embeddings_canonical.npy"
ID_CACHE = CACHE_DIR / "gin_embedding_ids_canonical.json"
# Free-text variants — separate encoder, curves, embeddings
FT_ENCODER_PATH = CACHE_DIR / "gin_encoder_free_text.pt"
FT_CURVES_PATH = CACHE_DIR / "gin_autoencoder_curves_free_text.png"
FT_EMBEDDING_CACHE = CACHE_DIR / "gin_embeddings_free_text.npy"
FT_ID_CACHE = CACHE_DIR / "gin_embedding_ids_free_text.json"


def _encoder_path(label_source: str) -> Path:
    """Return encoder weights path for a given label source."""
    return FT_ENCODER_PATH if label_source == "free_text" else ENCODER_PATH


def _curves_path(label_source: str) -> Path:
    """Return training curves path for a given label source."""
    return FT_CURVES_PATH if label_source == "free_text" else CURVES_PATH


def _graph_dir(label_source: str) -> Path:
    """Return graph directory for a given label source."""
    return FREE_TEXT_DIR if label_source == "free_text" else CANONICAL_DIR

# ── Architecture constants ────────────────────────────────────────────────────
IN_CHANNELS = 388  # 4 type one-hot + 384 label embedding
HIDDEN_DIM = 256
OUT_CHANNELS = 128
N_NODE_TYPES = 4  # Construct, Value, Stance, CognitiveStyleMarker

# ── Training configuration ────────────────────────────────────────────────────
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Model architecture
# ═══════════════════════════════════════════════════════════════════════════════


class GINEncoder(nn.Module):
    """Target-agnostic GIN encoder producing 128-dim graph embeddings.

    Two GINConv layers with batch norm, followed by global mean pool.
    Trained with self-supervised reconstruction — NEVER sees classification labels.
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
        """Forward pass returning both per-node and graph-level embeddings."""
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

    After training, ``self.encoder`` is extracted, frozen, and saved.
    """

    def __init__(self):
        super().__init__()
        self.encoder = GINEncoder()
        self.node_type_head = nn.Linear(OUT_CHANNELS, N_NODE_TYPES)

    def forward(self, data) -> torch.Tensor:
        """Produce node type logits for all nodes in the batch."""
        _graph_emb, node_emb = self.encoder(data.x, data.edge_index, data.batch)
        return self.node_type_head(node_emb)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def _get_node_type_labels(data) -> torch.Tensor:
    """Extract ground-truth node type indices from node features.

    The first 4 dimensions of ``data.x`` are the type one-hot encoding.
    """
    return data.x[:, :N_NODE_TYPES].argmax(dim=1).long()


def _node_type_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute node type prediction accuracy."""
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


def _precompute_graph_data(graph_paths: list[Path]) -> list:
    """Pre-load all graph Data objects to avoid re-encoding labels on each access."""
    dummy_labels = [-1] * len(graph_paths)
    dataset = GraphDataset(graph_paths, dummy_labels)
    print(f"Pre-loading {len(dataset)} graphs (sentence-transformer node label encoding)...")
    return [dataset[i] for i in range(len(dataset))]


def train_autoencoder(
    graph_dir: Path | None = None,
    max_epochs: int = MAX_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    label_source: str = "canonical",
) -> dict:
    """Train the GIN autoencoder on all graphs.

    Args:
        graph_dir: Directory containing graph JSON files. Defaults based on
            ``label_source``.
        label_source: ``"canonical"`` (default) or ``"free_text"``.
            Determines graph directory, encoder save path, and curves path.

    Returns:
        Dictionary with best_loss, best_epoch, final_accuracy, epochs_run,
        train_losses, train_accs.
    """
    device = torch.device("cpu")
    print(f"Training on device: {device}")
    print(f"Label source: {label_source}")

    if graph_dir is None:
        graph_dir = _graph_dir(label_source)

    graph_paths = _load_all_graph_paths(graph_dir)
    print(f"Loading {len(graph_paths)} graphs for self-supervised training...")
    data_list = _precompute_graph_data(graph_paths)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with {len(loader)} batches (batch_size={batch_size})")

    model = GINAutoencoder().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR,
    )

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
            torch.save(model.encoder.state_dict(), _encoder_path(label_source))
            mark = "* best"
        else:
            epochs_no_improve += 1
            mark = f"(no improve {epochs_no_improve}/{early_stopping_patience})"

        print(f"  Epoch {epoch}: loss={avg_loss:.4f} acc={avg_acc:.4f} {mark}")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    _plot_training_curves(train_losses, train_accs, _curves_path(label_source))

    print(f"\nAutoencoder training complete.")
    print(f"  Label source: {label_source}")
    print(f"  Best loss: {best_loss:.4f} at epoch {best_epoch}")
    print(f"  Final node type accuracy: {train_accs[-1]:.4f}")
    print(f"  Epochs run: {len(train_losses)}")
    print(f"  Encoder saved to: {_encoder_path(label_source)}")

    return {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "final_accuracy": train_accs[-1],
        "epochs_run": len(train_losses),
        "train_losses": train_losses,
        "train_accs": train_accs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Frozen encoder inference
# ═══════════════════════════════════════════════════════════════════════════════


def _cache_paths(label_source: str) -> tuple[Path, Path]:
    """Return (embedding_cache, id_cache) keyed by label source."""
    if label_source == "free_text":
        return FT_EMBEDDING_CACHE, FT_ID_CACHE
    return EMBEDDING_CACHE, ID_CACHE


def encode_graphs(
    graph_dir: Path | None = None,
    encoder_weights: Path | None = None,
    force: bool = False,
    label_source: str = "canonical",
) -> tuple[np.ndarray, list[str]]:
    """Produce 128-dim frozen graph embeddings.

    Cache-first: loads from cache unless ``force=True``. Cache paths are
    keyed by ``label_source`` so canonical and free-text embeddings coexist.

    Args:
        graph_dir: Directory containing graph JSON files. Defaults to
            ``s1_data/graphs/canonical`` or ``s1_data/graphs/free_text``
            based on ``label_source``.
        encoder_weights: Path to trained GINEncoder state dict.
        force: If True, re-encode even if cache exists.
        label_source: ``"canonical"`` (default) or ``"free_text"``.
            Determines both the default graph directory and the cache path.

    Returns:
        (embeddings, transcript_ids) — aligned arrays.
    """
    if graph_dir is None:
        graph_dir = FREE_TEXT_DIR if label_source == "free_text" else CANONICAL_DIR
    if encoder_weights is None:
        encoder_weights = _encoder_path(label_source)

    emb_cache, id_cache = _cache_paths(label_source)

    if emb_cache.exists() and id_cache.exists() and not force:
        print(f"loading cached GIN embeddings ({label_source} labels)")
        return np.load(emb_cache), json.loads(id_cache.read_text(encoding="utf-8"))

    if not encoder_weights.exists():
        raise FileNotFoundError(
            f"Encoder weights not found at {encoder_weights}. "
            "Run 'uv run python encoding/graph_gnn_encoder.py' first to train."
        )

    device = torch.device("cpu")
    encoder = GINEncoder().to(device)
    encoder.load_state_dict(torch.load(encoder_weights, map_location=device, weights_only=True))
    encoder.eval()
    print(f"Loaded frozen GIN encoder from {encoder_weights}")

    graph_paths = sorted(graph_dir.glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError(f"No graph files found in {graph_dir}")

    dummy_labels = [-1] * len(graph_paths)
    dataset = GraphDataset(graph_paths, dummy_labels)
    print(f"Pre-loading {len(dataset)} graphs...")
    data_list = [dataset[i] for i in range(len(dataset))]

    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    all_embeddings: list[np.ndarray] = []
    all_ids: list[str] = []

    print(f"Encoding {len(data_list)} graphs in batches of {BATCH_SIZE}...")
    with torch.no_grad():
        for batch in loader:
            graph_emb, _node_emb = encoder(
                batch.x.to(device),
                batch.edge_index.to(device),
                batch.batch.to(device),
            )
            all_embeddings.append(graph_emb.cpu().numpy())
            all_ids.extend(batch.transcript_id)

    result = np.concatenate(all_embeddings, axis=0)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_cache, result)
    id_cache.write_text(json.dumps(all_ids, ensure_ascii=False), encoding="utf-8")
    print(f"cached {len(all_ids)} GIN embeddings ({result.shape[1]}d, {label_source} labels) → {emb_cache}")

    return result, all_ids


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIN graph encoder: train or encode.")
    parser.add_argument(
        "--encode", action="store_true",
        help="Run frozen encoder inference (instead of training).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-encode even if cache exists (only with --encode).",
    )
    parser.add_argument(
        "--label-source", type=str, choices=["canonical", "free_text"],
        default="canonical",
        help="Which graph labels to use (default: canonical). "
             "Training: trains a separate encoder. "
             "Encoding: uses the matching encoder weights.",
    )
    args = parser.parse_args()

    if args.encode:
        embeddings, ids = encode_graphs(
            force=args.force, label_source=args.label_source,
        )
        print(f"Done. Shape: {embeddings.shape}, IDs: {len(ids)}")
    else:
        results = train_autoencoder(label_source=args.label_source)
        print(f"\nFinal node type reconstruction accuracy: {results['final_accuracy']:.4f}")
