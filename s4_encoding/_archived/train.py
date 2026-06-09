"""GNN training loop for Route 3 (text + GIN graph embedding).

Trains a GIN encoder to produce 128-dim graph embeddings, fuses them with
768-dim text embeddings, and classifies professional cohort (3 classes).

Usage:
    uv run python encoding/gnn/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from s4_encoding.gnn.dataset import GraphDataset
from s4_encoding.gnn.model import GraphEncoder
from torch import nn
from torch_geometric.loader import DataLoader

from s4_encoding.text_encoder import encode_transcripts
from s5_classification.split import load_split

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".")
CANONICAL_DIR = PROJECT_ROOT / "data" / "graphs" / "canonical"
CACHE_DIR = PROJECT_ROOT / "cache"
BEST_MODEL_PATH = CACHE_DIR / "best_gin.pt"
CURVES_PATH = CACHE_DIR / "gnn_curves.png"

# ── Mapping from graph file prefix to cohort name ────────────────────────────
# Graph files are named like creativity_0000.json, science_0000.json, work_0000.json
PREFIX_TO_COHORT = {
    "creativity": "creatives",
    "science": "scientists",
    "work": "workforce",
}

COHORT_TO_LABEL = {"workforce": 0, "creatives": 1, "scientists": 2}

# ── Training configuration ────────────────────────────────────────────────────
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
IN_CHANNELS = 388  # 4 type one-hot + 384 label embedding
GRAPH_EMB_DIM = 128
TEXT_EMB_DIM = 768
N_CLASSES = 3


def build_prefix_to_transcript_id(graph_dir: Path) -> dict[str, str]:
    """Map graph file prefix (stem) to the transcript_id stored in the JSON.

    Scans all graph JSON files and extracts their transcript_id field.
    Returns a dict mapping filename stem -> transcript_id.
    """
    mapping: dict[str, str] = {}
    for graph_file in sorted(graph_dir.glob("*.json")):
        data = json.loads(graph_file.read_text(encoding="utf-8"))
        tid = data.get("transcript_id", "")
        if tid:
            mapping[graph_file.stem] = tid
    return mapping


def build_transcript_id_to_label(
    prefix_to_tid: dict[str, str],
) -> dict[str, int]:
    """Map transcript_id to integer label using the graph file prefix.

    Uses the file naming convention (creativity_*, science_*, work_*)
    to determine the cohort label.
    """
    tid_to_label: dict[str, int] = {}
    for stem, tid in prefix_to_tid.items():
        prefix = stem.rsplit("_", 1)[0]
        cohort = PREFIX_TO_COHORT.get(prefix)
        if cohort is not None:
            tid_to_label[tid] = COHORT_TO_LABEL[cohort]
    return tid_to_label


def load_text_embedding_dict() -> dict[str, torch.Tensor]:
    """Load text embeddings into a dict: transcript_id -> (768,) tensor."""
    embeddings, ids = encode_transcripts()  # (N, 768), list of IDs
    return {
        tid: torch.tensor(emb, dtype=torch.float32)
        for tid, emb in zip(ids, embeddings, strict=True)
    }


def gather_text_embeddings(
    batch_transcript_ids: list[str],
    text_emb_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Gather text embeddings aligned to batch transcript IDs.

    Args:
        batch_transcript_ids: List of transcript IDs in the current batch.
        text_emb_dict: Mapping from transcript_id to embedding tensor.

    Returns:
        Tensor of shape (batch_size, 768).
    """
    return torch.stack([text_emb_dict[tid] for tid in batch_transcript_ids])


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights from training labels.

    Args:
        labels: List of integer labels (0/1/2) from the training set.

    Returns:
        Tensor of shape (n_classes,) with inverse-frequency weights.
    """
    counts = np.bincount(labels, minlength=N_CLASSES)
    # Inverse frequency: weight_i = N / (C * count_i)
    weights = len(labels) / (N_CLASSES * counts.astype(float))
    weights = np.where(counts > 0, weights, 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def precompute_graph_data(
    graph_paths: list[Path],
    labels: list[int],
    label_encoder_name: str = "all-MiniLM-L6-v2",
) -> list:
    """Pre-load all graph Data objects to avoid re-encoding labels on each access.

    This is necessary because GraphDataset.get() encodes node labels with
    sentence-transformers on every call, which is very slow.

    Args:
        graph_paths: Paths to graph JSON files.
        labels: Integer labels for each graph.
        label_encoder_name: Sentence-transformer model for node labels.

    Returns:
        List of PyG Data objects.
    """
    dataset = GraphDataset(graph_paths, labels, label_encoder_name=label_encoder_name)
    print(f"Pre-loading {len(dataset)} graphs (sentence-transformer node label encoding)...")
    data_list = [dataset[i] for i in range(len(dataset))]
    return data_list


def train_one_epoch(
    model: GraphEncoder,
    loader: DataLoader,
    text_emb_dict: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        transcript_ids = batch.transcript_id
        text_embs = gather_text_embeddings(transcript_ids, text_emb_dict).to(device)

        optimizer.zero_grad()
        logits = model(batch, text_embs)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: GraphEncoder,
    loader: DataLoader,
    text_emb_dict: dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on a dataset.

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
    macro_f1 = _compute_macro_f1(all_labels, all_preds)
    return avg_loss, macro_f1


def _compute_macro_f1(labels: list[int], preds: list[int]) -> float:
    """Compute macro-F1 from lists of labels and predictions."""
    from sklearn.metrics import f1_score

    return float(f1_score(labels, preds, average="macro", zero_division=0))


def plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    save_path: Path,
) -> None:
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
    axes[0].plot(epochs, val_losses, label="Val loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training and Validation Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_f1s, label="Val macro-F1", color="green", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
    axes[1].set_title("Validation Macro-F1")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def _prepare_data(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, dict[str, torch.Tensor], list[int]]:
    """Load and prepare train/val data for GNN training.

    Returns:
        (train_loader, val_loader, text_emb_dict, train_labels)
    """
    # ── Load split and labels ─────────────────────────────────────────────
    print("Loading split...")
    train_ids, val_ids, _test_ids, _labels_dict = load_split()

    # ── Map graph files to transcript IDs ──────────────────────────────────
    print("Mapping graph files to transcript IDs...")
    prefix_to_tid = build_prefix_to_transcript_id(CANONICAL_DIR)
    tid_to_graph_path: dict[str, Path] = {}
    for stem, tid in prefix_to_tid.items():
        tid_to_graph_path[tid] = CANONICAL_DIR / f"{stem}.json"

    # ── Build file/label lists for each split ──────────────────────────────
    train_paths = [tid_to_graph_path[tid] for tid in train_ids if tid in tid_to_graph_path]
    train_labels = [
        COHORT_TO_LABEL[PREFIX_TO_COHORT[tid.rsplit("_", 1)[0]]]
        for tid in train_ids
        if tid in tid_to_graph_path
    ]

    val_paths = [tid_to_graph_path[tid] for tid in val_ids if tid in tid_to_graph_path]
    val_labels = [
        COHORT_TO_LABEL[PREFIX_TO_COHORT[tid.rsplit("_", 1)[0]]]
        for tid in val_ids
        if tid in tid_to_graph_path
    ]

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(_test_ids)}")

    # ── Load text embeddings ──────────────────────────────────────────────
    print("Loading text embeddings...")
    text_emb_dict = load_text_embedding_dict()

    # ── Pre-compute graph data (avoids repeated sentence-transformer calls) ─
    train_data = precompute_graph_data(train_paths, train_labels)
    print("Train graphs loaded.")
    val_data = precompute_graph_data(val_paths, val_labels)
    print("Val graphs loaded.")

    # ── Create DataLoaders ─────────────────────────────────────────────────
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, text_emb_dict, train_labels


def run_training_loop(
    train_loader: DataLoader,
    val_loader: DataLoader,
    text_emb_dict: dict[str, torch.Tensor],
    train_labels: list[int],
    max_epochs: int = MAX_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
) -> dict:
    """Execute the GNN training loop with pre-prepared data.

    Args:
        train_loader: DataLoader for training graphs.
        val_loader: DataLoader for validation graphs.
        text_emb_dict: Mapping from transcript_id to text embedding.
        train_labels: Integer labels for the training set.
        max_epochs: Maximum number of training epochs.
        early_stopping_patience: Stop if val macro-F1 doesn't improve.

    Returns:
        Dictionary with training results.
    """
    device = torch.device("cpu")

    # ── Model, optimizer, scheduler, loss ──────────────────────────────────
    model = GraphEncoder(
        in_channels=IN_CHANNELS,
        out_channels=GRAPH_EMB_DIM,
        n_classes=N_CLASSES,
    ).to(device)

    class_weights = compute_class_weights(train_labels)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1s: list[float] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, text_emb_dict, optimizer, criterion, device
        )
        val_loss, val_f1 = evaluate(model, val_loader, text_emb_dict, criterion, device)

        scheduler.step(val_f1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} * best"
            )
        else:
            epochs_no_improve += 1
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
                f"(no improve {epochs_no_improve}/{early_stopping_patience})"
            )

        if epochs_no_improve >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {early_stopping_patience} epochs)"
            )
            break

    # ── Save training curves ───────────────────────────────────────────────
    plot_curves(train_losses, val_losses, val_f1s, CURVES_PATH)

    print("\nTraining complete.")
    print(f"  Best val macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"  Epochs run: {len(train_losses)}")
    print(f"  Best model saved to: {BEST_MODEL_PATH}")

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "epochs_run": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1s": val_f1s,
    }


def train(
    max_epochs: int = MAX_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Run the full GNN training loop.

    Args:
        max_epochs: Maximum number of training epochs.
        early_stopping_patience: Stop if val macro-F1 doesn't improve for this many epochs.
        batch_size: DataLoader batch size.

    Returns:
        Dictionary with training results:
        - best_val_f1: Best validation macro-F1 achieved
        - best_epoch: Epoch at which best F1 was achieved
        - epochs_run: Total epochs actually run
        - train_losses: List of train losses per epoch
        - val_losses: List of val losses per epoch
        - val_f1s: List of val macro-F1 per epoch
    """
    train_loader, val_loader, text_emb_dict, train_labels = _prepare_data(
        batch_size=batch_size,
    )
    return run_training_loop(
        train_loader,
        val_loader,
        text_emb_dict,
        train_labels,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
    )


if __name__ == "__main__":
    results = train()
    print(f"\nFinal best validation macro-F1: {results['best_val_f1']:.4f}")
