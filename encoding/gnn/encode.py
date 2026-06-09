"""Frozen GIN encoder inference — produces 128-dim graph embeddings.

Loads the trained GINEncoder from cache/gin_encoder.pt, runs inference on
all canonical graphs, and caches the resulting embeddings to disk. Follows
the same cache-first pattern as text_encoder.py and graph_stats.py.

The encoder is frozen — no gradients, no fine-tuning. These embeddings are
the target-agnostic graph modality representation consumed by downstream
classifiers.

Usage:
    uv run python encoding/gnn/encode.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from encoding.gnn.autoencoder import GINEncoder
from encoding.gnn.dataset import GraphDataset

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".")
CANONICAL_DIR = PROJECT_ROOT / "data" / "graphs" / "canonical"
CACHE_DIR = PROJECT_ROOT / "cache"
ENCODER_PATH = CACHE_DIR / "gin_encoder.pt"
EMBEDDING_CACHE = CACHE_DIR / "gin_embeddings.npy"
ID_CACHE = CACHE_DIR / "gin_embedding_ids.json"

BATCH_SIZE = 32


def encode_graphs(
    graph_dir: Path | None = None,
    encoder_weights: Path | None = None,
    force: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Produce 128-dim frozen graph embeddings for all canonical graphs.

    Cache-first: if cache/gin_embeddings.npy and cache/gin_embedding_ids.json
    exist, load from cache unless ``force=True``.

    Args:
        graph_dir: Directory containing canonical graph JSON files.
        encoder_weights: Path to trained GINEncoder state dict.
        force: If True, re-encode even if cache exists.

    Returns:
        (embeddings, transcript_ids) — aligned arrays.
        ``embeddings[i]`` corresponds to ``transcript_ids[i]``.
    """
    if graph_dir is None:
        graph_dir = CANONICAL_DIR
    if encoder_weights is None:
        encoder_weights = ENCODER_PATH

    # ── Check cache ────────────────────────────────────────────────────────
    if EMBEDDING_CACHE.exists() and ID_CACHE.exists() and not force:
        print("loading cached GIN embeddings")
        embeddings = np.load(EMBEDDING_CACHE)
        ids = json.loads(ID_CACHE.read_text(encoding="utf-8"))
        return embeddings, ids

    # ── Verify encoder exists ──────────────────────────────────────────────
    if not encoder_weights.exists():
        raise FileNotFoundError(
            f"Encoder weights not found at {encoder_weights}. "
            "Run 'uv run python encoding/gnn/autoencoder.py' first to train."
        )

    # ── Load encoder ───────────────────────────────────────────────────────
    device = torch.device("cpu")
    encoder = GINEncoder().to(device)
    encoder.load_state_dict(
        torch.load(encoder_weights, map_location=device, weights_only=True)
    )
    encoder.eval()
    print(f"Loaded frozen GIN encoder from {encoder_weights}")

    # ── Load graph paths ───────────────────────────────────────────────────
    graph_paths = sorted(graph_dir.glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError(f"No graph files found in {graph_dir}")
    print(f"Found {len(graph_paths)} graphs in {graph_dir}")

    # ── Pre-load all Data objects ──────────────────────────────────────────
    # Use dummy labels since we don't need classification labels
    dummy_labels = [-1] * len(graph_paths)
    dataset = GraphDataset(graph_paths, dummy_labels)
    print(f"Pre-loading {len(dataset)} graphs...")
    data_list = [dataset[i] for i in range(len(dataset))]

    # ── Inference ──────────────────────────────────────────────────────────
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

    result = np.concatenate(all_embeddings, axis=0)  # (N, 128)

    # ── Cache ──────────────────────────────────────────────────────────────
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_CACHE, result)
    ID_CACHE.write_text(
        json.dumps(all_ids, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"cached {len(all_ids)} GIN embeddings "
        f"({result.shape[1]}d) → {EMBEDDING_CACHE}"
    )

    return result, all_ids


if __name__ == "__main__":
    embeddings, ids = encode_graphs()
    print(f"Done. Shape: {embeddings.shape}, IDs: {len(ids)}")
    print(f"Sample (first graph, first 5 dims): {embeddings[0, :5]}")
