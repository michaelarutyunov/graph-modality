"""Label-bag baseline encoder — pooled MiniLM label embeddings, no edges.

Variant (c) of the Method-Review Phase 2 ablation (P2.2): for each canonical
graph, mean-pool the L2-normalized MiniLM embeddings of every node's ``label``
into a single 384-dim vector. No edges, no GNN, no training — this isolates
how much signal comes from node-label semantics alone vs. topology.

Usage:
    uv run python s4_encoding/label_bag_encoder.py [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CANONICAL_DIR = Path("s1_data/graphs/canonical")
CACHE_DIR = Path("cache")
EMBEDDING_CACHE = CACHE_DIR / "label_bag_embeddings.npy"
ID_CACHE = CACHE_DIR / "label_bag_embedding_ids.json"

LABEL_ENCODER_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def encode_label_bag(
    graph_dir: Path | None = None,
    force: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Produce 384-dim mean-pooled label-bag embeddings for all canonical graphs.

    Cache-first: loads from cache unless ``force=True``. Caching is only used for
    the default ``CANONICAL_DIR`` graph source — a custom ``graph_dir`` (e.g. in
    tests) is always recomputed and never written to the shared cache.

    Returns:
        (embeddings, transcript_ids) — aligned arrays.
    """
    use_cache = graph_dir is None

    if use_cache and EMBEDDING_CACHE.exists() and ID_CACHE.exists() and not force:
        print("loading cached label-bag embeddings")
        return np.load(EMBEDDING_CACHE), json.loads(ID_CACHE.read_text(encoding="utf-8"))

    if graph_dir is None:
        graph_dir = CANONICAL_DIR

    graph_paths = sorted(graph_dir.glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError(f"No graph files found in {graph_dir}")

    encoder = SentenceTransformer(LABEL_ENCODER_NAME)

    all_embeddings: list[np.ndarray] = []
    all_ids: list[str] = []

    for path in graph_paths:
        g_data = json.loads(path.read_text(encoding="utf-8"))
        nodes = g_data.get("nodes", [])
        all_ids.append(g_data.get("transcript_id", ""))

        if not nodes:
            all_embeddings.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
            continue

        labels_text = [n.get("label", "") for n in nodes]
        node_embeddings = encoder.encode(
            labels_text, normalize_embeddings=True, show_progress_bar=False
        )
        all_embeddings.append(node_embeddings.mean(axis=0).astype(np.float32))

    result = np.stack(all_embeddings, axis=0)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDING_CACHE, result)
        ID_CACHE.write_text(json.dumps(all_ids, ensure_ascii=False), encoding="utf-8")
        print(
            f"cached {len(all_ids)} label-bag embeddings ({result.shape[1]}d) -> {EMBEDDING_CACHE}"
        )

    return result, all_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label-bag baseline encoder.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if cache exists.",
    )
    args = parser.parse_args()

    embeddings, ids = encode_label_bag(force=args.force)
    print(f"Done. Shape: {embeddings.shape}, IDs: {len(ids)}")
