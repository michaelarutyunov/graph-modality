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

# v4 corpus (P6.6): all Phase 6 tests run on v4_think only.
CANONICAL_DIR = Path("s1_data/graphs/v4_think/canonical")
FREE_TEXT_DIR = Path("s1_data/graphs/v4_think/free_text")
CACHE_DIR = Path("cache")
EMBEDDING_CACHE = CACHE_DIR / "label_bag_embeddings.npy"
ID_CACHE = CACHE_DIR / "label_bag_embedding_ids.json"
EMBEDDING_CACHE_FREE_TEXT = CACHE_DIR / "label_bag_embeddings_free_text.npy"
ID_CACHE_FREE_TEXT = CACHE_DIR / "label_bag_embedding_ids_free_text.json"

LABEL_ENCODER_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Maps --label-source values to (default graph dir, embedding cache, id cache).
LABEL_SOURCE_PATHS: dict[str, tuple[Path, Path, Path]] = {
    "canonical": (CANONICAL_DIR, EMBEDDING_CACHE, ID_CACHE),
    "free_text": (FREE_TEXT_DIR, EMBEDDING_CACHE_FREE_TEXT, ID_CACHE_FREE_TEXT),
}


def encode_label_bag(
    graph_dir: Path | None = None,
    force: bool = False,
    label_source: str = "canonical",
) -> tuple[np.ndarray, list[str]]:
    """Produce 384-dim mean-pooled label-bag embeddings for all graphs.

    Cache-first: loads from cache unless ``force=True``. Caching is only used for
    the default graph source for ``label_source`` — a custom ``graph_dir`` (e.g. in
    tests) is always recomputed and never written to the shared cache.

    Args:
        graph_dir: Override the default graph directory. If set, caching is
            disabled and results are always recomputed.
        force: Re-encode even if cache exists.
        label_source: ``"canonical"`` (default) or ``"free_text"`` — selects the
            default graph directory and cache paths when ``graph_dir`` is None.

    Returns:
        (embeddings, transcript_ids) — aligned arrays.
    """
    if label_source not in LABEL_SOURCE_PATHS:
        raise ValueError(f"Unknown label_source: {label_source!r}")
    default_dir, embedding_cache, id_cache = LABEL_SOURCE_PATHS[label_source]

    use_cache = graph_dir is None

    if use_cache and embedding_cache.exists() and id_cache.exists() and not force:
        print(f"loading cached label-bag embeddings ({label_source})")
        return np.load(embedding_cache), json.loads(id_cache.read_text(encoding="utf-8"))

    if graph_dir is None:
        graph_dir = default_dir

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

        labels_text = [n.get("label") or "" for n in nodes]  # coerce None (v4 artifact) to ""
        node_embeddings = encoder.encode(
            labels_text, normalize_embeddings=True, show_progress_bar=False
        )
        all_embeddings.append(node_embeddings.mean(axis=0).astype(np.float32))

    result = np.stack(all_embeddings, axis=0)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(embedding_cache, result)
        id_cache.write_text(json.dumps(all_ids, ensure_ascii=False), encoding="utf-8")
        print(
            f"cached {len(all_ids)} label-bag embeddings ({result.shape[1]}d) -> {embedding_cache}"
        )

    return result, all_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label-bag baseline encoder.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if cache exists.",
    )
    parser.add_argument(
        "--label-source",
        choices=["canonical", "free_text"],
        default="canonical",
        help="Use canonical (default) or free-text node labels.",
    )
    args = parser.parse_args()

    embeddings, ids = encode_label_bag(force=args.force, label_source=args.label_source)
    print(f"Done. Shape: {embeddings.shape}, IDs: {len(ids)}")
