"""Encode speaker-tagged transcripts to sentence-transformer embeddings.

Shared across all classification routes (baseline, route 2, route 3).
Caches embeddings to ``cache/text_embeddings.npy`` — never re-encodes if
the cache exists.

Usage:
    uv run python encoding/text_encoder.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path("cache")
EMBEDDING_CACHE = CACHE_DIR / "text_embeddings.npy"
ID_CACHE = CACHE_DIR / "text_embedding_ids.json"

DEFAULT_MODEL = "all-mpnet-base-v2"
EXPECTED_DIM = 768


def encode_transcripts(
    model_name: str = DEFAULT_MODEL,
) -> tuple[np.ndarray, list[str]]:
    """Encode all tagged transcripts to a (N, 768) embedding matrix.

    Returns:
        (embeddings, transcript_ids) — aligned arrays.
        ``embeddings[i]`` corresponds to ``transcript_ids[i]``.
    """
    if EMBEDDING_CACHE.exists() and ID_CACHE.exists():
        print("loading cached text embeddings")
        embeddings = np.load(EMBEDDING_CACHE)
        ids = json.loads(ID_CACHE.read_text(encoding="utf-8"))
        return embeddings, ids

    # Load all tagged transcripts
    tagged_dir = Path("data/tagged")
    records: list[dict] = []
    for path in sorted(tagged_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            records.append(json.loads(line))

    transcript_ids = [r["transcript_id"] for r in records]
    # Use the raw concatenated transcript text (without speaker tags)
    # for the text encoder — the full transcript preserves semantic content.
    texts = [r["formatted"] for r in records]

    print(f"encoding {len(texts)} transcripts with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_CACHE, embeddings)
    ID_CACHE.write_text(json.dumps(transcript_ids, ensure_ascii=False),
                        encoding="utf-8")
    print(f"cached {len(embeddings)} embeddings ({embeddings.shape[1]}d) → {EMBEDDING_CACHE}")

    return embeddings, transcript_ids


if __name__ == "__main__":
    embeddings, ids = encode_transcripts()
    print(f"Done. Shape: {embeddings.shape}, IDs: {len(ids)}")
