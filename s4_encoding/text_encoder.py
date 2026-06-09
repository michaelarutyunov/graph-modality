"""Encode speaker-tagged transcripts to sentence-transformer embeddings.

Shared across all classification routes (baseline, route 2, route 3).
Caches embeddings to ``cache/text_embeddings.npy`` — never re-encodes if
the cache exists.

Usage:
    uv run python encoding/text_encoder.py                           # full transcript
    uv run python encoding/text_encoder.py --speaker-filter Human    # human-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path("cache")
EMBEDDING_CACHE = CACHE_DIR / "text_embeddings.npy"
ID_CACHE = CACHE_DIR / "text_embedding_ids.json"

DEFAULT_MODEL = "all-mpnet-base-v2"
EXPECTED_DIM = 768


def _cache_paths(speaker_filter: str | None) -> tuple[Path, Path]:
    """Return (embedding_cache, id_cache) keyed by speaker filter."""
    if speaker_filter:
        suffix = f"_{speaker_filter.lower()}_only"
        return (
            CACHE_DIR / f"text_embeddings{suffix}.npy",
            CACHE_DIR / f"text_embedding_ids{suffix}.json",
        )
    return EMBEDDING_CACHE, ID_CACHE


def _build_text(record: dict, speaker_filter: str | None) -> str:
    """Build the text to embed from a tagged transcript record.

    Args:
        record: Tagged transcript dict with ``turns`` and ``formatted`` keys.
        speaker_filter: If set (e.g. ``\"Human\"``), only include turns from
            that speaker.  If ``None``, use the full ``formatted`` field.

    Returns:
        String to pass to the sentence transformer.
    """
    if speaker_filter is None:
        return record["formatted"]

    # Reconstruct text from filtered turns only
    filtered = [
        f"[{t['speaker']}]: {t['text']}"
        for t in record["turns"]
        if t["speaker"] == speaker_filter
    ]
    return "\n\n".join(filtered)


def encode_transcripts(
    model_name: str = DEFAULT_MODEL,
    speaker_filter: str | None = "Human",
) -> tuple[np.ndarray, list[str]]:
    """Encode all tagged transcripts to a (N, 768) embedding matrix.

    Args:
        model_name: Sentence-transformer model identifier.
        speaker_filter: If ``\"Human\"``, embed only human turns
            (removes interviewer confound).  ``None`` embeds the full
            conversation including AI turns.

    Returns:
        (embeddings, transcript_ids) — aligned arrays.
        ``embeddings[i]`` corresponds to ``transcript_ids[i]``.
    """
    emb_cache, id_cache = _cache_paths(speaker_filter)

    if emb_cache.exists() and id_cache.exists():
        print(f"loading cached text embeddings ({speaker_filter or 'full'})")
        embeddings = np.load(emb_cache)
        ids = json.loads(id_cache.read_text(encoding="utf-8"))
        return embeddings, ids

    # Load all tagged transcripts
    tagged_dir = Path("s1_data/tagged")
    records: list[dict] = []
    for path in sorted(tagged_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            records.append(json.loads(line))

    transcript_ids = [r["transcript_id"] for r in records]
    texts = [_build_text(r, speaker_filter) for r in records]

    label = f"{speaker_filter}-only" if speaker_filter else "full transcript"
    print(f"encoding {len(texts)} transcripts ({label}) with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_cache, embeddings)
    id_cache.write_text(
        json.dumps(transcript_ids, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"cached {len(embeddings)} embeddings "
        f"({embeddings.shape[1]}d) → {emb_cache}"
    )

    return embeddings, transcript_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode transcripts to embeddings.")
    parser.add_argument(
        "--speaker-filter",
        type=str,
        default="Human",
        choices=["Human", "all"],
        help="Only embed turns from this speaker (default: Human-only). "
        "Use 'all' for full transcript.",
    )
    args = parser.parse_args()

    filter_arg = None if args.speaker_filter == "all" else args.speaker_filter
    embeddings, ids = encode_transcripts(speaker_filter=filter_arg)
    print(f"Done. Shape: {embeddings.shape}, IDs: {len(ids)}")
