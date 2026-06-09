"""Tests for encoding.text_encoder."""

from unittest.mock import patch

import numpy as np

from s4_encoding.text_encoder import encode_transcripts

# ── Cache-based tests (work with real cached output) ───────────────────────


def test_embedding_shape():
    """Matrix is (N, 768) where N = number of transcripts."""
    embeddings, ids = encode_transcripts()
    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
    assert embeddings.shape[1] == 768, f"Expected 768 dimensions, got {embeddings.shape[1]}"
    assert embeddings.shape[0] == len(ids), (
        f"Shape mismatch: {embeddings.shape[0]} embeddings vs {len(ids)} IDs"
    )


def test_id_list_length():
    """ID list length matches embedding count."""
    embeddings, ids = encode_transcripts()
    assert len(ids) == embeddings.shape[0], (
        f"ID count {len(ids)} != embedding count {embeddings.shape[0]}"
    )


def test_embedding_dtype():
    """Embeddings are float32."""
    embeddings, _ = encode_transcripts()
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"


def test_cache_files_exist():
    """After encoding, both cache files exist."""
    from pathlib import Path

    embeddings, _ = encode_transcripts()
    assert Path("cache/text_embeddings.npy").exists(), "Embedding cache file missing"
    assert Path("cache/text_embedding_ids.json").exists(), "ID cache file missing"
    assert embeddings.shape[0] > 0, "No embeddings were cached"


# ── Idempotency test ───────────────────────────────────────────────────────


def test_idempotent(tmp_path):
    """Second call to encode_transcripts() loads from cache (SentenceTransformer NOT called)."""
    # Mock the cache paths and data directory to use temp directory
    with (
        patch("encoding.text_encoder.EMBEDDING_CACHE", tmp_path / "embeddings.npy"),
        patch("encoding.text_encoder.ID_CACHE", tmp_path / "ids.json"),
        patch("encoding.text_encoder.CACHE_DIR", tmp_path),
    ):
        # Create dummy tagged data
        tagged_dir = tmp_path / "tagged"
        tagged_dir.mkdir(parents=True, exist_ok=True)
        (tagged_dir / "test.jsonl").write_text(
            '{"transcript_id": "t1", "formatted": "Test transcript."}\n'
        )

        # Mock SentenceTransformer at the module level to track construction
        with patch("encoding.text_encoder.SentenceTransformer") as mock_st:
            # Configure mock to return a functional encoder
            mock_encoder = mock_st.return_value
            mock_encoder.encode.return_value = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

            # First call - should construct SentenceTransformer
            embeddings1, ids1 = encode_transcripts()
            assert mock_st.call_count == 1, (
                "SentenceTransformer should be constructed on first call"
            )
            assert mock_st.call_args[0] == ("all-mpnet-base-v2",), "Should use default model"

            # Second call - should load from cache
            embeddings2, ids2 = encode_transcripts()
            assert mock_st.call_count == 1, (
                "SentenceTransformer should NOT be constructed on second call (should load cache)"
            )

            # Verify results are identical
            assert np.array_equal(embeddings1, embeddings2), "Cached embeddings should match"
            assert ids1 == ids2, "Cached IDs should match"
