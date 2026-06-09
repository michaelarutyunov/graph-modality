"""Modality embedding dataset builder.

Packages all three frozen modality embeddings (text, graph stats, GIN)
into .npz files per target and split. This is the single source of truth
for downstream classifiers.

Adding a new modality requires:
  1. Produce cached .npy + _ids.json (e.g., contrastive GIN)
  2. Add one line in the _load_*_embeddings() section below
  3. The new key is automatically available to classifiers that reference
     it in their modality_dims config.

Usage:
    uv run python encoding/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from s5_classification.split import load_split
from s4_encoding.graph_stats_encoder import compute_all_stats
from s4_encoding.graph_gnn_encoder import encode_graphs
from s4_encoding.text_encoder import encode_transcripts

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".")
CACHE_DIR = PROJECT_ROOT / "cache"
DATASET_DIR = CACHE_DIR / "modality_dataset"
DEMOGRAPHICS_PATH = CACHE_DIR / "demographics.jsonl"

# ── AI adoption label mapping ────────────────────────────────────────────────
# Binary classification: tool_user=0, integrated=1. novice/power_user excluded.
AI_LABEL_MAP = {"tool_user": 0, "integrated": 1}
AI_EXCLUDED = {"novice", "power_user"}


def _load_modality_arrays(
    label_source: str = "canonical",
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Load all three frozen modality embeddings and their ID lists.

    Args:
        label_source: ``"canonical"`` or ``"free_text"`` — which graph
            labels to use for GNN embeddings.

    Returns:
        Dict mapping modality name → (embeddings, transcript_ids).
    """
    print("Loading text embeddings...")
    text_embs, text_ids = encode_transcripts(speaker_filter="Human")
    print(f"  text: {text_embs.shape}, {len(text_ids)} IDs")

    print("Loading graph statistics...")
    stats_embs, stats_ids = compute_all_stats()
    print(f"  stats: {stats_embs.shape}, {len(stats_ids)} IDs")

    print("Loading GIN embeddings...")
    gin_embs, gin_ids = encode_graphs(label_source=label_source)
    print(f"  GIN ({label_source}): {gin_embs.shape}, {len(gin_ids)} IDs")

    return {
        "text": (text_embs, text_ids),
        "stats": (stats_embs, stats_ids),
        "graph": (gin_embs, gin_ids),
    }


def _load_ai_adoption_labels() -> dict[str, int]:
    """Load AI adoption labels, mapping tool_user→0, integrated→1.

    Excludes transcripts with novice or power_user labels (n=26).
    Returns dict of transcript_id → binary label.
    """
    if not DEMOGRAPHICS_PATH.exists():
        raise FileNotFoundError(
            f"Demographics file not found at {DEMOGRAPHICS_PATH}. "
            "Run the demographics extraction first."
        )

    labels: dict[str, int] = {}
    excluded = 0
    with open(DEMOGRAPHICS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            tid = record["transcript_id"]
            raw_label = record["ai_adoption"]["label"]
            if raw_label in AI_EXCLUDED:
                excluded += 1
                continue
            labels[tid] = AI_LABEL_MAP[raw_label]

    print(f"AI adoption labels: {len(labels)} transcripts (excluded {excluded})")
    return labels


def _build_id_index(id_list: list[str]) -> dict[str, int]:
    """Build transcript_id → array index mapping."""
    return {tid: i for i, tid in enumerate(id_list)}


def _build_split_masks(
    split_ids: dict[str, list[str]],
    valid_ids: set[str],
) -> dict[str, list[str]]:
    """Filter split IDs to those present in the target.

    Args:
        split_ids: The full split dict with 'train', 'val', 'test' keys.
        valid_ids: Set of transcript IDs valid for this target.

    Returns:
        Filtered split dict.
    """
    return {
        split_name: [tid for tid in ids if tid in valid_ids]
        for split_name, ids in split_ids.items()
    }


def build_dataset(label_source: str = "canonical") -> dict[str, Any]:
    """Build and save all modality .npz files.

    Args:
        label_source: ``"canonical"`` (default) or ``"free_text"``.
            Determines which graph labels the GNN uses. Graph stats are unaffected.

    Returns:
        Metadata dictionary with shapes, label distributions, and source info.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all modalities ─────────────────────────────────────────────────
    modalities = _load_modality_arrays(label_source=label_source)

    # Build ID→index lookups for each modality
    idx_lookups = {
        name: _build_id_index(id_list)
        for name, (_embs, id_list) in modalities.items()
    }

    # ── Load cohort labels ──────────────────────────────────────────────────
    print("\nLoading cohort split and labels...")
    train_ids, val_ids, test_ids, labels_dict = load_split()
    cohort_splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    print(f"  Cohort: {len(labels_dict)} labeled transcripts")

    # ── Load AI adoption labels ─────────────────────────────────────────────
    ai_labels = _load_ai_adoption_labels()
    ai_valid_ids = set(ai_labels.keys())
    ai_splits = _build_split_masks(cohort_splits, ai_valid_ids)

    for split_name, ids in ai_splits.items():
        print(f"  AI adoption {split_name}: {len(ids)} transcripts")

    # ── Build .npz files ────────────────────────────────────────────────────

    def _package_target(
        target: str,
        splits: dict[str, list[str]],
        labels: dict[str, int],
    ) -> dict[str, Any]:
        """Package one target's worth of .npz files."""
        shapes: dict[str, tuple] = {}
        label_counts: dict[str, dict[str, int]] = {}

        for split_name, split_id_list in splits.items():
            # Gather indices for each modality
            arrays: dict[str, np.ndarray] = {}
            for mod_name, (embs, _id_list) in modalities.items():
                lookup = idx_lookups[mod_name]
                indices = [lookup[tid] for tid in split_id_list]
                arrays[mod_name] = embs[indices]

            # Gather labels
            label_array = np.array(
                [labels[tid] for tid in split_id_list], dtype=np.int64
            )

            # Compute label distribution
            unique, counts = np.unique(label_array, return_counts=True)
            label_dist = {str(int(u)): int(c) for u, c in zip(unique, counts)}

            # Save
            out_path = DATASET_DIR / f"{target}_{split_name}.npz"
            np.savez(
                out_path,
                text_emb=arrays["text"],
                stats_emb=arrays["stats"],
                graph_emb=arrays["graph"],
                labels=label_array,
                transcript_ids=np.array(split_id_list),
            )
            shapes[split_name] = {
                "text_emb": arrays["text"].shape,
                "stats_emb": arrays["stats"].shape,
                "graph_emb": arrays["graph"].shape,
                "labels": label_array.shape,
            }
            label_counts[split_name] = label_dist
            print(f"  Saved {out_path}")

        return {"shapes": shapes, "label_distributions": label_counts}

    # ── Cohort target ───────────────────────────────────────────────────────
    print("\nPackaging cohort target...")
    cohort_info = _package_target("cohort", cohort_splits, labels_dict)

    # ── AI adoption target ──────────────────────────────────────────────────
    print("\nPackaging AI adoption target...")
    ai_info = _package_target("ai_adoption", ai_splits, ai_labels)

    # ── Metadata ────────────────────────────────────────────────────────────
    metadata = {
        "created_at": "2026-06-09",
        "source_encoders": {
            "text": "all-mpnet-base-v2 (frozen SBERT, human-only turns)",
            "stats": "networkx-derived graph statistics (30-dim, deterministic)",
            "graph": f"GIN autoencoder (self-supervised, node type reconstruction, 128-dim, {label_source} labels)",
        },
        "graph_label_source": label_source,
        "targets": {
            "cohort": cohort_info,
            "ai_adoption": ai_info,
        },
    }

    readme_path = DATASET_DIR / "README.json"
    readme_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetadata saved to {readme_path}")

    return metadata


def load_dataset(target: str, split: str) -> dict[str, np.ndarray]:
    """Load one .npz file for a given target and split.

    Args:
        target: ``"cohort"`` or ``"ai_adoption"``.
        split: ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        Dictionary with keys: ``text_emb``, ``stats_emb``, ``graph_emb``,
        ``labels``, ``transcript_ids``. Classifiers access modalities by name.
        Adding a new modality: the new key is automatically available to
        classifiers that reference it in their ``modality_dims`` config.
    """
    path = DATASET_DIR / f"{target}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Run 'uv run python encoding/build_dataset.py' first."
        )
    return dict(np.load(path, allow_pickle=False))


if __name__ == "__main__":
    metadata = build_dataset()

    # Print summary
    for target_name, target_data in metadata["targets"].items():
        print(f"\n{'='*60}")
        print(f"Target: {target_name}")
        for split_name, split_data in target_data["shapes"].items():
            dist = target_data["label_distributions"][split_name]
            print(f"  {split_name}: {split_data} — labels: {dist}")
