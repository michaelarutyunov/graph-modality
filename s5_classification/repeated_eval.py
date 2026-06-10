"""Repeated-split slicing utility over frozen modality embeddings.

Builds train/val/test ID lists for arbitrary seeds (replicating the two-stage
stratified procedure of ``split.py``) and slices the cached frozen embeddings
to those IDs, without recomputing any embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from s4_encoding.build_dataset import _load_ai_adoption_labels
from s4_encoding.graph_gnn_encoder import encode_graphs
from s4_encoding.graph_stats_encoder import compute_all_stats
from s4_encoding.text_encoder import encode_transcripts
from s5_classification.split import TRAIN_RATIO, load_transcript_ids_with_labels

CACHE_DIR = Path("cache")
SPLITS_DIR = CACHE_DIR / "repeated_splits"


def make_split(target: str, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Return (train_ids, val_ids, test_ids) for the given target and seed.

    Replicates the exact two-stage stratified procedure of ``split.py``,
    substituting ``random_state=seed``. Results are cached to
    ``cache/repeated_splits/{target}_seed{seed}.json``.
    """
    cache_path = SPLITS_DIR / f"{target}_seed{seed}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["train"], cached["val"], cached["test"]

    if target == "cohort":
        ids_to_labels = load_transcript_ids_with_labels()
    elif target == "ai_adoption":
        ids_to_labels = _load_ai_adoption_labels()
    else:
        raise ValueError(f"Unknown target: {target!r}")

    ids = list(ids_to_labels.keys())
    labels = [ids_to_labels[tid] for tid in ids]

    train_ids, temp_ids, _train_labels, temp_labels = train_test_split(
        ids,
        labels,
        train_size=TRAIN_RATIO,
        stratify=labels,
        random_state=seed,
    )
    val_ids, test_ids, _val_labels, _test_labels = train_test_split(
        temp_ids,
        temp_labels,
        train_size=0.5,
        stratify=temp_labels,
        random_state=seed,
    )

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)
    test_ids = sorted(test_ids)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids, "seed": seed}, f, indent=2)

    return train_ids, val_ids, test_ids


def slice_modalities(target: str, ids: list[str]) -> dict[str, np.ndarray]:
    """Load frozen modality caches and slice/align them to ``ids``.

    Raises:
        ValueError: if any requested id is missing from any modality cache.
    """
    text_embs, text_ids = encode_transcripts(speaker_filter="Human")
    stats_embs, stats_ids = compute_all_stats()
    graph_embs, graph_ids = encode_graphs(label_source="canonical")

    if target == "cohort":
        labels_dict = load_transcript_ids_with_labels()
    elif target == "ai_adoption":
        labels_dict = _load_ai_adoption_labels()
    else:
        raise ValueError(f"Unknown target: {target!r}")

    modalities = {
        "text_emb": (text_embs, text_ids),
        "stats_emb": (stats_embs, stats_ids),
        "graph_emb": (graph_embs, graph_ids),
    }

    for mod_name, (_embs, id_list) in modalities.items():
        id_set = set(id_list)
        missing = [tid for tid in ids if tid not in id_set]
        if missing:
            raise ValueError(f"Missing ids in {mod_name} cache: {missing}")

    missing_labels = [tid for tid in ids if tid not in labels_dict]
    if missing_labels:
        raise ValueError(f"Missing ids in label set for target {target!r}: {missing_labels}")

    result: dict[str, np.ndarray] = {}
    for mod_name, (embs, id_list) in modalities.items():
        lookup = {tid: i for i, tid in enumerate(id_list)}
        indices = [lookup[tid] for tid in ids]
        result[mod_name] = embs[indices]

    result["labels"] = np.array([labels_dict[tid] for tid in ids], dtype=np.int64)
    result["transcript_ids"] = np.array(ids)

    return result


def get_split_data(target: str, seed: int) -> tuple[dict, dict, dict]:
    """Return (train, val, test) dicts shaped like ``build_dataset.load_dataset()``."""
    train_ids, val_ids, test_ids = make_split(target, seed)
    return (
        slice_modalities(target, train_ids),
        slice_modalities(target, val_ids),
        slice_modalities(target, test_ids),
    )
