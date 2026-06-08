"""Fixed train/val/test split for transcript classification.

This module creates and loads a stratified 70/15/15 split of the 1,250 transcripts,
preserving class distribution across splits. The split is deterministic (seed=42)
and cached to disk for reproducibility.

Label mapping:
- workforce: 0
- creatives: 1
- scientists: 2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.model_selection import train_test_split

# Paths
TAGGED_DIR = Path("data/tagged")
CACHE_DIR = Path("cache")
SPLIT_PATH = CACHE_DIR / "split_ids.json"

# Configuration
LABEL_MAP = {"workforce": 0, "creatives": 1, "scientists": 2}
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_transcript_ids_with_labels() -> dict[str, int]:
    """Load all transcript IDs with their integer labels.

    Returns:
        Dictionary mapping transcript_id to integer label (0/1/2).
    """
    ids_to_labels: dict[str, int] = {}

    # Load each cohort file
    for cohort_file in TAGGED_DIR.glob("*.jsonl"):
        if cohort_file.name == ".gitkeep":
            continue

        # Get cohort name from filename (e.g., "workforce.jsonl" -> "workforce")
        cohort = cohort_file.stem

        # Read JSONL and extract IDs
        df = pl.read_ndjson(cohort_file).select(["transcript_id"])

        # Map to integer label
        label = LABEL_MAP[cohort]

        # Add to mapping
        for transcript_id in df["transcript_id"].to_list():
            ids_to_labels[transcript_id] = label

    return ids_to_labels


def create_split() -> dict[str, Any]:
    """Create stratified train/val/test split and save to cache.

    This function:
    1. Loads all transcript IDs with labels
    2. Creates 70/15/15 stratified split using sklearn
    3. Saves split IDs to cache/split_ids.json
    4. Returns the split dictionary

    Returns:
        Dictionary with train/val/test ID lists and metadata.
    """
    # Ensure cache directory exists
    CACHE_DIR.mkdir(exist_ok=True)

    # Load all transcript data
    ids_to_labels = load_transcript_ids_with_labels()
    ids = list(ids_to_labels.keys())
    labels = [ids_to_labels[tx_id] for tx_id in ids]

    print(f"Loaded {len(ids)} transcripts with labels")

    # First split: 70% train, 30% temp
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        ids,
        labels,
        train_size=TRAIN_RATIO,
        stratify=labels,
        random_state=SEED,
    )

    # Second split: temp -> 50% val, 50% test (gives 15%/15% of total)
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        temp_ids,
        temp_labels,
        train_size=0.5,  # 50% of temp = 15% of total
        stratify=temp_labels,
        random_state=SEED,
    )

    # Calculate class distributions
    def count_labels(label_list: list[int]) -> dict[str, int]:
        """Count occurrences of each label in a split."""
        counts = {"workforce": 0, "creatives": 0, "scientists": 0}
        reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
        for label in label_list:
            cohort_name = reverse_label_map[label]
            counts[cohort_name] += 1
        return counts

    train_dist = count_labels(train_labels)
    val_dist = count_labels(val_labels)
    test_dist = count_labels(test_labels)

    # Build split dictionary
    split_dict: dict[str, Any] = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "label_map": LABEL_MAP,
        "seed": SEED,
        "metadata": {
            "train_size": len(train_ids),
            "val_size": len(val_ids),
            "test_size": len(test_ids),
            "class_distribution": {
                "train": train_dist,
                "val": val_dist,
                "test": test_dist,
            },
        },
    }

    # Save to cache
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_dict, f, indent=2)

    # Print summary
    print(f"\nSplit created and saved to {SPLIT_PATH}")
    print(f"Train: {len(train_ids)} ({len(train_ids) / len(ids) * 100:.1f}%)")
    print(f"Val: {len(val_ids)} ({len(val_ids) / len(ids) * 100:.1f}%)")
    print(f"Test: {len(test_ids)} ({len(test_ids) / len(ids) * 100:.1f}%)")
    print("\nClass distribution:")
    for split_name, dist in [("train", train_dist), ("val", val_dist), ("test", test_dist)]:
        print(f"  {split_name}: {dist}")

    return split_dict


def load_split() -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Load cached split from disk.

    Returns:
        Tuple of (train_ids, val_ids, test_ids, labels_dict).
        labels_dict maps transcript_id -> int label.
    """
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found at {SPLIT_PATH}. Run create_split() first to generate it."
        )

    with open(SPLIT_PATH) as f:
        split_dict = json.load(f)

    train_ids = split_dict["train"]
    val_ids = split_dict["val"]
    test_ids = split_dict["test"]

    # Load labels from source data
    ids_to_labels = load_transcript_ids_with_labels()

    # Filter to only include transcripts in the split
    all_split_ids = set(train_ids + val_ids + test_ids)
    labels_dict = {tx_id: ids_to_labels[tx_id] for tx_id in all_split_ids}

    return train_ids, val_ids, test_ids, labels_dict


def create_split_if_needed() -> dict[str, Any]:
    """Create split if it doesn't exist, otherwise load from cache.

    This is idempotent - safe to call multiple times.

    Returns:
        Split dictionary with train/val/test IDs and metadata.
    """
    if SPLIT_PATH.exists():
        print(f"Loading existing split from {SPLIT_PATH}")
        with open(SPLIT_PATH) as f:
            return json.load(f)
    else:
        return create_split()


if __name__ == "__main__":
    # Create split and print summary
    split = create_split_if_needed()

    # Verify sizes
    metadata = split["metadata"]
    total = metadata["train_size"] + metadata["val_size"] + metadata["test_size"]
    print(f"\nTotal transcripts: {total}")
    assert total == 1250, f"Expected 1250 transcripts, got {total}"
