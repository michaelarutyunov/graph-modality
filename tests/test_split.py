"""Tests for classification/split.py module."""

import os
from pathlib import Path

from classification.split import (
    LABEL_MAP,
    SEED,
    create_split,
    create_split_if_needed,
    load_split,
    load_transcript_ids_with_labels,
)

# Set test environment variable to avoid polluting production beads DB
os.environ["BEADS_DB"] = "/tmp/test_beads.db"


def test_split_sizes():
    """Test that train/val/test sizes are correct."""
    split = create_split_if_needed()

    train_size = split["metadata"]["train_size"]
    val_size = split["metadata"]["val_size"]
    test_size = split["metadata"]["test_size"]
    total = train_size + val_size + test_size

    # Check total
    assert total == 1250, f"Expected 1250 total, got {total}"

    # Check sizes are within expected range (70/15/15)
    assert 870 <= train_size <= 880, f"Train size {train_size} not in range [870, 880]"
    assert 180 <= val_size <= 195, f"Val size {val_size} not in range [180, 195]"
    assert 180 <= test_size <= 195, f"Test size {test_size} not in range [180, 195]"


def test_stratification():
    """Test that class proportions are preserved across splits."""
    ids_to_labels = load_transcript_ids_with_labels()

    # Get total counts per class
    total_workforce = sum(1 for label in ids_to_labels.values() if label == 0)
    total_creatives = sum(1 for label in ids_to_labels.values() if label == 1)
    total_scientists = sum(1 for label in ids_to_labels.values() if label == 2)

    split = create_split_if_needed()
    train_dist = split["metadata"]["class_distribution"]["train"]
    val_dist = split["metadata"]["class_distribution"]["val"]
    test_dist = split["metadata"]["class_distribution"]["test"]

    # Check train distribution
    train_total = split["metadata"]["train_size"]
    train_workforce_ratio = train_dist["workforce"] / train_total
    train_creatives_ratio = train_dist["creatives"] / train_total
    train_scientists_ratio = train_dist["scientists"] / train_total

    # Expected ratios (should be close to population ratios)
    expected_workforce_ratio = total_workforce / 1250
    expected_creatives_ratio = total_creatives / 1250
    expected_scientists_ratio = total_scientists / 1250

    # Allow 5% tolerance due to stratification rounding
    assert abs(train_workforce_ratio - expected_workforce_ratio) < 0.05
    assert abs(train_creatives_ratio - expected_creatives_ratio) < 0.05
    assert abs(train_scientists_ratio - expected_scientists_ratio) < 0.05

    # Check that all splits have samples from all classes
    assert train_dist["workforce"] > 0
    assert train_dist["creatives"] > 0
    assert train_dist["scientists"] > 0
    assert val_dist["workforce"] > 0
    assert val_dist["creatives"] > 0
    assert val_dist["scientists"] > 0
    assert test_dist["workforce"] > 0
    assert test_dist["creatives"] > 0
    assert test_dist["scientists"] > 0


def test_no_leakage():
    """Test that there's no overlap between train/val/test sets."""
    split = create_split_if_needed()

    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])

    # Check no overlap
    assert len(train_ids & val_ids) == 0, "Train and val overlap"
    assert len(train_ids & test_ids) == 0, "Train and test overlap"
    assert len(val_ids & test_ids) == 0, "Val and test overlap"


def test_all_transcripts_included():
    """Test that all 1250 transcript IDs are in exactly one split."""
    split = create_split_if_needed()

    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])

    # Get all transcript IDs from source data
    ids_to_labels = load_transcript_ids_with_labels()
    all_source_ids = set(ids_to_labels.keys())

    # Union of splits should equal all source IDs
    split_union = train_ids | val_ids | test_ids

    assert len(split_union) == 1250, f"Union has {len(split_union)} IDs, expected 1250"
    assert split_union == all_source_ids, "Union of splits doesn't match source IDs"


def test_seed_reproducibility():
    """Test that deleting cache and re-running produces identical splits."""

    cache_path = Path("cache/split_ids.json")

    # Load original split
    split1 = create_split_if_needed()
    original_train = split1["train"]
    original_val = split1["val"]
    original_test = split1["test"]

    # Delete cache
    if cache_path.exists():
        cache_path.unlink()

    # Recreate split
    split2 = create_split()

    # Should be identical
    assert split2["train"] == original_train, "Train IDs changed on re-run"
    assert split2["val"] == original_val, "Val IDs changed on re-run"
    assert split2["test"] == original_test, "Test IDs changed on re-run"


def test_label_map_correct():
    """Test that labels are 0/1/2 for workforce/creatives/scientists."""
    assert LABEL_MAP == {"workforce": 0, "creatives": 1, "scientists": 2}

    # Verify by checking loaded data
    ids_to_labels = load_transcript_ids_with_labels()

    # Check that we have all three labels
    unique_labels = set(ids_to_labels.values())
    assert unique_labels == {0, 1, 2}, f"Expected labels {{0, 1, 2}}, got {unique_labels}"

    # Check some known IDs
    assert ids_to_labels.get("work_0000") == 0, "work_0000 should be label 0"
    assert ids_to_labels.get("creativity_0000") == 1, "creativity_0000 should be label 1"
    assert ids_to_labels.get("science_0000") == 2, "science_0000 should be label 2"


def test_load_split():
    """Test that load_split returns correct data."""
    train_ids, val_ids, test_ids, labels_dict = load_split()

    # Check sizes
    assert len(train_ids) == 875
    assert len(val_ids) == 187
    assert len(test_ids) == 188

    # Check that labels_dict has correct size
    assert len(labels_dict) == 1250

    # Check that labels are 0/1/2
    unique_labels = set(labels_dict.values())
    assert unique_labels == {0, 1, 2}

    # Check that all split IDs are in labels_dict
    all_split_ids = set(train_ids + val_ids + test_ids)
    assert all_split_ids.issubset(set(labels_dict.keys()))


def test_split_metadata():
    """Test that split metadata is complete and correct."""
    split = create_split_if_needed()

    # Check required fields
    assert "train" in split
    assert "val" in split
    assert "test" in split
    assert "label_map" in split
    assert "seed" in split
    assert "metadata" in split

    # Check metadata structure
    metadata = split["metadata"]
    assert "train_size" in metadata
    assert "val_size" in metadata
    assert "test_size" in metadata
    assert "class_distribution" in metadata

    # Check class distribution structure
    class_dist = metadata["class_distribution"]
    assert "train" in class_dist
    assert "val" in class_dist
    assert "test" in class_dist

    # Check that sizes match actual list lengths
    assert metadata["train_size"] == len(split["train"])
    assert metadata["val_size"] == len(split["val"])
    assert metadata["test_size"] == len(split["test"])

    # Check seed
    assert split["seed"] == SEED
