"""Tests for s5_classification/repeated_eval.py."""

import json
from pathlib import Path

import pytest

from s5_classification.repeated_eval import get_split_data, make_split, slice_modalities

SPLIT_IDS_PATH = Path("cache/split_ids.json")


@pytest.mark.skipif(not SPLIT_IDS_PATH.exists(), reason="cache/split_ids.json not present")
def test_make_split_matches_cached_seed42():
    with open(SPLIT_IDS_PATH) as f:
        cached = json.load(f)

    train_ids, val_ids, test_ids = make_split("cohort", 42)

    assert train_ids == cached["train"]
    assert val_ids == cached["val"]
    assert test_ids == cached["test"]


@pytest.mark.skipif(not Path("cache").exists(), reason="cache/ not present")
def test_get_split_data_ai_adoption():
    train, val, test = get_split_data("ai_adoption", 0)

    total = len(train["transcript_ids"]) + len(val["transcript_ids"]) + len(test["transcript_ids"])
    assert total == pytest.approx(1224, abs=5)

    for split in (train, val, test):
        for key in ("text_emb", "stats_emb", "graph_emb", "labels", "transcript_ids"):
            assert key in split
            assert len(split[key]) == len(split["transcript_ids"])


@pytest.mark.skipif(not Path("cache").exists(), reason="cache/ not present")
def test_slice_modalities_missing_id_raises():
    with pytest.raises(ValueError, match="not-a-real-id"):
        slice_modalities("cohort", ["not-a-real-id"])
