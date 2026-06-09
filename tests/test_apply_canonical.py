"""Tests for canonicalisation.apply_canonical."""

import json

import pytest

from s3_canonicalisation.apply_canonical import apply_all, canonicalise_graph, load_canonical_map


@pytest.fixture
def sample_map():
    return {
        "Construct": {
            "trust ↔ distrust": "trust in AI",
            "AI reliability": "trust in AI",
        },
        "Value": {"autonomy": "personal autonomy"},
        "Stance": {"sceptical": "cautious stance"},
        "CognitiveStyleMarker": {"verification-first": "verification-oriented"},
    }


def test_canonicalise_replaces_labels(sample_map):
    g = {
        "transcript_id": "test",
        "nodes": [
            {"id": "n1", "type": "Construct", "label": "trust ↔ distrust"},
            {"id": "n2", "type": "Construct", "label": "AI reliability"},
            {"id": "n3", "type": "Value", "label": "autonomy"},
        ],
        "edges": [],
    }
    cg = canonicalise_graph(g, sample_map)
    assert cg["nodes"][0]["label"] == "trust in AI"
    assert cg["nodes"][1]["label"] == "trust in AI"
    assert cg["nodes"][2]["label"] == "personal autonomy"


def test_canonicalise_preserves_non_label_fields(sample_map):
    g = {
        "transcript_id": "test",
        "nodes": [
            {
                "id": "n1",
                "type": "Construct",
                "label": "trust ↔ distrust",
                "label_negative": "distrust",
                "bipolarity_complete": True,
                "grounding_span": "do I trust AI",
            },
        ],
        "edges": [{"source": "n1", "target": "n2", "relation": "SERVES"}],
    }
    cg = canonicalise_graph(g, sample_map)
    n = cg["nodes"][0]
    assert n["id"] == "n1"
    assert n["label_negative"] == "distrust"
    assert n["bipolarity_complete"] is True
    assert n["grounding_span"] == "do I trust AI"
    assert cg["edges"][0]["relation"] == "SERVES"


def test_canonicalise_does_not_mutate_input(sample_map):
    g = {
        "transcript_id": "test",
        "nodes": [{"id": "n1", "type": "Construct", "label": "trust ↔ distrust"}],
        "edges": [],
    }
    original_label = g["nodes"][0]["label"]
    canonicalise_graph(g, sample_map)
    assert g["nodes"][0]["label"] == original_label


def test_canonicalise_handles_unknown_label(sample_map):
    g = {
        "transcript_id": "test",
        "nodes": [{"id": "n1", "type": "Construct", "label": "brand new concept"}],
        "edges": [],
    }
    cg = canonicalise_graph(g, sample_map)
    assert cg["nodes"][0]["label"] == "brand new concept"  # kept as-is


def test_load_canonical_map_detects_unlocked(tmp_path):
    map_path = tmp_path / "canonical_map.json"
    map_path.write_text(json.dumps({"Construct": {}, "_meta": {"locked": False}}))
    import s3_canonicalisation.apply_canonical as ac

    orig = ac.MAP_PATH
    ac.MAP_PATH = map_path
    try:
        with pytest.raises(RuntimeError, match="not locked"):
            load_canonical_map()
    finally:
        ac.MAP_PATH = orig


def test_apply_all_output_count_matches_input(tmp_path, sample_map):
    # Write test graphs
    free = tmp_path / "free"
    free.mkdir()
    for i in range(3):
        g = {
            "transcript_id": f"t{i}",
            "nodes": [{
                "id": "n1", "type": "Construct", "label": "trust ↔ distrust",
                "label_negative": "distrust", "bipolarity_complete": True,
                "grounding_span": "test",
            }],
            "edges": [],
        }
        (free / f"t{i}.json").write_text(json.dumps(g))

    # Write a locked map
    map_path = tmp_path / "map.json"
    cm = {**sample_map, "_meta": {"locked": True, "locked_at": "2026-01-01T00:00:00"}}
    map_path.write_text(json.dumps(cm))

    # Override MAP_PATH so load_canonical_map() reads our test map
    import s3_canonicalisation.apply_canonical as ac

    orig_map = ac.MAP_PATH
    ac.MAP_PATH = map_path
    try:
        canon_dir = tmp_path / "canonical"
        total, clean, unmapped = apply_all(free_text_dir=free, canonical_dir=canon_dir)
        assert total == 3
        assert clean == 3
        assert unmapped == 0
        assert len(list(canon_dir.glob("*.json"))) == 3
    finally:
        ac.MAP_PATH = orig_map
