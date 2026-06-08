"""Tests for canonicalisation.clusterer."""

import pytest
from sentence_transformers import SentenceTransformer

from canonicalisation.clusterer import (
    build_canonical_map,
    cluster_labels,
    load_all_labels,
)


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── load_all_labels ──────────────────────────────────────────────────


def test_load_all_labels_returns_all_entity_types(tmp_path):
    import json
    g = {
        "transcript_id": "test",
        "nodes": [
            {"id": "n1", "type": "Construct", "label": "trust ↔ distrust"},
            {"id": "n2", "type": "Value", "label": "autonomy"},
            {"id": "n3", "type": "Stance", "label": "sceptical"},
            {"id": "n4", "type": "CognitiveStyleMarker", "label": "verification-first"},
        ],
        "edges": [],
    }
    (tmp_path / "test.json").write_text(json.dumps(g))

    result = load_all_labels(tmp_path)
    assert result["Construct"] == {"trust ↔ distrust"}
    assert result["Value"] == {"autonomy"}
    assert result["Stance"] == {"sceptical"}
    assert result["CognitiveStyleMarker"] == {"verification-first"}


def test_load_all_labels_handles_missing_types(tmp_path):
    import json
    g = {
        "transcript_id": "test",
        "nodes": [{"id": "n1", "type": "Value", "label": "autonomy"}],
        "edges": [],
    }
    (tmp_path / "test.json").write_text(json.dumps(g))

    result = load_all_labels(tmp_path)
    assert result["Value"] == {"autonomy"}
    assert result["Construct"] == set()
    assert result["Stance"] == set()
    assert result["CognitiveStyleMarker"] == set()


def test_load_all_labels_deduplicates(tmp_path):
    import json
    g1 = {
        "transcript_id": "t1",
        "nodes": [{"id": "n1", "type": "Value", "label": "autonomy"}],
        "edges": [],
    }
    g2 = {
        "transcript_id": "t2",
        "nodes": [{"id": "n1", "type": "Value", "label": "autonomy"}],
        "edges": [],
    }
    (tmp_path / "t1.json").write_text(json.dumps(g1))
    (tmp_path / "t2.json").write_text(json.dumps(g2))

    result = load_all_labels(tmp_path)
    assert result["Value"] == {"autonomy"}


def test_load_all_labels_empty_dir(tmp_path):
    result = load_all_labels(tmp_path)
    for etype in ["Construct", "Value", "Stance", "CognitiveStyleMarker"]:
        assert result[etype] == set()


# ── cluster_labels ───────────────────────────────────────────────────


def test_cluster_identical_labels_map_to_same(embedder):
    labels = ["autonomy", "autonomy", "autonomy"]
    mapping = cluster_labels(labels, embedder)
    assert len(set(mapping.values())) == 1
    assert mapping["autonomy"] == "autonomy"


def test_cluster_single_label(embedder):
    mapping = cluster_labels(["autonomy"], embedder)
    assert mapping == {"autonomy": "autonomy"}


def test_cluster_empty_list(embedder):
    mapping = cluster_labels([], embedder)
    assert mapping == {}


def test_cluster_semantically_similar_labels(embedder):
    labels = [
        "AI reliability and accuracy",
        "AI dependability and correctness",
        "personal autonomy and self-expression",
    ]
    mapping = cluster_labels(labels, embedder)
    # First two should be in the same cluster
    canonicals = set(mapping.values())
    assert len(canonicals) <= 2  # at most 2 clusters for these labels


def test_cluster_returns_all_input_labels(embedder):
    labels = ["trust in AI", "AI reliability", "creative control", "efficiency"]
    mapping = cluster_labels(labels, embedder)
    for label in labels:
        assert label in mapping


# ── build_canonical_map ─────────────────────────────────────────────


def test_build_canonical_map_all_entity_types_present(tmp_path):
    import json
    g = {
        "transcript_id": "test",
        "nodes": [
            {"id": "n1", "type": "Construct", "label": "trust ↔ distrust"},
            {"id": "n2", "type": "Value", "label": "autonomy"},
            {"id": "n3", "type": "Stance", "label": "sceptical", "valence": "mixed"},
            {"id": "n4", "type": "CognitiveStyleMarker", "label": "verification-first"},
        ],
        "edges": [],
    }
    (tmp_path / "test.json").write_text(json.dumps(g))

    result = build_canonical_map(tmp_path)
    for etype in ["Construct", "Value", "Stance", "CognitiveStyleMarker"]:
        assert etype in result
        assert len(result[etype]) > 0


def test_build_canonical_map_canonical_self_maps(tmp_path):
    """Every canonical label should appear as its own value (it maps to itself)."""
    import json
    g = {
        "transcript_id": "test",
        "nodes": [
            {"id": "n1", "type": "Construct", "label": "trust ↔ distrust"},
            {"id": "n2", "type": "Value", "label": "autonomy"},
            {"id": "n3", "type": "Value", "label": "personal freedom"},
        ],
        "edges": [],
    }
    (tmp_path / "test.json").write_text(json.dumps(g))

    result = build_canonical_map(tmp_path)
    for etype, mapping in result.items():
        canonicals = set(mapping.values())
        for c in canonicals:
            assert mapping.get(c) == c, f"{c} does not map to itself in {etype}"
