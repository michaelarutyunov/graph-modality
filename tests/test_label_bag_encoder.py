"""Tests for s4_encoding/label_bag_encoder.py (P2.2)."""

import json

import numpy as np

from s4_encoding.label_bag_encoder import EMBEDDING_DIM, encode_label_bag


def _write_graph(path, transcript_id, nodes):
    path.write_text(json.dumps({"transcript_id": transcript_id, "nodes": nodes, "edges": []}))


def test_encode_label_bag_deterministic_and_empty_graph(tmp_path):
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()

    _write_graph(
        graph_dir / "t0.json",
        "t0",
        [{"id": "n1", "type": "Construct", "label": "trust"}],
    )
    _write_graph(graph_dir / "t1.json", "t1", [])  # empty graph

    embs1, ids1 = encode_label_bag(graph_dir=graph_dir, force=True)
    embs2, ids2 = encode_label_bag(graph_dir=graph_dir, force=True)

    assert ids1 == ids2 == ["t0", "t1"]
    assert embs1.shape == (2, EMBEDDING_DIM)
    np.testing.assert_array_equal(embs1, embs2)

    # Empty-node graph -> zero vector
    np.testing.assert_array_equal(embs1[1], np.zeros(EMBEDDING_DIM, dtype=np.float32))
    assert not np.allclose(embs1[0], 0.0)
