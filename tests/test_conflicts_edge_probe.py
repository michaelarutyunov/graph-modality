"""Unit tests for s5_classification.conflicts_edge_probe."""

from __future__ import annotations

from s5_classification.conflicts_edge_probe import graph_to_conflict_features


def _stance(sid: str, valence: str) -> dict:
    return {"id": sid, "type": "Stance", "valence": valence}


def _construct(cid: str) -> dict:
    return {"id": cid, "type": "Construct"}


def _edge(source: str, target: str, relation: str) -> dict:
    return {"source": source, "target": target, "relation": relation}


def test_opposite_valence_conflict_counted() -> None:
    """Two constructs with opposite dominant stance valence, joined by
    CONFLICTS_WITH, should count toward n_opp_valence_conflicts."""
    nodes = [
        _construct("c1"),
        _construct("c2"),
        _stance("s1", "positive"),
        _stance("s2", "negative"),
    ]
    edges = [
        _edge("s1", "c1", "EXPRESSED_VIA"),
        _edge("s2", "c2", "EXPRESSED_VIA"),
        _edge("c1", "c2", "CONFLICTS_WITH"),
    ]
    graph = {"nodes": nodes, "edges": edges}

    features = graph_to_conflict_features(graph)
    n_conflict_edges, frac_conflict, n_opp, n_distinct = features

    assert n_conflict_edges == 1
    assert frac_conflict == 1.0 / 3.0
    assert n_opp == 1
    assert n_distinct == 2
    assert frac_conflict >= 0


def test_same_valence_conflict_not_counted() -> None:
    """Two constructs with the SAME dominant stance valence, joined by
    CONFLICTS_WITH, must NOT count toward n_opp_valence_conflicts."""
    nodes = [
        _construct("c1"),
        _construct("c2"),
        _stance("s1", "positive"),
        _stance("s2", "positive"),
    ]
    edges = [
        _edge("s1", "c1", "EXPRESSED_VIA"),
        _edge("s2", "c2", "EXPRESSED_VIA"),
        _edge("c1", "c2", "CONFLICTS_WITH"),
    ]
    graph = {"nodes": nodes, "edges": edges}

    features = graph_to_conflict_features(graph)
    n_conflict_edges, _frac_conflict, n_opp, n_distinct = features

    assert n_conflict_edges == 1
    assert n_opp == 0
    assert n_distinct == 2


def test_combined_graph_with_opposite_and_same_valence_pairs() -> None:
    """A graph with one opposite-valence conflict pair and one same-valence
    conflict pair should count exactly one opposite-valence conflict."""
    nodes = [
        _construct("c1"),
        _construct("c2"),
        _construct("c3"),
        _construct("c4"),
        _stance("s1", "positive"),
        _stance("s2", "negative"),
        _stance("s3", "positive"),
        _stance("s4", "positive"),
    ]
    edges = [
        _edge("s1", "c1", "EXPRESSED_VIA"),
        _edge("s2", "c2", "EXPRESSED_VIA"),
        _edge("s3", "c3", "EXPRESSED_VIA"),
        _edge("s4", "c4", "EXPRESSED_VIA"),
        _edge("c1", "c2", "CONFLICTS_WITH"),  # opposite valence -> counted
        _edge("c3", "c4", "CONFLICTS_WITH"),  # same valence -> not counted
    ]
    graph = {"nodes": nodes, "edges": edges}

    features = graph_to_conflict_features(graph)
    n_conflict_edges, frac_conflict, n_opp, n_distinct = features

    assert n_conflict_edges == 2
    assert n_opp == 1
    assert n_distinct == 4
    assert frac_conflict > 0


def test_no_conflict_edges() -> None:
    """A graph with no CONFLICTS_WITH edges yields all-zero conflict features."""
    nodes = [_construct("c1"), _stance("s1", "positive")]
    edges = [_edge("s1", "c1", "EXPRESSED_VIA")]
    graph = {"nodes": nodes, "edges": edges}

    features = graph_to_conflict_features(graph)
    n_conflict_edges, frac_conflict, n_opp, n_distinct = features

    assert n_conflict_edges == 0
    assert frac_conflict == 0.0
    assert n_opp == 0
    assert n_distinct == 0


def test_tie_in_dominant_valence_not_counted() -> None:
    """If a construct has tied positive/negative stance counts, its dominant
    valence is 'neither' and any conflict edge involving it is not counted."""
    nodes = [
        _construct("c1"),
        _construct("c2"),
        _stance("s1a", "positive"),
        _stance("s1b", "negative"),
        _stance("s2", "negative"),
    ]
    edges = [
        _edge("s1a", "c1", "EXPRESSED_VIA"),
        _edge("s1b", "c1", "EXPRESSED_VIA"),  # c1 tied -> 'neither'
        _edge("s2", "c2", "EXPRESSED_VIA"),
        _edge("c1", "c2", "CONFLICTS_WITH"),
    ]
    graph = {"nodes": nodes, "edges": edges}

    features = graph_to_conflict_features(graph)
    _n_conflict_edges, _frac_conflict, n_opp, _n_distinct = features

    assert n_opp == 0
