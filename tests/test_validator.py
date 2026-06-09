"""Tests for extraction.validator."""

from s2_extraction.validator import is_valid, validate_graph


def _make_graph(
    transcript_id: str = "test_001",
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
) -> dict:
    return {
        "transcript_id": transcript_id,
        "nodes": nodes or [],
        "edges": edges or [],
    }


# ── valid graph ──────────────────────────────────────────────────────


def test_valid_graph_passes_all_checks():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "human nuance ↔ technical output",
                "label_negative": "technical output",
                "bipolarity_complete": True,
                "grounding_span": "AI can't capture the nuance",
            },
            {
                "id": "n2",
                "type": "Value",
                "label": "professional identity",
                "grounding_span": "my identity as a creative",
            },
            {
                "id": "n3",
                "type": "Stance",
                "label": "anxious",
                "valence": "negative",
                "grounding_span": "I'm worried about it",
            },
        ],
        edges=[
            {"source": "n1", "target": "n2", "relation": "SERVES"},
            {"source": "n3", "target": "n1", "relation": "EXPRESSED_VIA"},
        ],
    )
    violations = validate_graph(g)
    assert violations == []
    assert is_valid(g)


# ── bipolarity ───────────────────────────────────────────────────────


def test_missing_negative_pole_flags_incomplete():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "AI speed",
                "bipolarity_complete": True,
                "grounding_span": "AI makes things faster",
            },
        ],
    )
    violations = validate_graph(g)
    assert any("missing negative pole" in v for v in violations)
    # Side-effect: bipolarity_complete set to False
    assert g["nodes"][0]["bipolarity_complete"] is False


def test_present_negative_pole_no_violation():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "trust ↔ distrust",
                "label_negative": "distrust",
                "bipolarity_complete": True,
                "grounding_span": "do I trust it",
            },
        ],
    )
    violations = validate_graph(g)
    assert not any("missing negative pole" in v for v in violations)


# ── CognitiveStyleMarker ceiling ─────────────────────────────────────


def test_csm_within_ceiling_no_violation():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "CognitiveStyleMarker",
                "label": "verification-first",
                "grounding_span": "I always check",
            },
            {
                "id": "n2",
                "type": "CognitiveStyleMarker",
                "label": "loss-averse",
                "grounding_span": "I don't want to lose",
            },
        ],
    )
    violations = validate_graph(g)
    assert not any("CognitiveStyleMarker count" in v for v in violations)


def test_csm_exceeds_ceiling():
    g = _make_graph(
        nodes=[
            {
                "id": f"n{i}",
                "type": "CognitiveStyleMarker",
                "label": f"style-{i}",
                "grounding_span": f"span {i}",
            }
            for i in range(1, 4)
        ],
    )
    violations = validate_graph(g)
    assert any("CognitiveStyleMarker count" in v for v in violations)
    assert any("exceeds ceiling of 2" in v for v in violations)


# ── edge validity ────────────────────────────────────────────────────


def test_dangling_edge_source():
    g = _make_graph(
        nodes=[
            {"id": "n1", "type": "Value", "label": "autonomy", "grounding_span": "I value autonomy"}
        ],
        edges=[{"source": "nonexistent", "target": "n1", "relation": "SERVES"}],
    )
    violations = validate_graph(g)
    assert any("source" in v and "nonexistent" in v for v in violations)


def test_dangling_edge_target():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "speed ↔ quality",
                "label_negative": "quality",
                "bipolarity_complete": True,
                "grounding_span": "speed vs quality",
            },
            {
                "id": "n2",
                "type": "Value",
                "label": "efficiency",
                "grounding_span": "efficiency matters",
            },
        ],
        edges=[{"source": "n1", "target": "n2", "relation": "SERVES"}],
    )
    violations = validate_graph(g)
    assert violations == []


def test_unknown_relation_type():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "a ↔ b",
                "label_negative": "b",
                "bipolarity_complete": True,
                "grounding_span": "a vs b",
            },
            {"id": "n2", "type": "Value", "label": "c", "grounding_span": "c"},
        ],
        edges=[{"source": "n1", "target": "n2", "relation": "CAUSES"}],
    )
    violations = validate_graph(g)
    assert any("unknown relation" in v for v in violations)


# ── disallowed Stance → Value ────────────────────────────────────────


def test_stance_to_value_direct_edge():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Stance",
                "label": "sceptical",
                "valence": "mixed",
                "grounding_span": "I'm not sure",
            },
            {
                "id": "n2",
                "type": "Value",
                "label": "epistemic rigour",
                "grounding_span": "I need to be rigorous",
            },
        ],
        edges=[{"source": "n1", "target": "n2", "relation": "EXPRESSED_VIA"}],
    )
    violations = validate_graph(g)
    assert any("Stance→Value" in v for v in violations)


def test_stance_to_value_via_construct_no_violation():
    """Stance → Construct → Value is valid (value-mediated)."""
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Stance",
                "label": "sceptical",
                "valence": "mixed",
                "grounding_span": "I'm not sure",
            },
            {
                "id": "n2",
                "type": "Construct",
                "label": "trust ↔ verify",
                "label_negative": "verify",
                "bipolarity_complete": True,
                "grounding_span": "trust but verify",
            },
            {
                "id": "n3",
                "type": "Value",
                "label": "epistemic rigour",
                "grounding_span": "I need to be rigorous",
            },
        ],
        edges=[
            {"source": "n1", "target": "n2", "relation": "EXPRESSED_VIA"},
            {"source": "n2", "target": "n3", "relation": "SERVES"},
        ],
    )
    violations = validate_graph(g)
    assert not any("Stance→Value" in v for v in violations)


# ── unknown entity type ──────────────────────────────────────────────


def test_unknown_entity_type():
    g = _make_graph(
        nodes=[{"id": "n1", "type": "Emotion", "label": "happy", "grounding_span": "I'm happy"}],
    )
    violations = validate_graph(g)
    assert any("unknown entity type" in v for v in violations)


# ── Stance valence ───────────────────────────────────────────────────


def test_stance_missing_valence():
    g = _make_graph(
        nodes=[
            {"id": "n1", "type": "Stance", "label": "uncertain", "grounding_span": "I'm uncertain"}
        ],
    )
    violations = validate_graph(g)
    assert any("Stance missing valence" in v for v in violations)


def test_stance_invalid_valence():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Stance",
                "label": "uncertain",
                "valence": "neutral",
                "grounding_span": "I'm uncertain",
            }
        ],
    )
    violations = validate_graph(g)
    assert any("invalid valence" in v for v in violations)


# ── is_valid convenience ─────────────────────────────────────────────


def test_is_valid_true_for_clean_graph():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "a ↔ b",
                "label_negative": "b",
                "bipolarity_complete": True,
                "grounding_span": "a vs b",
            },
        ],
    )
    assert is_valid(g)


def test_is_valid_false_for_broken_graph():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "a",
                "bipolarity_complete": True,
                "grounding_span": "a",
            },
        ],
    )
    assert not is_valid(g)
