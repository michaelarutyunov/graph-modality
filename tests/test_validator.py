"""Tests for extraction.validator — v4 schema."""

from s2_extraction.validator import is_valid, validate_graph


def _make_graph(
    transcript_id: str = "test_001",
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
    domain: str = "AI's role in professional work",
) -> dict:
    return {
        "transcript_id": transcript_id,
        "domain": domain,
        "nodes": nodes or [],
        "edges": edges or [],
    }


def _construct(
    id_: str,
    label: str,
    label_negative: str | None = None,
    bipolarity_complete: bool = True,
    positive_spans: list[str] | None = None,
    negative_spans: list[str] | None = None,
) -> dict:
    """Factory for a v4 Construct node."""
    return {
        "id": id_,
        "type": "Construct",
        "label": label,
        "label_negative": label_negative,
        "bipolarity_complete": bipolarity_complete,
        "grounding_spans_positive": positive_spans or [f"span for {label}"],
        "grounding_spans_negative": negative_spans
        or ([f"span against {label}"] if label_negative else []),
    }


def _value(id_: str, label: str, spans: list[str] | None = None) -> dict:
    return {
        "id": id_,
        "type": "Value",
        "label": label,
        "grounding_spans": spans or [f"span for {label}"],
    }


def _stance(id_: str, label: str, valence: str, spans: list[str] | None = None) -> dict:
    return {
        "id": id_,
        "type": "Stance",
        "label": label,
        "valence": valence,
        "grounding_spans": spans or [f"span for {label}"],
    }


def _csm(id_: str, label: str, spans: list[str] | None = None) -> dict:
    return {
        "id": id_,
        "type": "CognitiveStyleMarker",
        "label": label,
        "grounding_spans": spans or [f"span 1 for {label}", f"span 2 for {label}"],
    }


def _edge(src: str, tgt: str, rel: str, rationale: str = "", grounding: str = "inferred") -> dict:
    return {
        "source": src,
        "target": tgt,
        "relation": rel,
        "rationale": rationale or f"{src} {rel} {tgt}",
        "grounding": grounding,
    }


# ── valid graph ──────────────────────────────────────────────────────


def test_valid_graph_passes_all_checks():
    g = _make_graph(
        nodes=[
            _construct("n1", "human nuance ↔ technical output", "technical output"),
            _value("n2", "professional identity"),
            _stance("n3", "anxious", "negative"),
        ],
        edges=[
            _edge("n1", "n2", "SERVES"),
            _edge("n3", "n1", "EXPRESSED_VIA"),
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
                "grounding_spans_positive": ["AI makes things faster"],
                "grounding_spans_negative": [],
            },
        ],
    )
    violations = validate_graph(g)
    # missing negative pole label
    assert any("missing negative pole" in v for v in violations)
    # bipolarity_complete set to False as side-effect
    assert g["nodes"][0]["bipolarity_complete"] is False


def test_present_negative_pole_no_violation():
    g = _make_graph(
        nodes=[
            _construct("n1", "trust ↔ distrust", "distrust"),
        ],
    )
    violations = validate_graph(g)
    assert not any("missing negative pole" in v for v in violations)


def test_bipolarity_complete_true_but_empty_grounding_violation():
    """bipolarity_complete=true but one pole has empty grounding list."""
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "trust ↔ distrust",
                "label_negative": "distrust",
                "bipolarity_complete": True,
                "grounding_spans_positive": ["I trust it"],
                "grounding_spans_negative": [],
            },
        ],
    )
    violations = validate_graph(g)
    assert any("bipolarity_complete=true but one pole has empty grounding" in v for v in violations)


# ── CognitiveStyleMarker recurrence ──────────────────────────────────


def test_csm_with_two_spans_no_violation():
    g = _make_graph(
        nodes=[
            _csm("n1", "verification-first", ["I always check", "I verify everything"]),
        ],
    )
    violations = validate_graph(g)
    assert not any("CognitiveStyleMarker" in v for v in violations)


def test_csm_insufficient_spans():
    """CSM with only 1 span should violate recurrence requirement."""
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "CognitiveStyleMarker",
                "label": "style-1",
                "grounding_spans": ["only one span"],
            },
        ],
    )
    violations = validate_graph(g)
    assert any("CognitiveStyleMarker has only 1" in v for v in violations)


def test_csm_many_allowed_no_ceiling():
    """v4 removed the max-2 CSM ceiling — many CSMs should be fine."""
    g = _make_graph(
        nodes=[_csm(f"n{i}", f"style-{i}", [f"span a {i}", f"span b {i}"]) for i in range(1, 6)],
    )
    violations = validate_graph(g)
    assert not any("ceiling" in v for v in violations)


# ── edge validity ────────────────────────────────────────────────────


def test_dangling_edge_source():
    g = _make_graph(
        nodes=[_value("n1", "autonomy")],
        edges=[_edge("nonexistent", "n1", "SERVES")],
    )
    violations = validate_graph(g)
    assert any("source" in v and "nonexistent" in v for v in violations)


def test_dangling_edge_target():
    g = _make_graph(
        nodes=[
            _construct("n1", "speed ↔ quality", "quality"),
            _value("n2", "efficiency"),
        ],
        edges=[_edge("n1", "n2", "SERVES")],
    )
    violations = validate_graph(g)
    assert violations == []


def test_unknown_relation_type():
    g = _make_graph(
        nodes=[
            _construct("n1", "a ↔ b", "b"),
            _value("n2", "c"),
        ],
        edges=[_edge("n1", "n2", "CAUSES")],
    )
    violations = validate_graph(g)
    assert any("unknown relation" in v for v in violations)


def test_edge_missing_rationale():
    g = _make_graph(
        nodes=[
            _construct("n1", "a ↔ b", "b"),
            _value("n2", "c"),
        ],
        edges=[{"source": "n1", "target": "n2", "relation": "SERVES"}],
    )
    violations = validate_graph(g)
    assert any("missing rationale" in v for v in violations)


def test_new_relation_types_accepted():
    """SUBSUMES and IMPLIES are valid v4 relations."""
    g = _make_graph(
        nodes=[
            _construct("n1", "a ↔ b", "b"),
            _construct("n2", "c ↔ d", "d"),
            _value("n3", "broad"),
            _value("n4", "specific"),
        ],
        edges=[
            _edge("n1", "n2", "IMPLIES"),
            _edge("n4", "n3", "SUBSUMES"),
        ],
    )
    violations = validate_graph(g)
    assert not any("unknown relation" in v for v in violations)


# ── relation type / node type consistency ────────────────────────────


def test_serves_requires_construct_to_value():
    g = _make_graph(
        nodes=[
            _value("n1", "autonomy"),
            _value("n2", "efficiency"),
        ],
        edges=[_edge("n1", "n2", "SERVES", "value serves value")],
    )
    violations = validate_graph(g)
    assert any("SERVES requires Construct→Value" in v for v in violations)


def test_expressed_via_requires_stance_to_construct():
    g = _make_graph(
        nodes=[
            _stance("n1", "happy", "positive"),
            _value("n2", "autonomy"),
        ],
        edges=[_edge("n1", "n2", "EXPRESSED_VIA", "stance to value")],
    )
    violations = validate_graph(g)
    assert any("EXPRESSED_VIA requires Stance→Construct" in v for v in violations)


def test_modulated_by_allows_stance_source():
    """v4: MODULATED_BY allowed from Stance → CSM, not just Construct → CSM."""
    g = _make_graph(
        nodes=[
            _stance("n1", "cautious", "negative"),
            _csm("n2", "verification-first"),
        ],
        edges=[_edge("n1", "n2", "MODULATED_BY", "cautious stance shaped by verification style")],
    )
    violations = validate_graph(g)
    assert violations == []


def test_modulated_by_disallows_value_source():
    g = _make_graph(
        nodes=[
            _value("n1", "autonomy"),
            _csm("n2", "verification-first"),
        ],
        edges=[_edge("n1", "n2", "MODULATED_BY", "value to csm")],
    )
    violations = validate_graph(g)
    assert any("MODULATED_BY requires Construct|Stance→CSM" in v for v in violations)


def test_subsumes_requires_value_to_value():
    g = _make_graph(
        nodes=[
            _construct("n1", "a ↔ b", "b"),
            _value("n2", "efficiency"),
        ],
        edges=[_edge("n1", "n2", "SUBSUMES", "construct subsumes value")],
    )
    violations = validate_graph(g)
    assert any("SUBSUMES requires Value→Value" in v for v in violations)


def test_implies_requires_construct_to_construct():
    g = _make_graph(
        nodes=[
            _construct("n1", "a ↔ b", "b"),
            _value("n2", "efficiency"),
        ],
        edges=[_edge("n1", "n2", "IMPLIES", "construct implies value")],
    )
    violations = validate_graph(g)
    assert any("IMPLIES requires Construct→Construct" in v for v in violations)


# ── edge grounding (C10, ADR-0004) ───────────────────────────────────


def test_missing_grounding_flagged():
    g = _make_graph(
        nodes=[_construct("n1", "a ↔ b", "b"), _value("n2", "efficiency")],
        edges=[{"source": "n1", "target": "n2", "relation": "SERVES", "rationale": "r"}],
    )
    violations = validate_graph(g)
    assert any("missing grounding" in v for v in violations)


def test_invalid_grounding_flagged():
    g = _make_graph(
        nodes=[_construct("n1", "a ↔ b", "b"), _value("n2", "efficiency")],
        edges=[_edge("n1", "n2", "SERVES", "r", grounding="quoted")],
    )
    violations = validate_graph(g)
    assert any("invalid grounding 'quoted'" in v for v in violations)


def test_explicit_grounding_accepted():
    g = _make_graph(
        nodes=[_construct("n1", "a ↔ b", "b"), _value("n2", "efficiency")],
        edges=[_edge("n1", "n2", "SERVES", "r", grounding="explicit")],
    )
    violations = validate_graph(g)
    assert violations == []


def test_inferential_relation_marked_explicit_flagged():
    """SUBSUMES/IMPLIES are inherently inferential — explicit grounding is contradictory."""
    g = _make_graph(
        nodes=[_value("n1", "efficiency"), _value("n2", "career success")],
        edges=[_edge("n1", "n2", "SUBSUMES", "efficiency is part of career success", "explicit")],
    )
    violations = validate_graph(g)
    assert any("inherently inferential" in v for v in violations)


# ── disallowed Stance → Value ────────────────────────────────────────


def test_stance_to_value_direct_edge():
    g = _make_graph(
        nodes=[
            _stance("n1", "sceptical", "mixed"),
            _value("n2", "epistemic rigour"),
        ],
        edges=[_edge("n1", "n2", "EXPRESSED_VIA", "stance to value")],
    )
    violations = validate_graph(g)
    assert any("Stance→Value" in v for v in violations)


def test_stance_to_value_via_construct_no_violation():
    """Stance → Construct → Value is valid (value-mediated)."""
    g = _make_graph(
        nodes=[
            _stance("n1", "sceptical", "mixed"),
            _construct("n2", "trust ↔ verify", "verify"),
            _value("n3", "epistemic rigour"),
        ],
        edges=[
            _edge("n1", "n2", "EXPRESSED_VIA", "scepticism via trust/verify"),
            _edge("n2", "n3", "SERVES", "trust/verify serves epistemic rigour"),
        ],
    )
    violations = validate_graph(g)
    assert not any("Stance→Value" in v for v in violations)


# ── unknown entity type ──────────────────────────────────────────────


def test_unknown_entity_type():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Emotion",
                "label": "happy",
                "grounding_spans": ["I'm happy"],
            }
        ],
    )
    violations = validate_graph(g)
    assert any("unknown entity type" in v for v in violations)


# ── Stance valence ───────────────────────────────────────────────────


def test_stance_missing_valence():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Stance",
                "label": "uncertain",
                "grounding_spans": ["I'm uncertain"],
            }
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
                "grounding_spans": ["I'm uncertain"],
            }
        ],
    )
    violations = validate_graph(g)
    assert any("invalid valence" in v for v in violations)


# ── grounding fields ─────────────────────────────────────────────────


def test_construct_with_old_grounding_span_warns():
    """v4 requires per-pole lists; old 'grounding_span' should warn."""
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
    violations = validate_graph(g)
    assert any("deprecated 'grounding_span' on Construct" in v for v in violations)


def test_value_with_old_grounding_span_warns():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Value",
                "label": "autonomy",
                "grounding_span": "I value autonomy",
            }
        ],
    )
    violations = validate_graph(g)
    assert any("deprecated 'grounding_span'" in v for v in violations)


def test_missing_grounding_spans_on_value():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Value",
                "label": "autonomy",
            }
        ],
    )
    violations = validate_graph(g)
    assert any("missing grounding_spans" in v for v in violations)


def test_missing_grounding_on_construct():
    g = _make_graph(
        nodes=[
            {
                "id": "n1",
                "type": "Construct",
                "label": "trust ↔ verify",
                "label_negative": "verify",
                "bipolarity_complete": True,
            }
        ],
    )
    violations = validate_graph(g)
    assert any("Construct missing grounding_spans_positive" in v for v in violations)


# ── domain ───────────────────────────────────────────────────────────


def test_missing_domain():
    g = {
        "transcript_id": "test_001",
        "nodes": [_construct("n1", "a ↔ b", "b")],
        "edges": [],
    }
    violations = validate_graph(g)
    assert any("missing 'domain' field" in v for v in violations)


# ── deprecated field backwards compatibility ─────────────────────────


def test_v3_style_graph_with_single_grounding_span_warns_but_handles():
    """v3 graphs with 'grounding_span' (singular) should produce deprecation
    warnings but not crash — they're upgraded to the list format."""
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
        ],
    )
    violations = validate_graph(g)
    # Should warn about deprecated field
    assert any("deprecated" in v for v in violations)
    # But should not crash
    assert isinstance(violations, list)


# ── is_valid convenience ─────────────────────────────────────────────


def test_is_valid_true_for_clean_graph():
    g = _make_graph(
        nodes=[_construct("n1", "a ↔ b", "b")],
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
                "grounding_spans_positive": ["a"],
                "grounding_spans_negative": [],
            },
        ],
    )
    assert not is_valid(g)
