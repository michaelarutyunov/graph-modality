"""Tests for extraction.validator — round-3 `selfpos` / v5 schema (ADR-0008).

Covers structural constraints S1-S12 from `.claude/context/graph-schema.md`
(§ ROUND-3). The v4 path (`validate_graph`) is tested separately in
`test_validator.py` and is unaffected by these.
"""

from s2_extraction.validator import is_valid_selfpos, validate_selfpos_graph

# ── factories ─────────────────────────────────────────────────────────


def _self() -> dict:
    return {"id": "self", "type": "Self"}


def _ai() -> dict:
    return {"id": "ai", "type": "AI"}


def _task(id_: str, label: str = "a task", span: str = "I do the thing") -> dict:
    return {"id": id_, "type": "Task", "label": label, "grounding_span": span}


def _value(id_: str, value_type: str = "competence", label: str = "skill") -> dict:
    return {
        "id": id_,
        "type": "Value",
        "label": label,
        "value_type": value_type,
        "grounding_span": f"span for {label}",
    }


def _edge(src: str, tgt: str, rel: str, span: str = "evidence", tags=None) -> dict:
    e: dict = {"source": src, "target": tgt, "relation": rel, "grounding_span": span}
    if tags is not None:
        e["rationale_tags"] = tags
    return e


def _graph(nodes=None, edges=None, domain="AI's role in professional work") -> dict:
    return {
        "transcript_id": "selfpos_001",
        "domain": domain,
        "nodes": nodes or [],
        "edges": edges or [],
    }


def _canonical_graph() -> dict:
    """A fully conformant selfpos graph: one retained+coupled task, one shared task."""
    return _graph(
        nodes=[
            _self(),
            _ai(),
            _task("t1", "final creative decisions", "the decisions are mine"),
            _task("t2", "drafting", "I let it draft but I rewrite"),
            _value("v1", "identity", "creative authorship"),
        ],
        edges=[
            _edge("t1", "self", "RETAINED_BY", "the decisions are mine", tags=[]),
            _edge("t1", "v1", "SERVES", "it's who I am as a writer"),
            _edge(
                "t2",
                "ai",
                "SHARED_WITH",
                "I let it draft but I rewrite",
                tags=["output_efficiency"],
            ),
        ],
    )


# ── valid graph ───────────────────────────────────────────────────────


def test_canonical_graph_passes():
    g = _canonical_graph()
    assert validate_selfpos_graph(g) == []
    assert is_valid_selfpos(g)


# ── S1: node types ────────────────────────────────────────────────────


def test_s1_unknown_node_type():
    g = _graph(nodes=[_self(), {"id": "x", "type": "Construct", "label": "a"}])
    assert any("unknown node type 'Construct'" in v for v in validate_selfpos_graph(g))


# ── S2: exactly one Self, id 'self' ───────────────────────────────────


def test_s2_missing_self():
    g = _graph(nodes=[_task("t1")], edges=[_edge("t1", "self", "RETAINED_BY")])
    assert any("expected exactly one Self" in v for v in validate_selfpos_graph(g))


def test_s2_two_self_nodes():
    g = _graph(nodes=[_self(), {"id": "self2", "type": "Self"}])
    assert any("expected exactly one Self node, found 2" in v for v in validate_selfpos_graph(g))


def test_s2_self_wrong_id():
    g = _graph(
        nodes=[{"id": "me", "type": "Self"}, _task("t1")],
        edges=[_edge("t1", "me", "RETAINED_BY")],
    )
    assert any("Self node id must be 'self'" in v for v in validate_selfpos_graph(g))


# ── S3: AI cardinality / id / presence-iff ────────────────────────────


def test_s3_ai_present_without_cede_or_share():
    g = _graph(
        nodes=[_self(), _ai(), _task("t1")],
        edges=[_edge("t1", "self", "RETAINED_BY")],
    )
    assert any("orphan anchor" in v for v in validate_selfpos_graph(g))


def test_s3_cede_without_ai_node():
    g = _graph(
        nodes=[_self(), _task("t1")],
        edges=[_edge("t1", "ai", "CEDED_TO")],
    )
    violations = validate_selfpos_graph(g)
    assert any("no AI node" in v for v in violations)


def test_s3_ai_wrong_id():
    g = _graph(
        nodes=[_self(), {"id": "bot", "type": "AI"}, _task("t1")],
        edges=[_edge("t1", "bot", "CEDED_TO")],
    )
    assert any("AI node id must be 'ai'" in v for v in validate_selfpos_graph(g))


# ── S4 / S5: Task / Value fields ──────────────────────────────────────


def test_s4_task_missing_grounding_span():
    g = _graph(
        nodes=[_self(), {"id": "t1", "type": "Task", "label": "x"}],
        edges=[_edge("t1", "self", "RETAINED_BY")],
    )
    assert any("Task missing grounding_span" in v for v in validate_selfpos_graph(g))


def test_s5_value_missing_value_type():
    g = _graph(
        nodes=[
            _self(),
            _task("t1"),
            {"id": "v1", "type": "Value", "label": "skill", "grounding_span": "s"},
        ],
        edges=[_edge("t1", "self", "RETAINED_BY"), _edge("t1", "v1", "SERVES")],
    )
    assert any("Value missing value_type" in v for v in validate_selfpos_graph(g))


def test_s5_value_invalid_value_type():
    g = _graph(
        nodes=[_self(), _task("t1"), _value("v1", value_type="autonomy")],
        edges=[_edge("t1", "self", "RETAINED_BY"), _edge("t1", "v1", "SERVES")],
    )
    assert any("invalid value_type 'autonomy'" in v for v in validate_selfpos_graph(g))


# ── S6: relation types ────────────────────────────────────────────────


def test_s6_unknown_relation():
    g = _graph(
        nodes=[_self(), _task("t1")],
        edges=[_edge("t1", "self", "OWNS")],
    )
    assert any("unknown relation 'OWNS'" in v for v in validate_selfpos_graph(g))


# ── S7: relation type signatures ──────────────────────────────────────


def test_s7_retained_by_wrong_target():
    # RETAINED_BY must target Self, not AI
    g = _graph(
        nodes=[_self(), _ai(), _task("t1")],
        edges=[
            _edge("t1", "ai", "RETAINED_BY"),
            _edge("t1", "ai", "CEDED_TO"),  # keep AI non-orphan + give t1 a real boundary
        ],
    )
    # Note: t1 now has 2 boundary edges (S8) too; we only assert the S7 message here.
    assert any("RETAINED_BY requires Task→Self" in v for v in validate_selfpos_graph(g))


def test_s7_serves_wrong_target():
    g = _graph(
        nodes=[_self(), _task("t1"), _task("t2")],
        edges=[
            _edge("t1", "self", "RETAINED_BY"),
            _edge("t2", "self", "RETAINED_BY"),
            _edge("t1", "t2", "SERVES"),  # Task→Task, not Task→Value
        ],
    )
    assert any("SERVES requires Task→Value" in v for v in validate_selfpos_graph(g))


# ── S8: partition invariant ───────────────────────────────────────────


def test_s8_task_with_no_boundary_edge():
    g = _graph(
        nodes=[_self(), _task("t1"), _value("v1")],
        edges=[_edge("t1", "v1", "SERVES")],
    )
    assert any(
        "exactly one boundary edge" in v and "found 0" in v for v in validate_selfpos_graph(g)
    )


def test_s8_task_with_two_boundary_edges():
    g = _graph(
        nodes=[_self(), _ai(), _task("t1")],
        edges=[
            _edge("t1", "self", "RETAINED_BY"),
            _edge("t1", "ai", "CEDED_TO"),
        ],
    )
    assert any("found 2" in v for v in validate_selfpos_graph(g))


# ── S9: referential integrity ─────────────────────────────────────────


def test_s9_dangling_target():
    g = _graph(
        nodes=[_self(), _task("t1")],
        edges=[_edge("t1", "ghost", "RETAINED_BY")],
    )
    # ghost target absent — and t1's boundary edge points nowhere valid
    assert any("'ghost' not in node IDs" in v for v in validate_selfpos_graph(g))


# ── S10: edge grounding span ──────────────────────────────────────────


def test_s10_edge_missing_grounding_span():
    g = _graph(
        nodes=[_self(), _task("t1")],
        edges=[{"source": "t1", "target": "self", "relation": "RETAINED_BY"}],
    )
    assert any("missing grounding_span" in v for v in validate_selfpos_graph(g))


# ── S11: rationale_tags placement / membership ────────────────────────


def test_s11_serves_with_rationale_tags():
    g = _graph(
        nodes=[_self(), _task("t1"), _value("v1")],
        edges=[
            _edge("t1", "self", "RETAINED_BY"),
            _edge("t1", "v1", "SERVES", tags=["trust_reliability"]),
        ],
    )
    assert any("SERVES must not carry rationale_tags" in v for v in validate_selfpos_graph(g))


def test_s11_invalid_rationale_tag():
    g = _graph(
        nodes=[_self(), _task("t1")],
        edges=[_edge("t1", "self", "RETAINED_BY", tags=["competence"])],
    )
    # 'competence' is a value_type, not a boundary rationale tag
    assert any("invalid rationale_tags" in v for v in validate_selfpos_graph(g))


def test_s11_valid_rationale_tags_pass():
    g = _graph(
        nodes=[_self(), _ai(), _task("t1")],
        edges=[
            _edge("t1", "ai", "CEDED_TO", tags=["output_efficiency", "competence_compensate"]),
        ],
    )
    assert validate_selfpos_graph(g) == []


# ── S12: domain ───────────────────────────────────────────────────────


def test_s12_missing_domain():
    g = {
        "transcript_id": "selfpos_002",
        "nodes": [_self(), _task("t1")],
        "edges": [_edge("t1", "self", "RETAINED_BY")],
    }
    assert any("missing 'domain' field" in v for v in validate_selfpos_graph(g))


# ── readout-shaped graphs are valid (sanity for downstream) ───────────


def test_breadth_and_alignment_graph_valid():
    """A graph exercising retained+coupled, ceded, and shared dispositions."""
    g = _graph(
        nodes=[
            _self(),
            _ai(),
            _task("t1", "analysis"),
            _task("t2", "formatting"),
            _task("t3", "drafting"),
            _value("v1", "competence", "judgment"),
        ],
        edges=[
            _edge("t1", "self", "RETAINED_BY", tags=["trust_reliability"]),
            _edge("t1", "v1", "SERVES"),
            _edge("t2", "ai", "CEDED_TO", tags=["output_efficiency"]),
            _edge("t3", "ai", "SHARED_WITH", tags=[]),
        ],
    )
    assert validate_selfpos_graph(g) == []
    assert is_valid_selfpos(g)
