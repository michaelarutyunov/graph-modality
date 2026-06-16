"""Structural validation of extracted concept graphs.

Runs after every extraction.  Checks against the ontology constraints
defined in the graph schema.  Returns violation strings; empty list = valid.

Usage:
    from s2_extraction.validator import validate_graph
    violations = validate_graph(graph)
    if violations:
        for v in violations:
            print(f"  VIOLATION: {v}")
"""

from __future__ import annotations

from typing import Any

# ── ontology constants ────────────────────────────────────────────────

ENTITY_TYPES = frozenset({"Construct", "Value", "Stance", "CognitiveStyleMarker"})
RELATION_TYPES = frozenset(
    {
        "SERVES",
        "EXPRESSED_VIA",
        "MODULATED_BY",
        "CONFLICTS_WITH",
        "SUBSUMES",
        "IMPLIES",
    }
)
VALID_VALENCES = frozenset({"positive", "negative", "mixed", "ambivalent"})
VALID_GROUNDING = frozenset({"explicit", "inferred"})
# Relations that are inherently inferential and must cite both endpoint spans (ADR-0004)
INFERENTIAL_RELATIONS = frozenset({"SUBSUMES", "IMPLIES"})
# CSM ceiling removed in v4 — quality controlled by recurrence requirement instead
CSM_MIN_SPANS = 2

# ── selfpos / round-3 (v5) ontology constants (ADR-0008) ──────────────
# Delegation-boundary ontology. NOT interoperable with the v4 constants above;
# validated by validate_selfpos_graph(), not validate_graph().

SELFPOS_NODE_TYPES = frozenset({"Self", "AI", "Task", "Value"})
SELFPOS_RELATION_TYPES = frozenset({"RETAINED_BY", "CEDED_TO", "SHARED_WITH", "SERVES"})
SELFPOS_BOUNDARY_RELATIONS = frozenset({"RETAINED_BY", "CEDED_TO", "SHARED_WITH"})
SELFPOS_VALUE_TYPES = frozenset({"competence", "identity"})
SELFPOS_RATIONALE_TAGS = frozenset(
    {"trust_reliability", "output_efficiency", "competence_compensate", "other"}
)
# Structural anchors injected by the extractor (grounding-exempt singletons).
SELFPOS_SELF_ID = "self"
SELFPOS_AI_ID = "ai"
# relation -> (required source type, required target type)
SELFPOS_RELATION_SIGNATURES: dict[str, tuple[str, str]] = {
    "RETAINED_BY": ("Task", "Self"),
    "CEDED_TO": ("Task", "AI"),
    "SHARED_WITH": ("Task", "AI"),
    "SERVES": ("Task", "Value"),
}


# ── public API ────────────────────────────────────────────────────────


def validate_graph(graph: dict[str, Any]) -> list[str]:
    """Validate a single extracted graph against all ontology constraints.

    Args:
        graph: A dict with ``transcript_id``, ``nodes``, and ``edges`` keys.

    Returns:
        List of human-readable violation strings.  An empty list means the
        graph passes all structural checks.
    """
    violations: list[str] = []

    nodes: list[dict[str, Any]] = graph.get("nodes", [])
    edges: list[dict[str, Any]] = graph.get("edges", [])
    tid: str = graph.get("transcript_id", "unknown")

    # Build lookup tables shared across checks.
    node_ids: set[str] = {n["id"] for n in nodes}
    node_type: dict[str, str] = {n["id"]: n["type"] for n in nodes}

    # ── per-node checks ────────────────────────────────────────────

    for n in nodes:
        nid = n.get("id", "?")
        ntype = n.get("type", "")

        # unknown entity type
        if ntype and ntype not in ENTITY_TYPES:
            violations.append(f"[{tid}] {nid}: unknown entity type '{ntype}'")

        # --- grounding checks (v4: lists required) ---
        if ntype == "Construct":
            # Constructs use per-pole grounding lists
            pos_spans = n.get("grounding_spans_positive")
            neg_spans = n.get("grounding_spans_negative")
            if pos_spans is None and neg_spans is None:
                # v3 backward compat: check old field name too
                old_span = n.get("grounding_span")
                if old_span:
                    violations.append(
                        f"[{tid}] {nid}: using deprecated 'grounding_span' on Construct "
                        f"— use 'grounding_spans_positive' / 'grounding_spans_negative'"
                    )
                else:
                    violations.append(
                        f"[{tid}] {nid}: Construct missing grounding_spans_positive "
                        f"and grounding_spans_negative"
                    )
            # bipolarity_complete must be consistent with grounding
            bc = n.get("bipolarity_complete")
            if (
                bc is True
                and pos_spans is not None
                and neg_spans is not None
                and (len(pos_spans) == 0 or len(neg_spans) == 0)
            ):
                violations.append(
                    f"[{tid}] {nid}: bipolarity_complete=true but one pole has "
                    f"empty grounding list "
                    f"(positive={len(pos_spans)} spans, negative={len(neg_spans)} spans)"
                )
            if not n.get("label_negative") and bc is not False:
                n["bipolarity_complete"] = False
                violations.append(
                    f"[{tid}] {nid}: missing negative pole label (label='{n.get('label', '?')}')"
                )
        else:
            # Value, Stance, CSM: use unified grounding_spans list
            spans = n.get("grounding_spans")
            if spans is None:
                old_span = n.get("grounding_span")
                if old_span:
                    violations.append(
                        f"[{tid}] {nid}: using deprecated 'grounding_span' "
                        f"— use 'grounding_spans' (list)"
                    )
                else:
                    violations.append(f"[{tid}] {nid}: {ntype} missing grounding_spans")

        # --- CSM recurrence check (v4: ≥2 spans required) ---
        if ntype == "CognitiveStyleMarker":
            spans = n.get("grounding_spans", [])
            if isinstance(spans, list) and len(spans) < CSM_MIN_SPANS:
                violations.append(
                    f"[{tid}] {nid}: CognitiveStyleMarker has only {len(spans)} "
                    f"grounding span(s); at least {CSM_MIN_SPANS} required "
                    f"(from different [Human] turns)"
                )

        # --- Stance: valence ---
        if ntype == "Stance":
            valence = n.get("valence")
            if valence is None:
                violations.append(f"[{tid}] {nid}: Stance missing valence")
            elif valence not in VALID_VALENCES:
                violations.append(
                    f"[{tid}] {nid}: invalid valence '{valence}' "
                    f"(expected one of {sorted(VALID_VALENCES)})"
                )

    # ── edge checks ─────────────────────────────────────────────────
    for e in edges:
        src = e.get("source", "?")
        tgt = e.get("target", "?")
        rel = e.get("relation", "?")

        # dangling references
        if src not in node_ids:
            violations.append(f"[{tid}] edge source '{src}' not in node IDs")
        if tgt not in node_ids:
            violations.append(f"[{tid}] edge target '{tgt}' not in node IDs")

        # unknown relation type
        if rel not in RELATION_TYPES:
            violations.append(f"[{tid}] {src}→{tgt}: unknown relation '{rel}'")

        # edge rationale required (v4)
        if not e.get("rationale"):
            violations.append(f"[{tid}] {src} --[{rel}]--> {tgt}: missing rationale")

        # edge grounding required (v4, ADR-0004): explicit | inferred
        grounding = e.get("grounding")
        if grounding is None:
            violations.append(f"[{tid}] {src} --[{rel}]--> {tgt}: missing grounding")
        elif grounding not in VALID_GROUNDING:
            violations.append(
                f"[{tid}] {src} --[{rel}]--> {tgt}: invalid grounding '{grounding}' "
                f"(expected one of {sorted(VALID_GROUNDING)})"
            )
        # inferential relations must be grounded as 'inferred' and cite endpoint spans
        if rel in INFERENTIAL_RELATIONS and grounding == "explicit":
            violations.append(
                f"[{tid}] {src} --[{rel}]--> {tgt}: {rel} is inherently inferential "
                f"but marked grounding='explicit'"
            )

        # disallowed Stance → Value direct edge
        if node_type.get(src) == "Stance" and node_type.get(tgt) == "Value":
            violations.append(
                f"[{tid}] direct Stance→Value edge disallowed: {src} --[{rel}]--> {tgt}"
            )

        # relation type / node type consistency (v4 type signatures)
        src_type = node_type.get(src, "")
        tgt_type = node_type.get(tgt, "")
        if src_type and tgt_type:
            if rel == "SERVES" and not (src_type == "Construct" and tgt_type == "Value"):
                violations.append(
                    f"[{tid}] {src} --[SERVES]--> {tgt}: "
                    f"SERVES requires Construct→Value, got {src_type}→{tgt_type}"
                )
            if rel == "EXPRESSED_VIA" and not (src_type == "Stance" and tgt_type == "Construct"):
                violations.append(
                    f"[{tid}] {src} --[EXPRESSED_VIA]--> {tgt}: "
                    f"EXPRESSED_VIA requires Stance→Construct, got {src_type}→{tgt_type}"
                )
            if rel == "MODULATED_BY" and not (
                src_type in ("Construct", "Stance") and tgt_type == "CognitiveStyleMarker"
            ):
                violations.append(
                    f"[{tid}] {src} --[MODULATED_BY]--> {tgt}: "
                    f"MODULATED_BY requires Construct|Stance→CSM, "
                    f"got {src_type}→{tgt_type}"
                )
            if rel == "SUBSUMES" and not (src_type == "Value" and tgt_type == "Value"):
                violations.append(
                    f"[{tid}] {src} --[SUBSUMES]--> {tgt}: "
                    f"SUBSUMES requires Value→Value, got {src_type}→{tgt_type}"
                )
            if rel == "IMPLIES" and not (src_type == "Construct" and tgt_type == "Construct"):
                violations.append(
                    f"[{tid}] {src} --[IMPLIES]--> {tgt}: "
                    f"IMPLIES requires Construct→Construct, got {src_type}→{tgt_type}"
                )

    # ── domain check (v4) ───────────────────────────────────────────
    if not graph.get("domain"):
        violations.append(f"[{tid}] missing 'domain' field")

    return violations


def is_valid(graph: dict[str, Any]) -> bool:
    """Return True if the graph passes all structural checks."""
    return len(validate_graph(graph)) == 0


def validate_selfpos_graph(graph: dict[str, Any]) -> list[str]:
    """Validate a single round-3 (`selfpos` / v5) delegation-boundary graph.

    Enforces constraints S1-S12 from `.claude/context/graph-schema.md`
    (§ ROUND-3) / ADR-0008.  Returns a list of human-readable violation
    strings; an empty list means the graph passes all structural checks.

    Args:
        graph: A dict with ``transcript_id``, ``domain``, ``nodes``, ``edges``.

    Returns:
        List of violation strings (empty = valid).
    """
    violations: list[str] = []

    nodes: list[dict[str, Any]] = graph.get("nodes", [])
    edges: list[dict[str, Any]] = graph.get("edges", [])
    tid: str = graph.get("transcript_id", "unknown")

    node_ids: set[str] = {n["id"] for n in nodes}
    node_type: dict[str, str] = {n["id"]: n.get("type", "") for n in nodes}

    # ── per-node checks (S1, S4, S5) ───────────────────────────────
    for n in nodes:
        nid = n.get("id", "?")
        ntype = n.get("type", "")

        # S1: known node type
        if ntype not in SELFPOS_NODE_TYPES:
            violations.append(
                f"[{tid}] {nid}: unknown node type '{ntype}' "
                f"(expected one of {sorted(SELFPOS_NODE_TYPES)})"
            )
            continue

        # S4: Task fields
        if ntype == "Task":
            if not n.get("label"):
                violations.append(f"[{tid}] {nid}: Task missing label")
            if not n.get("grounding_span"):
                violations.append(f"[{tid}] {nid}: Task missing grounding_span")

        # S5: Value fields
        elif ntype == "Value":
            if not n.get("label"):
                violations.append(f"[{tid}] {nid}: Value missing label")
            if not n.get("grounding_span"):
                violations.append(f"[{tid}] {nid}: Value missing grounding_span")
            vtype = n.get("value_type")
            if vtype is None:
                violations.append(f"[{tid}] {nid}: Value missing value_type")
            elif vtype not in SELFPOS_VALUE_TYPES:
                violations.append(
                    f"[{tid}] {nid}: invalid value_type '{vtype}' "
                    f"(expected one of {sorted(SELFPOS_VALUE_TYPES)})"
                )
        # Self / AI: structural anchors — grounding-exempt, no extra required fields.

    # ── anchor cardinality / id (S2, S3) ───────────────────────────
    self_ids = [n["id"] for n in nodes if n.get("type") == "Self"]
    ai_ids = [n["id"] for n in nodes if n.get("type") == "AI"]

    # S2: exactly one Self, id 'self'
    if len(self_ids) != 1:
        violations.append(f"[{tid}] expected exactly one Self node, found {len(self_ids)}")
    elif self_ids[0] != SELFPOS_SELF_ID:
        violations.append(f"[{tid}] Self node id must be '{SELFPOS_SELF_ID}', got '{self_ids[0]}'")

    # S3: at most one AI, id 'ai'; present iff a ceded/shared edge exists
    has_cede_share = any(e.get("relation") in ("CEDED_TO", "SHARED_WITH") for e in edges)
    if len(ai_ids) > 1:
        violations.append(f"[{tid}] expected at most one AI node, found {len(ai_ids)}")
    elif len(ai_ids) == 1 and ai_ids[0] != SELFPOS_AI_ID:
        violations.append(f"[{tid}] AI node id must be '{SELFPOS_AI_ID}', got '{ai_ids[0]}'")
    if has_cede_share and not ai_ids:
        violations.append(f"[{tid}] CEDED_TO/SHARED_WITH edge present but no AI node")
    if ai_ids and not has_cede_share:
        violations.append(
            f"[{tid}] AI node present but no CEDED_TO/SHARED_WITH edge (orphan anchor)"
        )

    # ── edge checks (S6, S7, S9, S10, S11) ─────────────────────────
    for e in edges:
        src = e.get("source", "?")
        tgt = e.get("target", "?")
        rel = e.get("relation", "?")

        # S9: referential integrity
        if src not in node_ids:
            violations.append(f"[{tid}] edge source '{src}' not in node IDs")
        if tgt not in node_ids:
            violations.append(f"[{tid}] edge target '{tgt}' not in node IDs")

        # S6: known relation
        if rel not in SELFPOS_RELATION_TYPES:
            violations.append(
                f"[{tid}] {src}→{tgt}: unknown relation '{rel}' "
                f"(expected one of {sorted(SELFPOS_RELATION_TYPES)})"
            )

        # S10: grounding span required on every edge
        if not e.get("grounding_span"):
            violations.append(f"[{tid}] {src} --[{rel}]--> {tgt}: missing grounding_span")

        # S7: relation type signatures
        sig = SELFPOS_RELATION_SIGNATURES.get(rel)
        src_type = node_type.get(src, "")
        tgt_type = node_type.get(tgt, "")
        if sig and src_type and tgt_type:
            want_src, want_tgt = sig
            if src_type != want_src or tgt_type != want_tgt:
                violations.append(
                    f"[{tid}] {src} --[{rel}]--> {tgt}: {rel} requires "
                    f"{want_src}→{want_tgt}, got {src_type}→{tgt_type}"
                )

        # S11: rationale_tags placement and membership
        tags = e.get("rationale_tags")
        if rel == "SERVES":
            if tags:
                violations.append(
                    f"[{tid}] {src} --[SERVES]--> {tgt}: SERVES must not carry rationale_tags"
                )
        elif rel in SELFPOS_BOUNDARY_RELATIONS and tags is not None:
            if not isinstance(tags, list):
                violations.append(
                    f"[{tid}] {src} --[{rel}]--> {tgt}: rationale_tags must be a list"
                )
            else:
                bad = [t for t in tags if t not in SELFPOS_RATIONALE_TAGS]
                if bad:
                    violations.append(
                        f"[{tid}] {src} --[{rel}]--> {tgt}: invalid rationale_tags {bad} "
                        f"(allowed: {sorted(SELFPOS_RATIONALE_TAGS)})"
                    )

    # ── partition invariant (S8) ───────────────────────────────────
    boundary_count: dict[str, int] = {n["id"]: 0 for n in nodes if n.get("type") == "Task"}
    for e in edges:
        if e.get("relation") in SELFPOS_BOUNDARY_RELATIONS:
            src = e.get("source", "")
            if src in boundary_count:
                boundary_count[src] += 1
    for task_id, count in boundary_count.items():
        if count != 1:
            violations.append(
                f"[{tid}] {task_id}: Task must have exactly one boundary edge "
                f"(RETAINED_BY|CEDED_TO|SHARED_WITH), found {count}"
            )

    # ── domain (S12) ───────────────────────────────────────────────
    if not graph.get("domain"):
        violations.append(f"[{tid}] missing 'domain' field")

    return violations


def is_valid_selfpos(graph: dict[str, Any]) -> bool:
    """Return True if the graph passes all `selfpos` (round-3) structural checks."""
    return len(validate_selfpos_graph(graph)) == 0


def fix_modulated_by_direction(graph: dict[str, Any]) -> int:
    """Auto-correct MODULATED_BY edges where CSM is the source.

    Some extractions (3/1250 in v4_think) write CSM→Construct|Stance
    instead of the schema-required Construct|Stance→CSM.  The rationale
    text confirms the CSM is the modulator (e.g. "tendency shapes the
    construct"), so the semantics are correct — only the arrow direction
    is flipped.  This function swaps source/target on such edges.

    Returns:
        Number of edges corrected.
    """
    nodes: list[dict[str, Any]] = graph.get("nodes", [])
    edges: list[dict[str, Any]] = graph.get("edges", [])

    node_type: dict[str, str] = {n["id"]: n["type"] for n in nodes}
    fixed = 0

    for e in edges:
        if e.get("relation") != "MODULATED_BY":
            continue
        src = e.get("source", "")
        tgt = e.get("target", "")
        src_type = node_type.get(src, "")
        tgt_type = node_type.get(tgt, "")

        # Detect reversed direction: CSM as source, Construct|Stance as target
        if src_type == "CognitiveStyleMarker" and tgt_type in ("Construct", "Stance"):
            e["source"], e["target"] = tgt, src
            fixed += 1

    return fixed
