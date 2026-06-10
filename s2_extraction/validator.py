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
