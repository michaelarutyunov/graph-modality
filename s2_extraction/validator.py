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
RELATION_TYPES = frozenset({"SERVES", "EXPRESSED_VIA", "MODULATED_BY", "CONFLICTS_WITH"})
VALID_VALENCES = frozenset({"positive", "negative", "mixed", "ambivalent"})
MAX_COGNITIVE_STYLE_MARKERS = 2


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
    csm_count = 0

    for n in nodes:
        nid = n.get("id", "?")
        ntype = n.get("type", "")

        # unknown entity type
        if ntype and ntype not in ENTITY_TYPES:
            violations.append(f"[{tid}] {nid}: unknown entity type '{ntype}'")

        # Construct: bipolarity
        if ntype == "Construct" and not n.get("label_negative"):
            n["bipolarity_complete"] = False
            violations.append(
                f"[{tid}] {nid}: missing negative pole (label='{n.get('label', '?')}')"
            )

        # Stance: valence
        if ntype == "Stance":
            valence = n.get("valence")
            if valence is None:
                violations.append(f"[{tid}] {nid}: Stance missing valence")
            elif valence not in VALID_VALENCES:
                violations.append(
                    f"[{tid}] {nid}: invalid valence '{valence}' "
                    f"(expected one of {sorted(VALID_VALENCES)})"
                )

        # CognitiveStyleMarker: count for ceiling check
        if ntype == "CognitiveStyleMarker":
            csm_count += 1

    # ── CSM ceiling ────────────────────────────────────────────────
    if csm_count > MAX_COGNITIVE_STYLE_MARKERS:
        violations.append(
            f"[{tid}] CognitiveStyleMarker count {csm_count} "
            f"exceeds ceiling of {MAX_COGNITIVE_STYLE_MARKERS}"
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

        # disallowed Stance → Value direct edge
        if node_type.get(src) == "Stance" and node_type.get(tgt) == "Value":
            violations.append(
                f"[{tid}] direct Stance→Value edge disallowed: {src} --[{rel}]--> {tgt}"
            )

    return violations


def is_valid(graph: dict[str, Any]) -> bool:
    """Return True if the graph passes all structural checks."""
    return len(validate_graph(graph)) == 0
