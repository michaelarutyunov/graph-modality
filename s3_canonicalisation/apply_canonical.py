"""Apply the locked canonical map to all free-text graphs.

Replaces every node label with its canonical equivalent and writes the
canonicalised graphs to ``s1_data/graphs/canonical/``.  Re-validates each
graph after label replacement.

Usage:
    uv run python canonicalisation/apply_canonical.py
"""

from __future__ import annotations

import json
from pathlib import Path

from s2_extraction.validator import validate_graph

FREE_TEXT_DIR = Path("s1_data/graphs/free_text")
CANONICAL_DIR = Path("s1_data/graphs/canonical")
MAP_PATH = Path("s3_canonicalisation/canonical_map.json")

# Labels not found in the canonical map are kept as-is with a warning.
# This should happen only for labels extracted after the map was locked.


def load_canonical_map() -> dict[str, dict[str, str]]:
    """Load and validate the locked canonical map."""
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"canonical map not found: {MAP_PATH}")

    cm = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    meta = cm.get("_meta", {})
    if not meta.get("locked"):
        raise RuntimeError("canonical map is not locked — run clusterer.py and manual review first")
    print(
        f"Loaded locked canonical map ({meta.get('total_labels', '?')} labels, "
        f"locked at {meta.get('locked_at', '?')})"
    )
    return {k: v for k, v in cm.items() if k != "_meta"}


def canonicalise_graph(graph: dict, canonical_map: dict[str, dict[str, str]]) -> dict:
    """Replace free-text labels with canonical equivalents in one graph.

    Returns a new dict — does not mutate the input.
    """
    import copy

    g = copy.deepcopy(graph)

    unmapped = 0
    for node in g.get("nodes", []):
        ntype = node.get("type", "")
        label = node.get("label", "")
        if ntype in canonical_map and label in canonical_map[ntype]:
            node["label"] = canonical_map[ntype][label]
        elif label:
            unmapped += 1

    if unmapped:
        g.setdefault("_unmapped_labels", 0)
        g["_unmapped_labels"] = unmapped

    return g


def apply_all(
    free_text_dir: Path = FREE_TEXT_DIR,
    canonical_dir: Path = CANONICAL_DIR,
) -> tuple[int, int, int]:
    """Canonicalise all free-text graphs.

    Returns:
        (total, canonicalised, unmapped_labels).
    """
    canonical_map = load_canonical_map()
    canonical_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(free_text_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no graph files found in {free_text_dir}")

    total = len(paths)
    total_unmapped = 0
    total_violations = 0

    for i, path in enumerate(paths):
        g = json.loads(path.read_text(encoding="utf-8"))
        cg = canonicalise_graph(g, canonical_map)

        # Re-validate
        violations = validate_graph(cg)
        cg["validation_violations"] = violations
        if violations:
            total_violations += 1

        total_unmapped += cg.get("_unmapped_labels", 0)
        cg.pop("_unmapped_labels", None)

        out_path = canonical_dir / path.name
        out_path.write_text(json.dumps(cg, indent=2, ensure_ascii=False), encoding="utf-8")

        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{total}...")

    return total, total - total_violations, total_unmapped


def main() -> None:
    """Apply canonical labels to all free-text graphs and write to s1_data/graphs/canonical/."""
    print(f"Canonicalising graphs from {FREE_TEXT_DIR} → {CANONICAL_DIR}")
    total, clean, unmapped = apply_all()

    print("\nDone.")
    print(f"  Graphs processed:    {total}")
    print(f"  Clean (0 violations): {clean}")
    print(f"  With violations:      {total - clean}")
    print(f"  Unmapped labels:      {unmapped}")
    print(f"  Output:               {CANONICAL_DIR}/")


if __name__ == "__main__":
    main()
