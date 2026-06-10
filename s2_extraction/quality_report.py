"""Quality checkpoint for v4 extraction runs.

Computes the metrics that decide whether to continue or abort a long extraction
(see bead graph-modality-vbe). Run against the live output dir at each checkpoint:

    uv run python s2_extraction/quality_report.py                       # default v4_think
    uv run python s2_extraction/quality_report.py --graph-dir s1_data/graphs/v4/free_text

Abort signals to watch across checkpoints:
- violation rate rising
- schema-filling share rising toward 1.0 (graphs with EXACTLY 1 SUBSUMES AND 1
  IMPLIES — the deepseek-chat rote pattern; the reasoner should stay well below)
- inferential-edge variance collapsing (uniform counts carry no signal)
- edge-count vs transcript-length correlation high (size confound, not cognition)
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from s2_extraction.extractor import load_tagged_transcripts

INFERENTIAL = ("SUBSUMES", "IMPLIES")


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def report(graph_dir: Path) -> None:
    tagged = load_tagged_transcripts()
    paths = sorted(graph_dir.glob("*.json"))
    if not paths:
        print(f"No graphs in {graph_dir}")
        return

    n = len(paths)
    by_cohort: Counter[str] = Counter()
    nodes, edges, infl_counts = [], [], []
    n_with_violation = 0
    total_violations = 0
    schema_fill = 0  # exactly 1 SUBSUMES AND 1 IMPLIES
    rel_dist: Counter[str] = Counter()
    grounding: Counter[str] = Counter()
    infl_grounding: Counter[str] = Counter()
    edge_len_pairs: list[tuple[int, int]] = []  # (edge_count, n_human_turns)
    per_cohort_infl: dict[str, list[int]] = {}

    for p in paths:
        g = json.loads(p.read_text())
        cohort = g.get("split", "?")
        by_cohort[cohort] += 1
        nn, ne = len(g.get("nodes", [])), len(g.get("edges", []))
        nodes.append(nn)
        edges.append(ne)
        rels = Counter(e.get("relation") for e in g.get("edges", []))
        rel_dist.update(rels)
        ic = rels.get("SUBSUMES", 0) + rels.get("IMPLIES", 0)
        infl_counts.append(ic)
        per_cohort_infl.setdefault(cohort, []).append(ic)
        if rels.get("SUBSUMES", 0) == 1 and rels.get("IMPLIES", 0) == 1:
            schema_fill += 1
        for e in g.get("edges", []):
            grounding[e.get("grounding", "MISSING")] += 1
            if e.get("relation") in INFERENTIAL:
                infl_grounding[e.get("grounding", "MISSING")] += 1
        v = g.get("validation_violations", [])
        if v:
            n_with_violation += 1
            total_violations += len(v)
        tid = g.get("transcript_id", "")
        if tid in tagged:
            edge_len_pairs.append((ne, tagged[tid].get("n_human_turns", 0)))

    def ms(xs: list[float]) -> str:
        return f"mean={statistics.mean(xs):.2f} sd={statistics.pstdev(xs):.2f}"

    print(f"\n{'=' * 60}\nQUALITY CHECKPOINT — {graph_dir}  (n={n})\n{'=' * 60}")
    print(f"cohorts: {dict(by_cohort)}")
    print(f"nodes:   {ms(nodes)}   edges: {ms(edges)}")
    print(
        f"violations: {n_with_violation}/{n} graphs ({n_with_violation / n:.1%}), "
        f"{total_violations} total"
    )
    print("\n-- inferential edges (SUBSUMES+IMPLIES) --")
    print(
        f"  per graph: {ms([float(x) for x in infl_counts])}  "
        f"zero={infl_counts.count(0)}/{n} ({infl_counts.count(0) / n:.1%})"
    )
    print(f"  count distribution: {dict(sorted(Counter(infl_counts).items()))}")
    print(
        f"  ** schema-filling (exactly 1+1): {schema_fill}/{n} ({schema_fill / n:.1%}) "
        f"— rising toward 100% = degrading to rote **"
    )
    for c, lst in sorted(per_cohort_infl.items()):
        print(f"    {c:11s}: {ms([float(x) for x in lst])}")
    print("\n-- relations --")
    tot = sum(rel_dist.values()) or 1
    for r, c in rel_dist.most_common():
        print(f"  {r:16s} {c:4d} ({c / tot:.1%})")
    print("\n-- grounding --")
    print(f"  all edges: {dict(grounding)}")
    print(f"  SUBSUMES/IMPLIES: {dict(infl_grounding)} (must be all 'inferred')")
    print("\n-- size confound --")
    if len(edge_len_pairs) >= 2:
        r = _pearson([float(e) for e, _ in edge_len_pairs], [float(t) for _, t in edge_len_pairs])
        flag = "  <-- HIGH: graph size tracks transcript length" if abs(r) > 0.5 else ""
        print(f"  corr(edge_count, n_human_turns) = {r:.3f}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="v4 extraction quality checkpoint")
    ap.add_argument("--graph-dir", type=str, default="s1_data/graphs/v4_think/free_text")
    args = ap.parse_args()
    report(Path(args.graph_dir))


if __name__ == "__main__":
    main()
