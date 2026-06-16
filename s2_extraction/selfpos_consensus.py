"""B7 consensus: merge Agnes + Haiku delegation-boundary labels into a per-transcript consensus,
flag disagreements for adjudication, and report agreement / worklist size.

Transcript-level label = the independent boundary outcome the downstream tests predict:
  - delegation_breadth (ordinal low/med/high)  -- PRIMARY; disagreements -> adjudication worklist
  - rationales_present (multi-label)           -- consensus = intersection; union also kept
  - boundary_talk_depth (1-3)                  -- agree/within-1 auto; gap>=2 -> worklist

    PYTHONPATH=. uv run python s2_extraction/selfpos_consensus.py

Outputs: cache/selfpos_boundary.jsonl, results/method_review/selfpos_consensus/stats.json,
and the adjudication worklist results/method_review/selfpos_consensus/worklist.jsonl.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

CACHE = Path("cache")
TAGGED = Path("s1_data/tagged")
OUT = Path("results/method_review/selfpos_consensus")
BREADTH_ORDER = {"low": 0, "medium": 1, "high": 2}


def load(tag: str) -> dict[str, dict]:
    out = {}
    for line in (CACHE / f"selfpos_{tag}.jsonl").read_text(encoding="utf-8").strip().split("\n"):
        if line:
            d = json.loads(line)
            if d.get("transcript_id") and "_error" not in d:
                out[d["transcript_id"]] = d
    return out


def cohorts() -> dict[str, str]:
    c = {}
    for p in sorted(TAGGED.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                r = json.loads(line)
                c[r["transcript_id"]] = r.get("split", "unknown")
    return c


def rset(d: dict) -> set:
    return {str(x).strip().lower() for x in (d.get("rationales_present") or [])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ag, hk, coh = load("agnes"), load("haiku"), cohorts()
    all_ids = sorted(set(ag) | set(hk))
    both = sorted(set(ag) & set(hk))

    rows, worklist = [], []
    breadth_agree = depth_agree = rset_exact = 0
    for tid in all_ids:
        a, h = ag.get(tid), hk.get(tid)
        cohort = coh.get(tid, "unknown")
        if a is None or h is None:  # single-coder fallback
            src = a or h
            rows.append(
                {
                    "transcript_id": tid,
                    "cohort": cohort,
                    "single_coder": True,
                    "delegation_breadth": src.get("delegation_breadth"),
                    "rationales_consensus": sorted(rset(src)),
                    "rationales_union": sorted(rset(src)),
                    "boundary_talk_depth": src.get("boundary_talk_depth"),
                    "needs_adjudication": False,
                }
            )
            continue

        ba, bh = a.get("delegation_breadth"), h.get("delegation_breadth")
        ra, rh = rset(a), rset(h)
        da, dh = a.get("boundary_talk_depth"), h.get("boundary_talk_depth")

        b_ok = ba == bh
        breadth_agree += b_ok
        rset_exact += ra == rh
        depth_gap = abs((da or 0) - (dh or 0))
        depth_agree += da == dh

        # consensus depth: agree -> value; gap 1 -> lower (conservative); gap>=2 -> None (worklist)
        depth_cons = da if da == dh else (min(da, dh) if depth_gap == 1 else None)
        needs_adj = (not b_ok) or depth_gap >= 2

        row = {
            "transcript_id": tid,
            "cohort": cohort,
            "single_coder": False,
            "delegation_breadth": ba if b_ok else None,
            "breadth_agnes": ba,
            "breadth_haiku": bh,
            "rationales_consensus": sorted(ra & rh),
            "rationales_union": sorted(ra | rh),
            "boundary_talk_depth": depth_cons,
            "depth_agnes": da,
            "depth_haiku": dh,
            "needs_adjudication": needs_adj,
        }
        rows.append(row)
        if needs_adj:
            worklist.append(
                {
                    "transcript_id": tid,
                    "cohort": cohort,
                    "breadth_agnes": ba,
                    "breadth_haiku": bh,
                    "depth_agnes": da,
                    "depth_haiku": dh,
                    "rationales_agnes": sorted(ra),
                    "rationales_haiku": sorted(rh),
                }
            )

    (CACHE / "selfpos_boundary.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    (OUT / "worklist.jsonl").write_text(
        "\n".join(json.dumps(w, ensure_ascii=False) for w in worklist) + "\n"
    )

    n = len(both)
    # breadth disagreement directionality
    adj_dir = Counter()
    for w in worklist:
        if w["breadth_agnes"] != w["breadth_haiku"]:
            lo, hi = sorted(
                [w["breadth_agnes"], w["breadth_haiku"]], key=lambda x: BREADTH_ORDER.get(x, 0)
            )
            adj_dir[f"{lo}<->{hi}"] += 1
    stats = {
        "n_total": len(all_ids),
        "n_dual_coded": n,
        "n_single_coder": len(all_ids) - n,
        "breadth_agreement": round(breadth_agree / n, 3),
        "depth_agreement_exact": round(depth_agree / n, 3),
        "rationales_set_exact_agreement": round(rset_exact / n, 3),
        "worklist_size": len(worklist),
        "worklist_breadth_disputes": dict(adj_dir),
        "consensus_breadth_dist": dict(
            Counter(r["delegation_breadth"] for r in rows if r["delegation_breadth"])
        ),
        "rationale_consensus_prevalence": {
            c: round(sum(1 for r in rows if c in r["rationales_consensus"]) / len(rows), 3)
            for c in [
                "competence",
                "competence_compensate",
                "identity",
                "trust_reliability",
                "output_efficiency",
                "other",
            ]
        },
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
