"""Gate-1 analysis (bead graph-modality-nu2): is the delegation boundary codable, varying,
cross-cohort, and not protocol-confounded?

Reads the two dual-coder label files (cache/selfpos_agnes.jsonl, cache/selfpos_haiku.jsonl) and
the cohort from s1_data/tagged (the 'split' field), and evaluates the four pinned PASS criteria.

    PYTHONPATH=. uv run python s2_extraction/selfpos_gate1.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import chi2_contingency

TAGGED = Path("s1_data/tagged")
CACHE = Path("cache")
OUT = Path("results/method_review/selfpos_gate1")
CORE = ["competence", "identity", "trust_reliability"]


def load_labels(tag: str) -> dict[str, dict]:
    out = {}
    p = CACHE / f"selfpos_{tag}.jsonl"
    for line in p.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        d = json.loads(line)
        if d.get("transcript_id") and "_error" not in d:
            out[d["transcript_id"]] = d
    return out


def cohorts() -> dict[str, str]:
    c = {}
    for path in sorted(TAGGED.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                r = json.loads(line)
                c[r["transcript_id"]] = r.get("split", "unknown")
    return c


def rset(d: dict) -> set:
    rp = d.get("rationales_present") or []
    return {str(x).strip().lower() for x in rp}


def cramers_v(labels_a: list, labels_b: list) -> float:
    cats_a = sorted(set(labels_a))
    cats_b = sorted(set(labels_b))
    table = [
        [sum(1 for x, y in zip(labels_a, labels_b) if x == ca and y == cb) for cb in cats_b]
        for ca in cats_a
    ]
    chi2 = chi2_contingency(table)[0]
    n = len(labels_a)
    k = min(len(cats_a), len(cats_b))
    return round(math.sqrt(chi2 / (n * (k - 1))), 3) if k > 1 and n else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ag, hk = load_labels("agnes"), load_labels("haiku")
    coh = cohorts()
    ids = sorted(set(ag) & set(hk))
    cohort_list = ["workforce", "creatives", "scientists"]
    print(f"dual-coded transcripts: {len(ids)} (agnes={len(ag)} haiku={len(hk)})")

    # --- (3) codability: prevalence-robust inter-coder agreement. Gwet AC1 is primary;
    #     Cohen kappa is reported but unreliable at the observed ~80% prevalence (kappa paradox).
    agree = {}
    for cat in CORE:
        a = [int(cat in rset(ag[t])) for t in ids]
        b = [int(cat in rset(hk[t])) for t in ids]
        n = len(ids)
        po = sum(x == y for x, y in zip(a, b)) / n
        pa, pb = sum(a) / n, sum(b) / n
        pe_k = pa * pb + (1 - pa) * (1 - pb)
        cohen = (po - pe_k) / (1 - pe_k) if pe_k < 1 else 1.0
        q = (pa + pb) / 2
        pe_g = 2 * q * (1 - q)
        ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else 1.0
        agree[cat] = {
            "obs_agreement": round(po, 3),
            "cohen_kappa": round(cohen, 3),
            "gwet_ac1": round(ac1, 3),
            "pabak": round(2 * po - 1, 3),
        }
    mean_core_ac1 = round(sum(agree[c]["gwet_ac1"] for c in CORE) / len(CORE), 3)
    codable = agree["competence"]["gwet_ac1"] >= 0.40

    # inter-coder core Jaccard
    def jac(a, b):
        a, b = a & set(CORE), b & set(CORE)
        return 1.0 if not a and not b else len(a & b) / len(a | b) if (a | b) else 1.0

    inter_jac = round(sum(jac(rset(ag[t]), rset(hk[t])) for t in ids) / len(ids), 3)

    # --- (1) boundary varies ---
    def breadth_dist(src):
        from collections import Counter

        c = Counter(src[t].get("delegation_breadth") for t in ids)
        return {k: round(v / len(ids), 3) for k, v in c.items()}

    bd_ag, bd_hk = breadth_dist(ag), breadth_dist(hk)
    breadth_varies = (
        sum(v >= 0.10 for v in bd_ag.values()) >= 2 and sum(v >= 0.10 for v in bd_hk.values()) >= 2
    )
    # rationale variety: union-present prevalence, >=3 rationales each >=5%
    allcats = [
        "competence",
        "competence_compensate",
        "identity",
        "trust_reliability",
        "output_efficiency",
        "other",
    ]
    union_prev = {
        c: round(sum(1 for t in ids if c in (rset(ag[t]) | rset(hk[t]))) / len(ids), 3)
        for c in allcats
    }
    rationale_varies = sum(v >= 0.05 for v in union_prev.values()) >= 3
    varies = breadth_varies and rationale_varies

    # --- (2) competence cross-cohort (consensus: both coders mark competence present) ---
    comp_by_cohort = {}
    for ch in cohort_list:
        cids = [t for t in ids if coh.get(t) == ch]
        if not cids:
            comp_by_cohort[ch] = None
            continue
        consensus = sum(
            1 for t in cids if "competence" in rset(ag[t]) and "competence" in rset(hk[t])
        )
        comp_by_cohort[ch] = {
            "n": len(cids),
            "consensus_competence_frac": round(consensus / len(cids), 3),
            "either_frac": round(
                sum(1 for t in cids if "competence" in (rset(ag[t]) | rset(hk[t]))) / len(cids), 3
            ),
        }
    competence_cross_cohort = all(
        comp_by_cohort[ch] and comp_by_cohort[ch]["consensus_competence_frac"] >= 0.15
        for ch in cohort_list
    )

    # --- (4) depth x cohort confound (Cramer V per coder) ---
    depth_v = {}
    for tag, src in (("agnes", ag), ("haiku", hk)):
        cohs = [coh.get(t) for t in ids]
        depths = [str(src[t].get("boundary_talk_depth")) for t in ids]
        depth_v[tag] = cramers_v(cohs, depths)
    depth_not_confounded = max(depth_v.values()) < 0.30

    overall = varies and competence_cross_cohort and codable and depth_not_confounded

    summary = {
        "n_dual_coded": len(ids),
        "criteria": {
            "1_boundary_varies": {
                "pass": varies,
                "breadth_agnes": bd_ag,
                "breadth_haiku": bd_hk,
                "union_rationale_prevalence": union_prev,
            },
            "2_competence_cross_cohort": {
                "pass": competence_cross_cohort,
                "by_cohort": comp_by_cohort,
                "threshold": 0.15,
            },
            "3_codable_inter_coder": {
                "pass": codable,
                "statistic": "Gwet AC1 primary (Cohen kappa unreliable at high prevalence)",
                "per_core_category": agree,
                "mean_core_ac1": mean_core_ac1,
                "inter_core_jaccard": inter_jac,
                "primary": "competence",
                "threshold": 0.40,
            },
            "4_depth_not_confounded": {
                "pass": depth_not_confounded,
                "cramers_v_depth_x_cohort": depth_v,
                "threshold": 0.30,
            },
        },
        "OVERALL_VERDICT": "GO" if overall else "NO-GO",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
