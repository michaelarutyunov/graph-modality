"""Merge Agnes + Haiku stance_ambivalence labels into a consensus dataset.

Agreements are accepted directly. Disagreements are written to a worklist
for manual adjudication; only adjudicated disagreements are included in the
final consensus output. No pre-committed gold set.

Usage:
    PYTHONPATH=. uv run python s2_extraction/ambivalence_consensus.py --report
    # ... edit cache/ambivalence_adjudications.json ...
    PYTHONPATH=. uv run python s2_extraction/ambivalence_consensus.py --finalize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

CACHE_DIR = Path("cache")
PATH_A = CACHE_DIR / "ambivalence_agnes.jsonl"
PATH_B = CACHE_DIR / "ambivalence_haiku.jsonl"
DISAGREE_PATH = CACHE_DIR / "ambivalence_disagreements.json"
ADJUDICATE_PATH = CACHE_DIR / "ambivalence_adjudications.json"
FINAL_PATH = CACHE_DIR / "ambivalence.jsonl"

NON_ORDINAL = {"uncertain", "error"}


def _label(entry: dict) -> str:
    return entry["stance_ambivalence"]["label"]


def _load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        out[entry["transcript_id"]] = entry
    return out


def compute_agreement(a: dict[str, dict], b: dict[str, dict]) -> dict:
    common = sorted(set(a) & set(b))
    labels_a = [_label(a[tid]) for tid in common]
    labels_b = [_label(b[tid]) for tid in common]
    n_agree = sum(1 for la, lb in zip(labels_a, labels_b, strict=True) if la == lb)
    disagreements = [
        tid for tid, la, lb in zip(common, labels_a, labels_b, strict=True) if la != lb
    ]
    kappa = cohen_kappa_score(labels_a, labels_b) if common else float("nan")
    return {
        "n_common": len(common),
        "n_agree": n_agree,
        "agreement_rate": n_agree / len(common) if common else 0.0,
        "kappa": kappa,
        "disagreements": disagreements,
    }


def merge_consensus(
    a: dict[str, dict], b: dict[str, dict], adjudications: dict[str, str]
) -> dict[str, dict]:
    final: dict[str, dict] = {}
    for tid in sorted(set(a) & set(b)):
        la, lb = _label(a[tid]), _label(b[tid])
        if la == lb:
            final[tid] = {
                "transcript_id": tid,
                "stance_ambivalence": {"label": la, "source": "consensus"},
            }
        elif tid in adjudications:
            final[tid] = {
                "transcript_id": tid,
                "stance_ambivalence": {"label": adjudications[tid], "source": "adjudicated"},
            }
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Ambivalence consensus / adjudication")
    parser.add_argument(
        "--report", action="store_true", help="Compute agreement + write disagreement worklist"
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Merge consensus + adjudications into ambivalence.jsonl",
    )
    args = parser.parse_args()

    a = _load_jsonl(PATH_A)
    b = _load_jsonl(PATH_B)

    if args.report:
        stats = compute_agreement(a, b)
        print(
            f"n_common={stats['n_common']} n_agree={stats['n_agree']} "
            f"agreement_rate={stats['agreement_rate']:.3f} kappa={stats['kappa']:.3f}"
        )
        worklist = [
            {
                "transcript_id": tid,
                "agnes": _label(a[tid]),
                "haiku": _label(b[tid]),
            }
            for tid in stats["disagreements"]
        ]
        DISAGREE_PATH.write_text(json.dumps(worklist, indent=2), encoding="utf-8")
        print(f"Wrote {len(worklist)} disagreements to {DISAGREE_PATH}")

    if args.finalize:
        adjudications: dict[str, str] = {}
        if ADJUDICATE_PATH.exists():
            adjudications = json.loads(ADJUDICATE_PATH.read_text(encoding="utf-8"))
        final = merge_consensus(a, b, adjudications)
        with open(FINAL_PATH, "w", encoding="utf-8") as f:
            for tid in sorted(final):
                json.dump(final[tid], f, ensure_ascii=False)
                f.write("\n")
        print(f"Wrote {len(final)} consensus labels to {FINAL_PATH}")


if __name__ == "__main__":
    main()
