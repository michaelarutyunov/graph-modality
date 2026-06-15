"""B1 corrective: do conflict/relational features add anything over FULL stats30?

B1 (``conflicts_edge_probe.py``) showed the 4 conflict features beat a thin
9-dim stance-valence baseline (Δ=+0.042). But that 9-dim baseline is a
sub-vector of the full 30-dim ``graph_stats_encoder.graph_to_features``
representation, which alone scores 0.433 macro-F1 (Phase 6, v4) — beating the
conflict-augmented 9+4=13-dim model (0.399). So the conflict features may only
have "won" because they partially recovered information already present in
the other 21 stats dims.

This module re-runs the comparison against the FULL 30-dim stats baseline:

  (1) PRIMARY:               stats30 + conflict(all 4)        vs stats30
  (2) DECOMPOSITION-circular: stats30 + n_conflict_edges       vs stats30
  (3) DECOMPOSITION-relational: stats30 + n_opp_valence_conflicts vs stats30
  (4) RELATIONAL-net-of-count: stats30 + n_opp_valence_conflicts
                                  vs stats30 + n_conflict_edges

Feature builders are reused unmodified from ``conflicts_edge_probe``.

Usage:
    uv run python s5_classification/conflicts_edge_corrective.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from s4_encoding.build_dataset import _load_ambivalence_labels
from s4_encoding.graph_stats_encoder import graph_to_features
from s5_classification.conflicts_edge_probe import (
    GRAPH_DIR,
    LABEL_NAMES,
    SEEDS,
    _ci95,
    _slice,
    graph_to_conflict_features,
)
from s5_classification.repeated_eval import make_split

OUT_DIR = Path("results/method-review/conflicts_edge_corrective_v4")

# Names of the 4 conflict feature columns, matching graph_to_conflict_features order.
CONFLICT_FEATURE_NAMES = [
    "n_conflict_edges",
    "frac_conflict",
    "n_opp_valence_conflicts",
    "n_distinct_constructs_in_conflict",
]


def build_feature_matrices(
    graph_dir: Path = GRAPH_DIR,
) -> dict[str, object]:
    """Build stats30 and the 4 conflict feature columns for all graphs.

    Returns:
        dict with keys "stats30" (N,30), "conflict" (N,4), "ids" (list[str]).
    """
    graph_paths = sorted(graph_dir.glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError(f"no graph files found in {graph_dir}")

    stats_rows: list[np.ndarray] = []
    conflict_rows: list[np.ndarray] = []
    ids: list[str] = []

    for gp in graph_paths:
        g = json.loads(gp.read_text(encoding="utf-8"))
        stats_rows.append(graph_to_features(g))
        conflict_rows.append(graph_to_conflict_features(g))
        ids.append(g["transcript_id"])

    return {
        "stats30": np.stack(stats_rows).astype(np.float32),
        "conflict": np.stack(conflict_rows).astype(np.float32),
        "ids": ids,
    }


def probe(
    x: np.ndarray,
    lookup: dict[str, int],
    labels_dict: dict[str, int],
    seed: int,
) -> dict:
    """Fit a logistic-regression probe for one feature matrix/seed."""
    train_ids, _val_ids, test_ids = make_split("stance_ambivalence", seed)

    x_train, y_train = _slice(x, lookup, labels_dict, train_ids)
    x_test, y_test = _slice(x, lookup, labels_dict, test_ids)

    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(x_train, y_train)

    test_preds = np.asarray(model.predict(x_test))
    test_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))  # type: ignore[arg-type]
    per_class_f1 = np.asarray(
        f1_score(y_test, test_preds, average=None, zero_division=0, labels=[0, 1, 2])  # type: ignore[arg-type,call-overload]
    )

    return {
        "test_macro_f1": test_f1,
        "per_class_f1": {LABEL_NAMES[i]: float(per_class_f1[i]) for i in range(3)},
    }


def _summarize_pair(
    name_a: str,
    name_b: str,
    f1s_a: np.ndarray,
    f1s_b: np.ndarray,
    per_class_a: list[dict],
    per_class_b: list[dict],
) -> dict:
    """Build a contrast summary dict for variant b - variant a (paired, per-seed)."""
    delta = f1s_b - f1s_a
    mean_a, std_a, ci_lo_a, ci_hi_a = _ci95(f1s_a)
    mean_b, std_b, ci_lo_b, ci_hi_b = _ci95(f1s_b)
    mean_delta, std_delta, ci_lo_delta, ci_hi_delta = _ci95(delta)

    pass_ = bool((ci_lo_delta > 0 or ci_hi_delta < 0) and mean_delta >= 0.01)

    pc_a = {label: float(np.mean([r[label] for r in per_class_a])) for label in LABEL_NAMES}
    pc_b = {label: float(np.mean([r[label] for r in per_class_b])) for label in LABEL_NAMES}

    return {
        "contrast": f"{name_b} - {name_a}",
        name_a: {
            "mean_test_macro_f1": mean_a,
            "std_test_macro_f1": std_a,
            "ci95_test_macro_f1": [ci_lo_a, ci_hi_a],
            "mean_per_class_f1": pc_a,
        },
        name_b: {
            "mean_test_macro_f1": mean_b,
            "std_test_macro_f1": std_b,
            "ci95_test_macro_f1": [ci_lo_b, ci_hi_b],
            "mean_per_class_f1": pc_b,
        },
        "delta": {
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "ci95_delta": [ci_lo_delta, ci_hi_delta],
        },
        "verdict": "PASS" if pass_ else "FAIL",
    }


def run_corrective(out_dir: Path = OUT_DIR) -> dict:
    """Run the corrective contrasts and write results to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    mats = build_feature_matrices()
    stats30 = mats["stats30"]
    conflict = mats["conflict"]
    ids = mats["ids"]
    assert isinstance(stats30, np.ndarray)
    assert isinstance(conflict, np.ndarray)
    assert isinstance(ids, list)

    labels_dict = _load_ambivalence_labels()
    # Align on transcript_id intersection with labels.
    common_ids = [tid for tid in ids if tid in labels_dict]
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    keep_idx = [id_to_idx[tid] for tid in common_ids]

    stats30 = stats30[keep_idx]
    conflict = conflict[keep_idx]
    ids = common_ids
    lookup = {tid: i for i, tid in enumerate(ids)}

    n_opp_idx = CONFLICT_FEATURE_NAMES.index("n_opp_valence_conflicts")  # 2
    n_count_idx = CONFLICT_FEATURE_NAMES.index("n_conflict_edges")  # 0

    # Variant feature matrices.
    variants = {
        "stats30": stats30,
        "stats30_plus_conflict4": np.concatenate([stats30, conflict], axis=1),
        "stats30_plus_count": np.concatenate([stats30, conflict[:, [n_count_idx]]], axis=1),
        "stats30_plus_oppval": np.concatenate([stats30, conflict[:, [n_opp_idx]]], axis=1),
    }

    runs_path = out_dir / "runs.jsonl"
    per_variant_f1s: dict[str, list[float]] = {k: [] for k in variants}
    per_variant_pc: dict[str, list[dict]] = {k: [] for k in variants}

    with open(runs_path, "w") as f:
        for seed in SEEDS:
            print(f"[seed={seed}] probing {list(variants)}")
            row: dict = {"seed": seed}
            for name, x in variants.items():
                result = probe(x, lookup, labels_dict, seed)
                per_variant_f1s[name].append(result["test_macro_f1"])
                per_variant_pc[name].append(result["per_class_f1"])
                row[name] = {
                    "test_macro_f1": result["test_macro_f1"],
                    "per_class_f1": result["per_class_f1"],
                }
            f.write(json.dumps(row) + "\n")

    f1_arrays = {k: np.array(v) for k, v in per_variant_f1s.items()}

    contrasts = {
        "primary_conflict4_vs_stats30": _summarize_pair(
            "stats30",
            "stats30_plus_conflict4",
            f1_arrays["stats30"],
            f1_arrays["stats30_plus_conflict4"],
            per_variant_pc["stats30"],
            per_variant_pc["stats30_plus_conflict4"],
        ),
        "decomposition_circular_count_vs_stats30": _summarize_pair(
            "stats30",
            "stats30_plus_count",
            f1_arrays["stats30"],
            f1_arrays["stats30_plus_count"],
            per_variant_pc["stats30"],
            per_variant_pc["stats30_plus_count"],
        ),
        "decomposition_relational_oppval_vs_stats30": _summarize_pair(
            "stats30",
            "stats30_plus_oppval",
            f1_arrays["stats30"],
            f1_arrays["stats30_plus_oppval"],
            per_variant_pc["stats30"],
            per_variant_pc["stats30_plus_oppval"],
        ),
        "relational_net_of_count_oppval_vs_count": _summarize_pair(
            "stats30_plus_count",
            "stats30_plus_oppval",
            f1_arrays["stats30_plus_count"],
            f1_arrays["stats30_plus_oppval"],
            per_variant_pc["stats30_plus_count"],
            per_variant_pc["stats30_plus_oppval"],
        ),
    }

    abs_f1 = {
        name: {
            "mean_test_macro_f1": float(np.mean(arr)),
            "std_test_macro_f1": float(np.std(arr, ddof=1)),
        }
        for name, arr in f1_arrays.items()
    }

    summary = {
        "seeds": SEEDS,
        "n_graphs": len(ids),
        "feature_dims": {name: int(x.shape[1]) for name, x in variants.items()},
        "absolute_macro_f1": abs_f1,
        "contrasts": contrasts,
        "reference": {
            "stats30_phase6_v4": 0.433,
            "label_bag_phase6_v4": 0.402,
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = run_corrective()
    print(json.dumps(summary, indent=2))
