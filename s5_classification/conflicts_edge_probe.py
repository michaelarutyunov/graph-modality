"""B1: endpoint-aware CONFLICTS_WITH features vs stance-valence baseline.

Tests whether RELATIONAL conflict structure (CONFLICTS_WITH edges, endpoint-
valence-aware) adds signal over the proven distributional stance-valence
baseline on ``stance_ambivalence``.

Feature set A (baseline): the 9 stance-valence dims at indices 16-24 of
``graph_stats_encoder.graph_to_features`` (4 valence fractions + 5-way
dominant-valence one-hot incl. 'absent').

Feature set B (treatment): A + 4 conflict features:
  1. n_conflict_edges
  2. frac_conflict = n_conflict_edges / max(n_edges, 1)
  3. n_opp_valence_conflicts — CONFLICTS_WITH (Construct->Construct) edges
     where the two constructs have opposite dominant stance valence (per the
     EXPRESSED_VIA majority rule, ignoring mixed/ambivalent; ties/no-stance
     => neither, never counted).
  4. n_distinct_constructs_in_conflict

Probe: sklearn LogisticRegression, class_weight='balanced', mirroring
``ablation_probe.probe_variant``. Evaluation: 10-seed paired CI (seeds 42-51)
on ``stance_ambivalence``, mirroring ``repeated_run``.

Usage:
    uv run python s5_classification/conflicts_edge_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from s4_encoding.build_dataset import _load_ambivalence_labels
from s4_encoding.graph_stats_encoder import graph_to_features
from s5_classification.repeated_eval import make_split

GRAPH_DIR = Path("s1_data/graphs/v4_think/canonical")
OUT_DIR = Path("results/method-review/conflicts_edge_probe_v4")

SEEDS = list(range(42, 52))
LABEL_NAMES = ["low", "med", "high"]  # ordinal 0/1/2 per _load_ambivalence_labels

T_975_DF9 = float(t_dist.ppf(0.975, 9))


def _ci95(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_lo, ci_hi) for a 95% t-CI with df=9."""
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    margin = T_975_DF9 * std / np.sqrt(len(values))
    return mean, std, mean - margin, mean + margin


def _dominant_construct_valences(nodes: list[dict], edges: list[dict]) -> dict[str, str]:
    """Map each Construct node id -> dominant stance valence ('positive'/'negative'/'neither').

    Dominant valence is the majority over {positive, negative} of Stance nodes
    pointing into the construct via ``Stance --EXPRESSED_VIA--> Construct``.
    Mixed/ambivalent stances are ignored. Ties or no positive/negative stances
    => 'neither'.
    """
    stance_valence: dict[str, str] = {}
    for n in nodes:
        if n.get("type") == "Stance":
            stance_valence[n["id"]] = n.get("valence", "")

    construct_ids = {n["id"] for n in nodes if n.get("type") == "Construct"}

    # construct_id -> pos/neg counts from incoming EXPRESSED_VIA stances
    counts: dict[str, dict[str, int]] = {
        cid: {"positive": 0, "negative": 0} for cid in construct_ids
    }
    for e in edges:
        if e.get("relation") != "EXPRESSED_VIA":
            continue
        target = e.get("target")
        source = e.get("source")
        if target is None or target not in construct_ids:
            continue
        valence = stance_valence.get(source, "") if source is not None else ""
        if valence in ("positive", "negative"):
            counts[target][valence] += 1

    result: dict[str, str] = {}
    for cid, c in counts.items():
        pos, neg = c["positive"], c["negative"]
        if pos > neg:
            result[cid] = "positive"
        elif neg > pos:
            result[cid] = "negative"
        else:
            result[cid] = "neither"

    return result


def graph_to_conflict_features(graph_data: dict) -> np.ndarray:
    """Compute the 4 conflict features for a single graph.

    Returns:
        Float32 array of shape (4,): [n_conflict_edges, frac_conflict,
        n_opp_valence_conflicts, n_distinct_constructs_in_conflict].
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    n_edges = len(edges)
    conflict_edges = [e for e in edges if e.get("relation") == "CONFLICTS_WITH"]
    n_conflict_edges = len(conflict_edges)
    frac_conflict = n_conflict_edges / max(n_edges, 1)

    dom_valence = _dominant_construct_valences(nodes, edges)

    n_opp = 0
    distinct_constructs: set[str] = set()
    for e in conflict_edges:
        src, tgt = e.get("source"), e.get("target")
        distinct_constructs.add(src)
        distinct_constructs.add(tgt)
        v_src = dom_valence.get(src, "neither")
        v_tgt = dom_valence.get(tgt, "neither")
        valences = ("positive", "negative")
        if v_src in valences and v_tgt in valences and v_src != v_tgt:
            n_opp += 1

    n_distinct = len(distinct_constructs)

    return np.array(
        [n_conflict_edges, frac_conflict, n_opp, n_distinct],
        dtype=np.float32,
    )


def build_feature_sets(
    graph_dir: Path = GRAPH_DIR,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Build feature sets A (N,9) and B (N,13) for all graphs in ``graph_dir``.

    Returns:
        (feature_a, feature_b, transcript_ids, n_conflict_edges) where
        n_conflict_edges is (N,) for sparsity reporting.
    """
    graph_paths = sorted(graph_dir.glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError(f"no graph files found in {graph_dir}")

    a_rows: list[np.ndarray] = []
    b_extra_rows: list[np.ndarray] = []
    ids: list[str] = []

    for gp in graph_paths:
        g = json.loads(gp.read_text(encoding="utf-8"))
        full_features = graph_to_features(g)
        valence_block = full_features[16:25]  # 9 dims: 16-19 fractions, 20-24 one-hot
        conflict_features = graph_to_conflict_features(g)

        a_rows.append(valence_block)
        b_extra_rows.append(conflict_features)
        ids.append(g["transcript_id"])

    feature_a = np.stack(a_rows).astype(np.float32)
    b_extra = np.stack(b_extra_rows).astype(np.float32)
    feature_b = np.concatenate([feature_a, b_extra], axis=1)
    n_conflict_edges = b_extra[:, 0].astype(np.int64)

    return feature_a, feature_b, ids, n_conflict_edges


def _slice(
    x: np.ndarray,
    lookup: dict[str, int],
    labels_dict: dict[str, int],
    ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    missing = [tid for tid in ids if tid not in lookup]
    if missing:
        raise ValueError(f"Missing ids in feature matrix: {missing}")
    idx = [lookup[tid] for tid in ids]
    x_sliced = x[idx]
    y_sliced = np.array([labels_dict[tid] for tid in ids], dtype=np.int64)
    return x_sliced, y_sliced


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
        "predictions": test_preds.tolist(),
        "labels": y_test.tolist(),
    }


def run_all(out_dir: Path = OUT_DIR) -> dict:
    """Run the 10-seed paired A-vs-B comparison and write results to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_a, feature_b, ids, n_conflict_edges = build_feature_sets()
    lookup = {tid: i for i, tid in enumerate(ids)}
    labels_dict = _load_ambivalence_labels()

    runs_path = out_dir / "runs.jsonl"
    rows: list[dict] = []

    with open(runs_path, "w") as f:
        for seed in SEEDS:
            print(f"[seed={seed}] probing A and B")
            result_a = probe(feature_a, lookup, labels_dict, seed)
            result_b = probe(feature_b, lookup, labels_dict, seed)

            row = {
                "seed": seed,
                "a": {
                    "test_macro_f1": result_a["test_macro_f1"],
                    "per_class_f1": result_a["per_class_f1"],
                },
                "b": {
                    "test_macro_f1": result_b["test_macro_f1"],
                    "per_class_f1": result_b["per_class_f1"],
                },
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    # n_conflict_edges sparsity distribution (over the full corpus, label-independent).
    n_conflict_total = int(n_conflict_edges.sum())
    n_graphs = len(ids)
    n_graphs_with_conflict = int((n_conflict_edges > 0).sum())
    conflict_dist = {
        str(k): int(v)
        for k, v in zip(*np.unique(n_conflict_edges, return_counts=True), strict=False)
    }

    a_f1s = np.array([r["a"]["test_macro_f1"] for r in rows])
    b_f1s = np.array([r["b"]["test_macro_f1"] for r in rows])
    delta = b_f1s - a_f1s

    mean_a, std_a, ci_lo_a, ci_hi_a = _ci95(a_f1s)
    mean_b, std_b, ci_lo_b, ci_hi_b = _ci95(b_f1s)
    mean_delta, std_delta, ci_lo_delta, ci_hi_delta = _ci95(delta)

    real_effect = bool((ci_lo_delta > 0 or ci_hi_delta < 0) and mean_delta >= 0.01)

    # Per-class F1 averaged across seeds.
    per_class_a = {
        label: float(np.mean([r["a"]["per_class_f1"][label] for r in rows]))
        for label in LABEL_NAMES
    }
    per_class_b = {
        label: float(np.mean([r["b"]["per_class_f1"][label] for r in rows]))
        for label in LABEL_NAMES
    }

    summary = {
        "seeds": SEEDS,
        "n_graphs": n_graphs,
        "feature_a_dim": int(feature_a.shape[1]),
        "feature_b_dim": int(feature_b.shape[1]),
        "n_conflict_edges_total": n_conflict_total,
        "n_graphs_with_conflict": n_graphs_with_conflict,
        "n_conflict_edges_distribution": conflict_dist,
        "feature_a": {
            "mean_test_macro_f1": mean_a,
            "std_test_macro_f1": std_a,
            "ci95_test_macro_f1": [ci_lo_a, ci_hi_a],
            "mean_per_class_f1": per_class_a,
        },
        "feature_b": {
            "mean_test_macro_f1": mean_b,
            "std_test_macro_f1": std_b,
            "ci95_test_macro_f1": [ci_lo_b, ci_hi_b],
            "mean_per_class_f1": per_class_b,
        },
        "delta_b_minus_a": {
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "ci95_delta": [ci_lo_delta, ci_hi_delta],
            "real_effect": real_effect,
        },
        "verdict": "PASS" if real_effect and mean_delta >= 0.01 else "FAIL",
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = run_all()
    print(json.dumps(summary, indent=2))
    print(f"\nVerdict: {summary['verdict']}")
    print(f"Mean delta (B - A): {summary['delta_b_minus_a']['mean_delta']:.4f}")
    print(f"95% CI: {summary['delta_b_minus_a']['ci95_delta']}")
    print(
        f"\nConflict-edge sparsity: {summary['n_conflict_edges_total']} total edges "
        f"across {summary['n_graphs']} graphs "
        f"({summary['n_graphs_with_conflict']} graphs with >=1 conflict edge)"
    )
