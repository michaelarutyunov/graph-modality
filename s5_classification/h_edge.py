"""H_edge — edge-axis ablation: does edge information add signal? (P6.4)

The decisive, confound-free edge test for ``stance_ambivalence``. All three rungs
use **structure-only** node features (type one-hot + degree, 5-d) on the same
**v4_think** graph corpus (the only corpus with the v4 edge ontology: SUBSUMES /
IMPLIES added to break v3 edge-type determinism, per ADR-0004). Because the node
features carry no label semantics, a win here cannot be "pooled label text in
disguise" (METHOD_REVIEW concerns #2/#4).

Edge axis (3 rungs, capacity-matched via a fixed-capacity logistic probe):
  - no_edges  : 11-dim bag-of-types histogram (node + edge-type frequencies +
                mean degree). Captures type DISTRIBUTION, ZERO wiring.
  - untyped   : GINConv autoencoder → 128-d. Message passing over the adjacency
                (which nodes connect) but edge-type-AGNOSTIC.
  - typed     : GINEConv autoencoder → 128-d. Same, but consumes the 6-dim
                relation-type one-hot per edge.

Primary criterion (the H_edge hypothesis):
    typed - untyped : does edge TYPE add over untyped wiring? (PASS = CI excludes
    0 AND mean delta >= +0.01)
Secondary:
    untyped - histogram : does wiring of any kind add over the distributional null?

Encoders are self-supervised (node-type reconstruction) and target-agnostic, so
they train ONCE; only the class-weighted logistic probe + stratified split vary
across the 10 seeds (42-51).

Usage:
    PYTHONPATH=. uv run python s5_classification/h_edge.py --target stance_ambivalence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from s4_encoding.graph_gnn_encoder import V4_FREE_TEXT_DIR, encode_graphs_gine
from s5_classification.null_ladder import _class_names, build_histogram_features
from s5_classification.repeated_eval import make_split
from s5_classification.structure_only_probe import (
    _encode,
    _load_graphs,
    _load_labels,
    _train_gin,
    majority_class_macro_f1,
)

SEEDS = list(range(42, 52))
ARMS = ["no_edges", "untyped", "typed"]
OUT_DIR = Path("cache/h_edge")


def _weighted_logreg_f1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Fixed-capacity, class-weighted logistic probe. Returns (macro_f1, preds)."""
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return float(f1_score(y_test, preds, average="macro")), preds


def _ci(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% t-CI (df=n-1)."""
    m = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    lo, hi = stats.t.interval(0.95, df=len(values) - 1, loc=m, scale=sd / np.sqrt(len(values)))
    return m, float(lo), float(hi)


def _verdict(deltas: np.ndarray) -> tuple[float, float, float, str]:
    """Render PASS/FAIL on a paired-delta array under the frozen CI criterion."""
    m, lo, hi = _ci(deltas)
    passed = (lo > 0 or hi < 0) and m >= 0.01
    return m, lo, hi, "PASS" if passed else "FAIL"


def run(target: str) -> dict:
    """Run the structure-only edge ladder across 10 seeds."""
    chance = majority_class_macro_f1(np.array(list(_load_labels(target).values()), dtype=np.int64))

    # ── Prepare the three frozen embeddings ONCE (target-agnostic) ──────────
    print("Building histogram (no-edge) features...")
    hist_feats, hist_ids = build_histogram_features(target)
    hist_idx = {tid: i for i, tid in enumerate(hist_ids)}

    print("Encoding typed GINEConv embeddings (cached)...")
    typed_embs, typed_ids = encode_graphs_gine()
    typed_idx = {tid: i for i, tid in enumerate(typed_ids)}

    print("Training + encoding untyped GINConv (structure-only, v4_think)...")
    graphs = _load_graphs()
    untyped_ids = [p.stem for p in sorted(V4_FREE_TEXT_DIR.glob("*.json"))]
    untyped_embs = _encode(_train_gin(graphs, seed=42), graphs)
    untyped_idx = {tid: i for i, tid in enumerate(untyped_ids)}

    feats = {
        "no_edges": (hist_feats, hist_idx),
        "untyped": (untyped_embs, untyped_idx),
        "typed": (typed_embs, typed_idx),
    }

    ids_to_labels = _load_labels(target)
    n_classes = len(_class_names(target))
    per_seed: dict[str, list[float]] = {a: [] for a in ARMS}
    per_class: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
    paired = {"typed_minus_untyped": [], "untyped_minus_histogram": [], "typed_minus_histogram": []}

    for seed in SEEDS:
        train_ids, _val_ids, test_ids = make_split(target, seed)
        y_train = np.array([ids_to_labels[t] for t in train_ids], dtype=np.int64)
        y_test = np.array([ids_to_labels[t] for t in test_ids], dtype=np.int64)

        f1s = {}
        for arm in ARMS:
            embs, idx = feats[arm]
            x_tr = np.array([embs[idx[t]] for t in train_ids])
            x_te = np.array([embs[idx[t]] for t in test_ids])
            f1, preds = _weighted_logreg_f1(x_tr, y_train, x_te, y_test, seed)
            f1s[arm] = f1
            per_seed[arm].append(f1)
            per_class[arm].append(
                np.array(
                    [f1_score(y_test == c, preds == c, zero_division=0) for c in range(n_classes)]
                )
            )

        paired["typed_minus_untyped"].append(f1s["typed"] - f1s["untyped"])
        paired["untyped_minus_histogram"].append(f1s["untyped"] - f1s["no_edges"])
        paired["typed_minus_histogram"].append(f1s["typed"] - f1s["no_edges"])
        print(
            f"  seed {seed}: hist={f1s['no_edges']:.4f} untyped={f1s['untyped']:.4f} "
            f"typed={f1s['typed']:.4f}"
        )

    # ── Aggregate ──────────────────────────────────────────────────────────
    arms_summary = {}
    for arm in ARMS:
        m, lo, hi = _ci(np.array(per_seed[arm]))
        pc = np.mean(np.stack(per_class[arm]), axis=0)
        arms_summary[arm] = {
            "mean_macro_f1": m,
            "ci95": [lo, hi],
            "delta_vs_chance": m - chance,
            "per_class_f1": {name: float(pc[i]) for i, name in enumerate(_class_names(target))},
        }

    deltas_summary = {}
    for name, vals in paired.items():
        m, lo, hi, verdict = _verdict(np.array(vals))
        deltas_summary[name] = {"mean_delta": m, "ci95": [lo, hi], "verdict": verdict}

    primary = deltas_summary["typed_minus_untyped"]["verdict"]

    payload = {
        "target": target,
        "graph_corpus": "v4_think (ADR-0004 edge ontology)",
        "node_features": "structure_only (type one-hot + degree, 5-d) — no label semantics",
        "seeds": SEEDS,
        "chance_baseline": chance,
        "arms": arms_summary,
        "paired_deltas": deltas_summary,
        "primary_verdict_edge_type": primary,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{target}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved to {OUT_DIR / f'{target}.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="H_edge structure-only edge ladder")
    parser.add_argument(
        "--target",
        choices=["cohort", "ai_adoption", "stance_ambivalence"],
        default="stance_ambivalence",
    )
    args = parser.parse_args()
    p = run(args.target)

    print(f"\n{'=' * 64}\nH_EDGE — {args.target} (structure-only, v4_think)\n{'=' * 64}")
    print(f"chance: {p['chance_baseline']:.4f}")
    for arm in ARMS:
        a = p["arms"][arm]
        print(
            f"  {arm:10s} F1={a['mean_macro_f1']:.4f} CI={a['ci95']} "
            f"d_chance={a['delta_vs_chance']:+.4f}"
        )
    print("Paired:")
    for name, d in p["paired_deltas"].items():
        print(f"  {name:26s} Δ={d['mean_delta']:+.4f} CI={d['ci95']} {d['verdict']}")
    print(f"\nPRIMARY (edge-type, typed>untyped): {p['primary_verdict_edge_type']}")


if __name__ == "__main__":
    main()
