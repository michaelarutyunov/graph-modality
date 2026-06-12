"""Null-ladder edge test: GINEConv(typed) vs histogram baseline.

Decisive experiment for the v4 edge-modality hypothesis: does typed relational
structure (GINEConv with edge-type embeddings) carry classification signal
beyond a bag-of-types null (node+edge-type frequency histogram)?

Design:
  - Both arms use STRUCTURE_ONLY features (type+degree, 5-d) — no label semantics
  - Null: per-graph 11-dim feature vector (4 node-type bins + 6 edge-type bins
    + mean degree) → LogisticRegression classifier. Captures type DISTRIBUTION
    but ZERO wiring/topology information.
  - Alternative: GINEConv(typed) encoder → frozen 128-dim embeddings →
    LogisticRegression classifier. Captures wiring WITH edge-type labels.
  - Target: cohort (workforce/creatives/scientists)
  - 10 seeds (42-51), frozen encoder per seed, paired per-seed deltas
  - CI criterion: 95% CI on mean delta excludes 0 AND mean delta >= +0.01 macro-F1

Usage:
    PYTHONPATH=. uv run python s5_classification/null_ladder.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from s4_encoding.graph_gnn_encoder import (
    V4_FREE_TEXT_DIR,
    encode_graphs_gine,
    train_gine_autoencoder,
)
from s5_classification.repeated_eval import make_split
from s5_classification.split import load_transcript_ids_with_labels

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
LADDER_DIR = CACHE_DIR / "null_ladder"
HISTOGRAM_CACHE = LADDER_DIR / "histogram_features.npy"
HISTOGRAM_IDS_CACHE = LADDER_DIR / "histogram_ids.json"
RESULTS_PATH = LADDER_DIR / "results.json"

# ── Constants ────────────────────────────────────────────────────────────────
SEEDS = list(range(42, 52))  # 42..51 inclusive
NODE_TYPES = ["Construct", "Value", "Stance", "CognitiveStyleMarker"]
RELATIONS = ["SERVES", "EXPRESSED_VIA", "MODULATED_BY", "CONFLICTS_WITH", "SUBSUMES", "IMPLIES"]
TARGET = "cohort"


# ═══════════════════════════════════════════════════════════════════════════════
# Histogram baseline (bag-of-types, no wiring)
# ═══════════════════════════════════════════════════════════════════════════════


def build_histogram_features(graph_dir: Path | None = None) -> tuple[np.ndarray, list[str]]:
    """Build 11-dim bag-of-types feature vectors for all graphs.

    Features (11 dims):
      0-3:  node type histogram (Construct, Value, Stance, CSM)
      4-9:  edge type histogram (SERVES, EXPRESSED_VIA, MODULATED_BY,
            CONFLICTS_WITH, SUBSUMES, IMPLIES)
      10:   mean degree (total degree / n_nodes, or 0 for empty graphs)

    Returns:
        (features, transcript_ids) — aligned arrays.
    """
    if graph_dir is None:
        graph_dir = V4_FREE_TEXT_DIR

    if HISTOGRAM_CACHE.exists() and HISTOGRAM_IDS_CACHE.exists():
        print("Loading cached histogram features...")
        return np.load(HISTOGRAM_CACHE), json.loads(
            HISTOGRAM_IDS_CACHE.read_text(encoding="utf-8")
        )

    paths = sorted(graph_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No graph files found in {graph_dir}")

    n_type_idx = {t: i for i, t in enumerate(NODE_TYPES)}
    n_rel_idx = {r: i for i, r in enumerate(RELATIONS)}

    features_list: list[np.ndarray] = []
    ids: list[str] = []

    for p in paths:
        g = json.loads(p.read_text(encoding="utf-8"))
        nodes = g.get("nodes", [])
        edges = g.get("edges", [])

        n_nodes = len(nodes)
        n_edges = len(edges)

        # Node type histogram (normalised by node count)
        node_hist = np.zeros(len(NODE_TYPES), dtype=np.float32)
        for n in nodes:
            nt = n.get("type", "")
            if nt in n_type_idx:
                node_hist[n_type_idx[nt]] += 1.0
        if n_nodes > 0:
            node_hist /= n_nodes

        # Edge type histogram (normalised by edge count)
        edge_hist = np.zeros(len(RELATIONS), dtype=np.float32)
        for e in edges:
            rel = e.get("relation", "?")
            if rel in n_rel_idx:
                edge_hist[n_rel_idx[rel]] += 1.0
        if n_edges > 0:
            edge_hist /= n_edges

        # Mean degree
        mean_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0

        fv = np.concatenate(
            [node_hist, edge_hist, np.array([mean_degree], dtype=np.float32)]
        )
        features_list.append(fv)
        ids.append(g.get("transcript_id", p.stem))

    result = np.stack(features_list, axis=0)  # (N, 11)

    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    np.save(HISTOGRAM_CACHE, result)
    HISTOGRAM_IDS_CACHE.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"Cached {len(ids)} histogram features → {HISTOGRAM_CACHE}")

    return result, ids


# ═══════════════════════════════════════════════════════════════════════════════
# Per-seed evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def _train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> float:
    """Train a LogisticRegression classifier and return macro-F1 on test set."""
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return float(f1_score(y_test, preds, average="macro"))


def run_seed(seed: int) -> dict:
    """Run one seed of the null-ladder protocol.

    Returns:
        Dict with seed, gine_f1, hist_f1, delta.
    """
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    # ── Split ────────────────────────────────────────────────────────────
    train_ids, _val_ids, test_ids = make_split(TARGET, seed)
    ids_to_labels = load_transcript_ids_with_labels()

    y_train = np.array([ids_to_labels[tid] for tid in train_ids], dtype=np.int64)
    y_test = np.array([ids_to_labels[tid] for tid in test_ids], dtype=np.int64)

    # ── GINE arm ─────────────────────────────────────────────────────────
    print("  Training GINE autoencoder...")
    gine_results = train_gine_autoencoder(
        graph_dir=V4_FREE_TEXT_DIR,
        max_epochs=100,  # full training per seed
    )
    print(f"    Best loss: {gine_results['best_loss']:.4f} (epoch {gine_results['best_epoch']})")

    # Encode (uses the just-trained encoder — cached at V4_GINE_ENCODER_PATH)
    gine_embs, gine_ids = encode_graphs_gine(force=True)

    # Slice to split
    gine_lookup = {tid: i for i, tid in enumerate(gine_ids)}
    gine_train = np.array([gine_embs[gine_lookup[tid]] for tid in train_ids])
    gine_test = np.array([gine_embs[gine_lookup[tid]] for tid in test_ids])

    # Use val set for... LogisticRegression doesn't have early stopping.
    # We train on train and evaluate on test directly.
    gine_f1 = _train_and_eval(gine_train, y_train, gine_test, y_test, seed)
    print(f"  GINEConv(typed) test macro-F1: {gine_f1:.4f}")

    # ── Histogram arm ────────────────────────────────────────────────────
    hist_feats, hist_ids = build_histogram_features()
    hist_lookup = {tid: i for i, tid in enumerate(hist_ids)}
    hist_train = np.array([hist_feats[hist_lookup[tid]] for tid in train_ids])
    hist_test = np.array([hist_feats[hist_lookup[tid]] for tid in test_ids])

    hist_f1 = _train_and_eval(hist_train, y_train, hist_test, y_test, seed)
    print(f"  Histogram (bag-of-types) test macro-F1: {hist_f1:.4f}")

    delta = gine_f1 - hist_f1
    print(f"  Δ (GINE - histogram): {delta:+.4f}")

    return {
        "seed": seed,
        "gine_f1": gine_f1,
        "hist_f1": hist_f1,
        "delta": delta,
        "gine_best_loss": gine_results["best_loss"],
        "gine_best_epoch": gine_results["best_epoch"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the full null-ladder protocol and record the verdict."""
    print("=" * 60)
    print("NULL-LADDER EDGE TEST")
    print(f"Target: {TARGET}")
    print(f"Seeds: {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} seeds)")
    print("Null: 11-dim bag-of-types histogram → LogisticRegression")
    print("Alt:  GINEConv(typed) 128-dim embeddings → LogisticRegression")
    print("=" * 60)

    # Pre-build histogram features once (deterministic)
    print("\nBuilding histogram features...")
    build_histogram_features()

    # Run all seeds
    results: list[dict] = []
    for seed in SEEDS:
        r = run_seed(seed)
        results.append(r)

    # ── Statistics ───────────────────────────────────────────────────────
    deltas = np.array([r["delta"] for r in results])
    gine_scores = np.array([r["gine_f1"] for r in results])
    hist_scores = np.array([r["hist_f1"] for r in results])

    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1))
    n = len(deltas)

    # 95% CI via t-distribution
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_delta, scale=std_delta / np.sqrt(n))

    # Verdict
    ci_excludes_zero = bool(ci_low > 0 or ci_high < 0)
    meets_threshold = bool(mean_delta >= 0.01)
    verdict = "PASS" if (ci_excludes_zero and meets_threshold) else "FAIL"

    # ── Output ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print("\nPer-seed scores:")
    for r in results:
        print(
            f"  Seed {r['seed']:2d}:  GINE={r['gine_f1']:.4f}  "
            f"Hist={r['hist_f1']:.4f}  Δ={r['delta']:+.4f}"
        )

    print("\nSummary:")
    gine_mean = np.mean(gine_scores)
    gine_std = np.std(gine_scores, ddof=1)
    hist_mean = np.mean(hist_scores)
    hist_std = np.std(hist_scores, ddof=1)
    print(f"  GINE mean macro-F1:     {gine_mean:.4f} +/- {gine_std:.4f}")
    print(f"  Histogram mean macro-F1: {hist_mean:.4f} +/- {hist_std:.4f}")
    print(f"  Mean Δ:                  {mean_delta:+.4f}")
    print(f"  95% CI:                  [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  CI excludes 0:           {ci_excludes_zero}")
    print(f"  Mean Δ ≥ +0.01:          {meets_threshold}")
    print(f"\n  VERDICT: {verdict}")

    # ── Per-class breakdown ──────────────────────────────────────────────
    # Compute per-class F1 for the last seed (representative)
    print(f"\nPer-class F1 (seed {SEEDS[-1]}):")
    ids_to_labels = load_transcript_ids_with_labels()
    _train_ids, _val_ids, test_ids = make_split(TARGET, SEEDS[-1])
    y_test = np.array([ids_to_labels[tid] for tid in test_ids], dtype=np.int64)

    # GINE
    gine_embs, gine_ids = encode_graphs_gine()
    gine_lookup = {tid: i for i, tid in enumerate(gine_ids)}
    gine_test = np.array([gine_embs[gine_lookup[tid]] for tid in test_ids])
    clf_g = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEEDS[-1])
    clf_g.fit(
        np.array([gine_embs[gine_lookup[tid]] for tid in _train_ids]),
        np.array([ids_to_labels[tid] for tid in _train_ids], dtype=np.int64),
    )
    gine_preds = clf_g.predict(gine_test)

    # Histogram
    hist_feats, hist_ids = build_histogram_features()
    hist_lookup = {tid: i for i, tid in enumerate(hist_ids)}
    hist_test = np.array([hist_feats[hist_lookup[tid]] for tid in test_ids])
    clf_h = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEEDS[-1])
    clf_h.fit(
        np.array([hist_feats[hist_lookup[tid]] for tid in _train_ids]),
        np.array([ids_to_labels[tid] for tid in _train_ids], dtype=np.int64),
    )
    hist_preds = clf_h.predict(hist_test)

    class_names = ["workforce", "creatives", "scientists"]
    for i, name in enumerate(class_names):
        gine_cf1 = f1_score(y_test == i, gine_preds == i)
        hist_cf1 = f1_score(y_test == i, hist_preds == i)
        delta_cf1 = gine_cf1 - hist_cf1
        print(f"  {name:12s}: GINE={gine_cf1:.4f}  Hist={hist_cf1:.4f}  delta={delta_cf1:+.4f}")

    # ── Persist ──────────────────────────────────────────────────────────
    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": TARGET,
        "seeds": SEEDS,
        "null_arm": "11-dim bag-of-types histogram → LogisticRegression",
        "alt_arm": "GINEConv(typed) 128-dim embeddings → LogisticRegression",
        "results": results,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_95": [float(ci_low), float(ci_high)],
        "ci_excludes_zero": ci_excludes_zero,
        "meets_threshold": meets_threshold,
        "verdict": verdict,
        "gine_mean_f1": float(np.mean(gine_scores)),
        "hist_mean_f1": float(np.mean(hist_scores)),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Record verdict in results-log.md ─────────────────────────────────
    _append_verdict_to_log(payload)


def _append_verdict_to_log(payload: dict) -> None:
    """Append the null-ladder verdict to the results log."""
    log_path = Path(".claude/context/results-log.md")

    verdict = payload["verdict"]
    mean_delta = payload["mean_delta"]
    ci_low, ci_high = payload["ci_95"]

    entry = f"""
### Null-Ladder Edge Test (v4) --- {verdict}

**Date:** 2026-06-12
**Target:** cohort (workforce/creatives/scientists)
**Protocol:** 10-seed frozen CI ({payload['seeds'][0]}-{payload['seeds'][-1]})

**Arms:**
- Null: 11-dim bag-of-types histogram
  (node type + edge type frequencies + mean degree) -> LogisticRegression
- Alternative: GINEConv(typed) 128-dim frozen embeddings -> LogisticRegression

**Results:**
- GINEConv mean macro-F1: {payload['gine_mean_f1']:.4f}
- Histogram mean macro-F1: {payload['hist_mean_f1']:.4f}
- Mean delta: {mean_delta:+.4f}
- 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]
- CI excludes 0: {payload['ci_excludes_zero']}
- Mean delta >= +0.01: {payload['meets_threshold']}

**Per-seed:**
| Seed | GINE F1 | Hist F1 | delta |
|------|---------|---------|---|
"""
    for r in payload["results"]:
        entry += (
            f"| {r['seed']} | {r['gine_f1']:.4f} | "
            f"{r['hist_f1']:.4f} | {r['delta']:+.4f} |\n"
        )

    if verdict == "PASS":
        interpretation = (
            "Typed relational structure carries signal beyond bag-of-types "
            "-- edge-type-aware GNN beats the no-wiring null."
        )
    else:
        interpretation = (
            "Typed relational structure does NOT reliably beat the "
            "bag-of-types null under the CI criterion -- the edge-type "
            "hypothesis is not supported."
        )

    entry += f"""
**Verdict:** {verdict}
**Interpretation:** {interpretation}

---
"""

    # Append to log
    with open(log_path, "a") as f:
        f.write(entry)
    print(f"Verdict appended to {log_path}")


if __name__ == "__main__":
    main()
