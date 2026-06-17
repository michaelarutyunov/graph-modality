"""Synthetic demonstration for the graph-vs-text essay.

This is a CONTROLLED FICTION, not evidence about the world. It illustrates the
*logic* behind two claims that survive the data-processing inequality (DPI),
using numeric channels only — no prose is generated, and no claim is made that
these encodings resemble real interviews.

Panel A — EXTRACTABILITY (rig-proof).
    The "text" representation is the FULL flattened adjacency of a latent DAG:
    it contains *strictly more* information than the handful of graph-topology
    features (those features are a lossy function of the adjacency). So the
    graph view cannot smuggle in extra signal — the DPI is respected by
    construction. Yet a WEAK learner (logistic regression) does much better with
    the pre-computed topology features than with the raw adjacency, because the
    outcome depends on a non-linear structural quantity (longest-path length)
    the weak learner cannot recover from raw edges. A STRONG learner
    (gradient boosting) closes the gap on the raw adjacency — empirically
    confirming the information was there all along. The graph's advantage is
    EXTRACTION, not information.

Panel B — INDEPENDENT ELICITATION (the open question, as a knob).
    Two noisy observations of the same latent DAG: a "text" view and a "graph"
    view. A complementarity knob c interpolates from c=0 (the graph observes
    EXACTLY what the text observed — a graph *derived* from text, the DPI case,
    no gain) to c=1 (the graph is an INDEPENDENT elicitation that can recover
    structure the text dropped). We plot the incremental predictive value of
    adding the graph over text alone, as a function of c. Reality sits somewhere
    on this axis; we do not know where — that is the empirical question a human
    study would answer. The plot characterises the *dependence*, it does not
    assert a point.

Run:  uv run python synthetic_demo.py
Out:  synthetic_demo.png  + a numeric summary on stdout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ── configuration ─────────────────────────────────────────────────────
SEED = 42
# nodes per latent DAG; kept small so the strong learner can recover structure from raw adjacency
N_NODES = 8
EDGE_RHO = 0.30  # forward-edge density in the latent DAG
LABEL_NOISE = 0.07  # fraction of labels flipped (keeps AUROC < 1)

# Panel A (extractability)
A_SAMPLES = 6000  # enough data for the strong learner to extract longest-path from raw adjacency
A_REPEATS = 12

# Panel B (complementarity)
B_SAMPLES = 900
B_REPEATS = 16
B_P_OBSERVE = 0.50  # per-edge probability a true edge survives a single channel
B_GRID = np.linspace(0.0, 1.0, 9)  # complementarity knob c


# ── latent structure + features ───────────────────────────────────────
def make_random_dag(n: int, rho: float, rng: np.random.Generator) -> nx.DiGraph:
    """Random DAG: nodes in topological order, forward edge i->j (i<j) w.p. rho."""
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < rho:
                g.add_edge(i, j)
    return g


def flat_adjacency(g: nx.DiGraph, n: int) -> np.ndarray:
    """Full upper-triangular adjacency as a flat 0/1 vector — the complete structure."""
    vec = []
    for i in range(n):
        for j in range(i + 1, n):
            vec.append(1.0 if g.has_edge(i, j) else 0.0)
    return np.asarray(vec, dtype=np.float64)


def graph_features(g: nx.DiGraph) -> np.ndarray:
    """A small bag of topology features — a lossy summary of the adjacency."""
    out_deg = [d for _, d in g.out_degree()]
    in_deg = [d for _, d in g.in_degree()]
    return np.asarray(
        [
            nx.dag_longest_path_length(g),  # the Y-relevant non-linear quantity
            g.number_of_edges(),
            max(out_deg) if out_deg else 0,
            max(in_deg) if in_deg else 0,
            sum(1 for _, d in g.in_degree() if d == 0),  # source count
            sum(1 for _, d in g.out_degree() if d == 0),  # sink count
            len(nx.descendants(g, 0)),  # reach from root
        ],
        dtype=np.float64,
    )


def labels_from_longest_path(lp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Binary label: is the longest path above the sample median? (+ label noise)."""
    y = (lp > np.median(lp)).astype(int)
    flip = rng.random(len(y)) < LABEL_NOISE
    y[flip] = 1 - y[flip]
    return y


# ── Panel A: extractability ───────────────────────────────────────────
def build_extractability(rng: np.random.Generator):
    """Returns (X_flat, X_graph, y). X_flat is the COMPLETE structure (>= info of X_graph)."""
    flats, feats, lps = [], [], []
    for _ in range(A_SAMPLES):
        g = make_random_dag(N_NODES, EDGE_RHO, rng)
        flats.append(flat_adjacency(g, N_NODES))
        feats.append(graph_features(g))
        lps.append(nx.dag_longest_path_length(g))
    y = labels_from_longest_path(np.asarray(lps), rng)
    return np.vstack(flats), np.vstack(feats), y


def auroc(model, x_tr, x_te, y_tr, y_te) -> float:
    model.fit(x_tr, y_tr)
    prob = model.predict_proba(x_te)[:, 1]
    return roc_auc_score(y_te, prob)


def run_extractability(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Weak (LogReg) vs strong (gradient boosting) learner, on flat-text vs graph features."""
    keys = ["weak_text", "weak_graph", "strong_text", "strong_graph"]
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(A_REPEATS):
        x_flat, x_graph, y = build_extractability(rng)
        idx = np.arange(len(y))
        tr, te = train_test_split(
            idx, test_size=0.3, random_state=int(rng.integers(1e9)), stratify=y
        )

        weak = lambda: LogisticRegression(max_iter=2000)  # noqa: E731
        strong = lambda: HistGradientBoostingClassifier(random_state=0, max_iter=400)  # noqa: E731

        acc["weak_text"].append(auroc(weak(), x_flat[tr], x_flat[te], y[tr], y[te]))
        acc["weak_graph"].append(auroc(weak(), x_graph[tr], x_graph[te], y[tr], y[te]))
        acc["strong_text"].append(auroc(strong(), x_flat[tr], x_flat[te], y[tr], y[te]))
        acc["strong_graph"].append(auroc(strong(), x_graph[tr], x_graph[te], y[tr], y[te]))
    return {k: np.asarray(v) for k, v in acc.items()}


# ── Panel B: independent elicitation (complementarity sweep) ──────────
def observe(g: nx.DiGraph, p: float, rng: np.random.Generator) -> dict[tuple[int, int], bool]:
    """A single noisy observation: each TRUE edge survives with probability p."""
    return {e: (rng.random() < p) for e in g.edges()}


def graph_from_kept(kept: dict[tuple[int, int], bool], n: int) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(e for e, keep in kept.items() if keep)
    return g


def build_complementarity(rng: np.random.Generator):
    """Returns true DAGs + labels; observations are generated per-c later (shared truth)."""
    dags, lps = [], []
    for _ in range(B_SAMPLES):
        g = make_random_dag(N_NODES, EDGE_RHO, rng)
        dags.append(g)
        lps.append(nx.dag_longest_path_length(g))
    y = labels_from_longest_path(np.asarray(lps), rng)
    return dags, y


def features_for_c(dags, c: float, rng: np.random.Generator):
    """Text view + graph view under complementarity c.

    c=0  -> the graph observes EXACTLY what the text observed (derived graph, DPI).
    c=1  -> the graph is an INDEPENDENT observation of the same latent DAG.
    """
    feat_t, feat_g = [], []
    for g in dags:
        t_keep = observe(g, B_P_OBSERVE, rng)
        g_keep = {}
        for e, t_decision in t_keep.items():
            if rng.random() < c:  # independent re-observation of this edge
                g_keep[e] = rng.random() < B_P_OBSERVE
            else:  # copy the text channel's decision (redundant)
                g_keep[e] = t_decision
        feat_t.append(graph_features(graph_from_kept(t_keep, N_NODES)))
        feat_g.append(graph_features(graph_from_kept(g_keep, N_NODES)))
    return np.vstack(feat_t), np.vstack(feat_g)


def run_complementarity(rng: np.random.Generator) -> dict[float, np.ndarray]:
    """Incremental AUROC of (text+graph) over (text alone) as a function of c."""
    deltas: dict[float, list[float]] = {float(c): [] for c in B_GRID}
    for _ in range(B_REPEATS):
        dags, y = build_complementarity(rng)
        idx = np.arange(len(y))
        tr, te = train_test_split(
            idx, test_size=0.3, random_state=int(rng.integers(1e9)), stratify=y
        )
        for c in B_GRID:
            x_t, x_g = features_for_c(dags, float(c), rng)
            x_tg = np.hstack([x_t, x_g])
            auc_t = auroc(
                HistGradientBoostingClassifier(random_state=0), x_t[tr], x_t[te], y[tr], y[te]
            )
            auc_tg = auroc(
                HistGradientBoostingClassifier(random_state=0), x_tg[tr], x_tg[te], y[tr], y[te]
            )
            deltas[float(c)].append(auc_tg - auc_t)
    return {c: np.asarray(v) for c, v in deltas.items()}


# ── reporting ─────────────────────────────────────────────────────────
def ci(arr: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% percentile interval."""
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def report(ext: dict[str, np.ndarray], comp: dict[float, np.ndarray]) -> None:
    print("\n=== Panel A — extractability (AUROC, mean [95% CI]) ===")
    print("  text = full flattened adjacency (>= information of graph features)")
    for label, key in [
        ("weak  (logistic regression)  text ", "weak_text"),
        ("weak  (logistic regression)  graph", "weak_graph"),
        ("strong(gradient boosting)    text ", "strong_text"),
        ("strong(gradient boosting)    graph", "strong_graph"),
    ]:
        m, lo, hi = ci(ext[key])
        print(f"  {label}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
    wg, _, _ = ci(ext["weak_graph"])
    wt, _, _ = ci(ext["weak_text"])
    sg, _, _ = ci(ext["strong_graph"])
    st, _, _ = ci(ext["strong_text"])
    print(f"  -> weak  graph-text gap = {wg - wt:+.3f}  (extractability advantage)")
    print(f"  -> strong graph-text gap = {sg - st:+.3f}  (gap closes: information was there)")

    print("\n=== Panel B — incremental AUROC of (text+graph) over text, by complementarity c ===")
    for c in sorted(comp):
        m, lo, hi = ci(comp[c])
        print(f"  c={c:.3f}: delta = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("  c=0 = graph derived from text (DPI: no gain);  c=1 = independent elicitation.")
    print("  Reality is somewhere on this axis — and we do not know where.\n")


def make_figure(ext: dict[str, np.ndarray], comp: dict[float, np.ndarray], path: str) -> None:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: grouped bars
    groups = ["weak\n(logistic reg.)", "strong\n(grad. boosting)"]
    text_means = [ext["weak_text"].mean(), ext["strong_text"].mean()]
    graph_means = [ext["weak_graph"].mean(), ext["strong_graph"].mean()]
    text_err = [
        [
            text_means[0] - np.percentile(ext["weak_text"], 2.5),
            text_means[1] - np.percentile(ext["strong_text"], 2.5),
        ],
        [
            np.percentile(ext["weak_text"], 97.5) - text_means[0],
            np.percentile(ext["strong_text"], 97.5) - text_means[1],
        ],
    ]
    graph_err = [
        [
            graph_means[0] - np.percentile(ext["weak_graph"], 2.5),
            graph_means[1] - np.percentile(ext["strong_graph"], 2.5),
        ],
        [
            np.percentile(ext["weak_graph"], 97.5) - graph_means[0],
            np.percentile(ext["strong_graph"], 97.5) - graph_means[1],
        ],
    ]
    x = np.arange(len(groups))
    w = 0.36
    axa.bar(
        x - w / 2,
        text_means,
        w,
        yerr=text_err,
        capsize=4,
        label="text (full adjacency)",
        color="#4C72B0",
    )
    axa.bar(
        x + w / 2,
        graph_means,
        w,
        yerr=graph_err,
        capsize=4,
        label="graph (topology features)",
        color="#C44E52",
    )
    axa.set_xticks(x)
    axa.set_xticklabels(groups)
    axa.set_ylabel("test AUROC")
    axa.set_ylim(0.5, 1.0)
    axa.set_title("A. Extractability\n(text has MORE information; graph is easier to use)")
    axa.legend(loc="lower right", fontsize=9)
    axa.grid(axis="y", alpha=0.3)

    # Panel B: complementarity sweep
    cs = np.asarray(sorted(comp))
    means = np.asarray([comp[c].mean() for c in cs])
    los = np.asarray([np.percentile(comp[c], 2.5) for c in cs])
    his = np.asarray([np.percentile(comp[c], 97.5) for c in cs])
    axb.axhline(0.0, color="grey", lw=1, ls="--")
    axb.plot(cs, means, marker="o", color="#55A868", label="graph's incremental AUROC")
    axb.fill_between(cs, los, his, alpha=0.2, color="#55A868")
    axb.annotate(
        "derived graph\n(= DPI, no gain)",
        xy=(0.0, means[0]),
        xytext=(0.05, max(his) * 0.55 + 0.005),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="grey"),
    )
    axb.axvspan(0.35, 0.85, color="orange", alpha=0.08)
    axb.text(
        0.6,
        max(his) * 0.92 + 0.002,
        "reality = unknown",
        ha="center",
        fontsize=8,
        color="darkorange",
    )
    axb.set_xlabel(
        "channel complementarity  c\n(0 = graph copies text,  1 = independent elicitation)"
    )
    axb.set_ylabel("AUROC(text+graph) - AUROC(text)")
    axb.set_title("B. Independent elicitation\n(the open empirical question, as a knob)")
    axb.grid(alpha=0.3)

    fig.suptitle(
        "A graph derived from text cannot beat text (DPI). It can be easier to extract from (A); "
        "and an independently-elicited graph can add signal (B) — IF channels are complementary.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    print(f"figure written -> {path}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("Running synthetic demo (controlled fiction — illustrates logic, not reality)...")
    ext = run_extractability(rng)
    comp = run_complementarity(rng)
    report(ext, comp)
    make_figure(ext, comp, str(Path(__file__).with_name("synthetic_demo.png")))

    # sanity checks (the demo's internal logic must hold)
    weak_gap = ext["weak_graph"].mean() - ext["weak_text"].mean()
    strong_gap = ext["strong_graph"].mean() - ext["strong_text"].mean()
    assert weak_gap > 0, "expected weak-learner extractability gap (graph > text)"
    assert strong_gap < weak_gap, (
        "expected the gap to narrow for the strong learner (info was in text)"
    )
    assert abs(comp[0.0].mean()) < 0.02, "derived graph (c=0) must add ~nothing (DPI)"


if __name__ == "__main__":
    main()
