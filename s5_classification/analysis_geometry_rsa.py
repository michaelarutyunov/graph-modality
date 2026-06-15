"""RSA/Mantel geometry test: is the concept graph a label-free distinct modality (v4).

Compares the similarity geometry of text, graph-statistic, and label-bag
representations over the cached v4 embeddings for all 1,250 transcripts using
Representational Similarity Analysis (Kriegeskorte 2008) and Mantel permutation
tests (Mantel 1967).  Includes anti-noise gates and an interpretability gate.

Usage:
    uv run python s5_classification/analysis_geometry_rsa.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from s4_encoding.graph_stats_encoder import compute_all_stats
from s4_encoding.label_bag_encoder import encode_label_bag
from s4_encoding.text_encoder import encode_transcripts
from s5_classification.analysis_feature_importance import FEATURE_NAMES
from s5_classification.split import load_split

# ── Paths and constants ───────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
DEMOGRAPHICS_PATH = CACHE_DIR / "demographics.jsonl"
AMBIVALENCE_PATH = CACHE_DIR / "ambivalence.jsonl"
DEFAULT_OUT_DIR = Path("results/method_review/geometry_rsa_v4")

N_PERM_MANTEL = 9_999
N_RANDOM_NULL = 100
N_SPLIT_HALF = 100
SPLIT_HALF_THRESHOLD = 0.5
RANDOM_NULL_PERCENTILE = 95.0
DISTINCT_THRESHOLD = 0.7
SEED = 42


# ── Distance helpers ──────────────────────────────────────────────────────────
def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """Z-score each column of ``matrix`` using the sample standard deviation."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    # Guard against zero-variance columns; distance matrices remain well-defined.
    std_safe = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std_safe


def _cosine_distance_condensed(matrix: np.ndarray) -> np.ndarray:
    """Return the condensed cosine-distance vector for ``matrix``."""
    return pdist(matrix.astype(np.float64), metric="cosine")


def _square_to_condensed(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build index tables for mapping a condensed distance vector to square form.

    Returns:
        (i_idx, j_idx, cond_index) where ``cond_index[i, j]`` gives the
        position in the condensed vector for pair ``(i, j)``.
    """
    m = n * (n - 1) // 2
    i_idx = np.empty(m, dtype=np.int64)
    j_idx = np.empty(m, dtype=np.int64)
    cond_index = np.full((n, n), -1, dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            i_idx[k] = i
            j_idx[k] = j
            cond_index[i, j] = k
            cond_index[j, i] = k
            k += 1
    return i_idx, j_idx, cond_index


def _permute_condensed(
    condensed: np.ndarray,
    perm: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    cond_index: np.ndarray,
) -> np.ndarray:
    """Apply a row/column permutation to a condensed distance vector."""
    new_positions = cond_index[perm[i_idx], perm[j_idx]]
    return condensed[new_positions]


# ── Mantel core ───────────────────────────────────────────────────────────────
def _spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two equally-sized vectors."""
    ra = rankdata(a)
    rb = rankdata(b)
    za = (ra - ra.mean()) / ra.std(ddof=1)
    zb = (rb - rb.mean()) / rb.std(ddof=1)
    return float(np.dot(za, zb) / (len(a) - 1))


def mantel_test(
    d1: np.ndarray,
    d2: np.ndarray,
    n_perm: int = N_PERM_MANTEL,
    seed: int = SEED,
    two_sided: bool = True,
) -> dict[str, Any]:
    """Mantel test with Spearman correlation between two condensed distance vectors.

    Args:
        d1, d2: Condensed distance vectors of equal length.
        n_perm: Number of row/column permutations of ``d2`` under the null.
        seed: Random seed for permutations.
        two_sided: If True, report a two-sided p-value; otherwise one-sided
            (right tail, i.e. proportion of null correlations >= observed).

    Returns:
        Dictionary with ``r`` (observed correlation), ``p`` (permutation p),
        ``n`` (number of objects), and ``n_perm``.
    """
    if d1.ndim != 1 or d2.ndim != 1:
        raise ValueError("d1 and d2 must be 1-D condensed distance vectors")

    if len(d1) != len(d2):
        raise ValueError("d1 and d2 must have the same length")

    m = len(d1)
    n = int((1 + np.sqrt(1 + 8 * m)) / 2)
    if m != n * (n - 1) // 2:
        raise ValueError("d1 is not a valid condensed distance vector")

    i_idx, j_idx, cond_index = _square_to_condensed(n)

    r1 = rankdata(d1)
    r2 = rankdata(d2)
    z1 = (r1 - r1.mean()) / r1.std(ddof=1)
    z2 = (r2 - r2.mean()) / r2.std(ddof=1)
    r_obs = float(np.dot(z1, z2) / (m - 1))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p_i in range(n_perm):
        perm = rng.permutation(n)
        z2_perm = _permute_condensed(z2, perm, i_idx, j_idx, cond_index)
        null[p_i] = float(np.dot(z1, z2_perm) / (m - 1))

    if two_sided:
        p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1)
    else:
        p = (np.sum(null >= r_obs) + 1) / (n_perm + 1)

    return {
        "r": r_obs,
        "p": float(p),
        "n": n,
        "n_perm": n_perm,
    }


def partial_mantel_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_perm: int = N_PERM_MANTEL,
    seed: int = SEED,
) -> dict[str, Any]:
    """Partial Mantel test: correlation of ``x`` and ``y`` residualised on ``z``.

    Uses Spearman correlation of residuals.  The permutation distribution is
    obtained by permuting the residual vector of ``y``.
    """
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise ValueError("x, y, and z must be 1-D condensed distance vectors")

    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, and z must have the same length")

    m = len(x)
    n = int((1 + np.sqrt(1 + 8 * m)) / 2)
    if m != n * (n - 1) // 2:
        raise ValueError("x is not a valid condensed distance vector")

    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    rz_c = rz - rz.mean()

    beta_x = np.dot(rx_c, rz_c) / np.dot(rz_c, rz_c)
    beta_y = np.dot(ry_c, rz_c) / np.dot(rz_c, rz_c)

    ex = rx_c - beta_x * rz_c
    ey = ry_c - beta_y * rz_c

    zx = (ex - ex.mean()) / ex.std(ddof=1)
    zy = (ey - ey.mean()) / ey.std(ddof=1)

    r_obs = float(np.dot(zx, zy) / (m - 1))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p_i in range(n_perm):
        perm = rng.permutation(m)
        null[p_i] = float(np.dot(zx, zy[perm]) / (m - 1))

    p = (np.sum(null >= r_obs) + 1) / (n_perm + 1)

    return {
        "r": r_obs,
        "p": float(p),
        "n": n,
        "n_perm": n_perm,
    }


# ── Anti-noise gates ───────────────────────────────────────────────────────────
def random_embedding_null(
    text_d: np.ndarray,
    n_objects: int,
    n_features: int = 30,
    n_draws: int = N_RANDOM_NULL,
    seed: int = SEED,
) -> dict[str, Any]:
    """Draw random Gaussian embeddings and Mantel each against text.

    Returns the null distribution of Spearman r and the 95th percentile.
    """
    rng = np.random.default_rng(seed)
    null_rs: list[float] = []
    for _ in range(n_draws):
        rand = rng.standard_normal((n_objects, n_features))
        rand_z = _zscore_columns(rand)
        rand_d = _cosine_distance_condensed(rand_z)
        null_rs.append(_spearman_r(text_d, rand_d))

    pct95 = float(np.percentile(null_rs, RANDOM_NULL_PERCENTILE))
    return {
        "n_draws": n_draws,
        "n_features": n_features,
        "percentile": RANDOM_NULL_PERCENTILE,
        "pct95_r": pct95,
        "null_rs": null_rs,
    }


def split_half_reliability(
    stats_matrix: np.ndarray,
    n_splits: int = N_SPLIT_HALF,
    seed: int = SEED,
) -> dict[str, Any]:
    """Split the 30 graph-stats features into two random halves and Mantel them."""
    rng = np.random.default_rng(seed)
    n_features = stats_matrix.shape[1]
    half = n_features // 2
    rs: list[float] = []

    for _ in range(n_splits):
        perm = rng.permutation(n_features)
        half_a = perm[:half]
        half_b = perm[half : half * 2]
        za = _zscore_columns(stats_matrix[:, half_a])
        zb = _zscore_columns(stats_matrix[:, half_b])
        da = _cosine_distance_condensed(za)
        db = _cosine_distance_condensed(zb)
        rs.append(_spearman_r(da, db))

    return {
        "n_splits": n_splits,
        "half_size": half,
        "mean_r": float(np.mean(rs)),
        "rs": rs,
    }


# ── Data loading ───────────────────────────────────────────────────────────────
def _load_anchor_labels() -> dict[str, dict[str, Any]]:
    """Load anchor label dictionaries for cohort, stance_ambivalence, ai_adoption."""
    anchors: dict[str, dict[str, Any]] = {}

    # cohort
    _, _, _, cohort_labels = load_split()
    anchors["cohort"] = cohort_labels

    # stance_ambivalence
    amb_labels: dict[str, str] = {}
    with open(AMBIVALENCE_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            amb_labels[rec["transcript_id"]] = rec["stance_ambivalence"]["label"]
    anchors["stance_ambivalence"] = amb_labels

    # ai_adoption (binary; exclude novice/power_user)
    ai_labels: dict[str, str] = {}
    with open(DEMOGRAPHICS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            raw = rec["ai_adoption"]["label"]
            if raw in ("tool_user", "integrated"):
                ai_labels[rec["transcript_id"]] = raw
    anchors["ai_adoption"] = ai_labels

    return anchors


def _align_modalities() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load cached embeddings and align them to the common transcript set."""
    text_emb, text_ids = encode_transcripts(speaker_filter="Human")
    stats_emb, stats_ids = compute_all_stats()
    label_emb, label_ids = encode_label_bag()

    common = sorted(set(text_ids) & set(stats_ids) & set(label_ids))
    if not common:
        raise ValueError("No common transcript IDs across modalities")

    text_map = {tid: i for i, tid in enumerate(text_ids)}
    stats_map = {tid: i for i, tid in enumerate(stats_ids)}
    label_map = {tid: i for i, tid in enumerate(label_ids)}

    text_aligned = np.stack([text_emb[text_map[tid]] for tid in common])
    stats_aligned = np.stack([stats_emb[stats_map[tid]] for tid in common])
    label_aligned = np.stack([label_emb[label_map[tid]] for tid in common])

    return text_aligned, stats_aligned, label_aligned, common


# ── Interpretability gate ─────────────────────────────────────────────────────
def _write_disagreement_cases(
    out_path: Path,
    ids: list[str],
    graph_z: np.ndarray,
    text_d_square: np.ndarray,
    graph_d_square: np.ndarray,
    top_k: int = 20,
    n_driving: int = 5,
) -> None:
    """Write the top-k disagreement pairs in each direction."""
    n = len(ids)
    i_idx, j_idx = np.triu_indices(n, k=1)
    graph_d = graph_d_square[i_idx, j_idx]
    text_d = text_d_square[i_idx, j_idx]

    # Close in graph, far in text
    score_graph_close = text_d - graph_d
    close_graph_order = np.argsort(score_graph_close)[::-1][:top_k]

    # Close in text, far in graph
    score_text_close = graph_d - text_d
    close_text_order = np.argsort(score_text_close)[::-1][:top_k]

    lines = [
        "# Geometry disagreement cases",
        "",
        "Pairs where the graph-statistic geometry and the text geometry disagree ",
        "most strongly.  Features are z-scored across the corpus; larger absolute",
        "z-differences contribute more to cosine distance.",
        "",
    ]

    for direction, order in [
        ("Close in graph / far in text", close_graph_order),
        ("Close in text / far in graph", close_text_order),
    ]:
        lines.append(f"## {direction}")
        lines.append("")
        for rank, pair_k in enumerate(order, start=1):
            i = int(i_idx[pair_k])
            j = int(j_idx[pair_k])
            diff = np.abs(graph_z[i] - graph_z[j])
            top_feat_idx = np.argsort(diff)[::-1][:n_driving]

            lines.append(
                f"### {rank}. `{ids[i]}` ↔ `{ids[j]}`  "
                f"graph={graph_d[pair_k]:.4f}, text={text_d[pair_k]:.4f}"
            )
            lines.append("")
            lines.append("| feature | z-diff | value A | value B |")
            lines.append("|---|---|---|---|")
            for fi in top_feat_idx:
                lines.append(
                    f"| {FEATURE_NAMES[fi]} | {diff[fi]:.3f} | "
                    f"{graph_z[i, fi]:.3f} | {graph_z[j, fi]:.3f} |"
                )
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── Main analysis ─────────────────────────────────────────────────────────────
def run_analysis(out_dir: Path) -> dict[str, Any]:
    """Run the full RSA/Mantel geometry analysis and write outputs."""
    if out_dir.exists():
        # fail-if-exists: timestamped sub-directory is created below, so the base
        # dir may exist but an existing timestamp collision is vanishingly rare.
        pass

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "summary.json"
    disagreement_path = run_dir / "disagreement_cases.md"

    print("Loading and aligning modalities...")
    text_emb, stats_emb, label_emb, ids = _align_modalities()
    n = len(ids)
    print(f"  common transcripts: {n}")
    print(f"  text: {text_emb.shape}, stats: {stats_emb.shape}, label_bag: {label_emb.shape}")

    print("\nBuilding distance matrices...")
    stats_z = _zscore_columns(stats_emb)
    graph_d = _cosine_distance_condensed(stats_z)
    text_d = _cosine_distance_condensed(text_emb)
    label_d = _cosine_distance_condensed(label_emb)
    print(f"  condensed size: {len(graph_d)}")

    # Main Mantel comparisons
    print("\nMantel: text vs graph_stats...")
    text_graph = mantel_test(text_d, graph_d, n_perm=N_PERM_MANTEL, seed=SEED)
    print(f"  r = {text_graph['r']:.4f}, p = {text_graph['p']:.4f}")

    print("\nMantel: text vs label_bag...")
    text_label = mantel_test(text_d, label_d, n_perm=N_PERM_MANTEL, seed=SEED)
    print(f"  r = {text_label['r']:.4f}, p = {text_label['p']:.4f}")

    # Anti-noise gates
    print("\nAnti-noise gate 1: random-embedding null...")
    random_null = random_embedding_null(text_d, n_objects=n, seed=SEED)
    random_null_pass = text_graph["r"] > random_null["pct95_r"]
    print(
        f"  graph-vs-text r = {text_graph['r']:.4f}, "
        f"random null 95th pct = {random_null['pct95_r']:.4f}, pass={random_null_pass}"
    )

    print("\nAnti-noise gate 2: split-half reliability...")
    split_half = split_half_reliability(stats_emb, seed=SEED)
    split_half_pass = split_half["mean_r"] >= SPLIT_HALF_THRESHOLD
    print(f"  mean r = {split_half['mean_r']:.4f}, pass={split_half_pass}")

    # Partial Mantel anchors
    print("\nPartial Mantel anchors...")
    anchors = _load_anchor_labels()
    partial_results: dict[str, Any] = {}
    meaningful_partial = False

    for anchor_name, label_dict in anchors.items():
        subset = [tid for tid in ids if tid in label_dict]
        if len(subset) < 3:
            continue

        idx = [ids.index(tid) for tid in subset]
        text_d_sub = _cosine_distance_condensed(text_emb[idx])
        graph_d_sub = _cosine_distance_condensed(_zscore_columns(stats_emb[idx]))
        labels = np.array([label_dict[tid] for tid in subset])
        anchor_square = (labels[:, None] != labels[None, :]).astype(np.float64)
        anchor_d_sub = squareform(anchor_square, checks=False)

        graph_anchor_given_text = partial_mantel_test(
            graph_d_sub, anchor_d_sub, text_d_sub, n_perm=N_PERM_MANTEL, seed=SEED
        )
        text_anchor_given_graph = partial_mantel_test(
            text_d_sub, anchor_d_sub, graph_d_sub, n_perm=N_PERM_MANTEL, seed=SEED
        )

        partial_results[anchor_name] = {
            "n": len(subset),
            "graph_vs_anchor_given_text": graph_anchor_given_text,
            "text_vs_anchor_given_graph": text_anchor_given_graph,
        }

        sig = graph_anchor_given_text["r"] > 0 and graph_anchor_given_text["p"] < 0.05
        meaningful_partial = meaningful_partial or sig

        print(
            f"  {anchor_name} (n={len(subset)}): "
            f"r(graph,anchor|text)={graph_anchor_given_text['r']:.4f} "
            f"p={graph_anchor_given_text['p']:.4f} | "
            f"r(text,anchor|graph)={text_anchor_given_graph['r']:.4f} "
            f"p={text_anchor_given_graph['p']:.4f}"
        )

    # Interpretability gate
    print("\nExtracting disagreement cases...")
    graph_d_square = squareform(graph_d)
    text_d_square = squareform(text_d)
    _write_disagreement_cases(disagreement_path, ids, stats_z, text_d_square, graph_d_square)
    print(f"  wrote {disagreement_path}")

    # Verdicts
    distinct = text_graph["r"] < DISTINCT_THRESHOLD
    not_noise = random_null_pass and split_half_pass
    meaningful = meaningful_partial
    overall = distinct and not_noise and meaningful

    summary = {
        "timestamp": timestamp,
        "n_transcripts": n,
        "n_perm": N_PERM_MANTEL,
        "seed": SEED,
        "mantel": {
            "text_vs_graph_stats": text_graph,
            "text_vs_label_bag": text_label,
        },
        "anti_noise": {
            "random_embedding_null": random_null,
            "random_embedding_null_pass": random_null_pass,
            "split_half_reliability": split_half,
            "split_half_pass": split_half_pass,
            "not_noise": not_noise,
        },
        "partial_mantel": partial_results,
        "verdict": {
            "distinct": distinct,
            "not_noise": not_noise,
            "meaningful": meaningful,
            "overall": overall,
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote summary to {summary_path}")

    print("\n" + "=" * 60)
    print("Pre-registered verdict")
    print("=" * 60)
    print(f"  DISTINCT       (text↔graph r < {DISTINCT_THRESHOLD}): {distinct}")
    print(f"  NOT_NOISE      (random null + split-half): {not_noise}")
    print(f"  MEANINGFUL     (graph|text partial r > 0, p < 0.05): {meaningful}")
    print(f"  OVERALL        (all three): {overall}")
    print("=" * 60)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSA/Mantel geometry test (v4).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory base (default: {DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args(argv)
    run_analysis(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
