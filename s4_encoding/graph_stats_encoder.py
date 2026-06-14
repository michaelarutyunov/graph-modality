"""Route 2 feature engineering: hand-crafted graph statistics.

Produces a 30-dimensional feature vector per transcript from canonicalised
concept graphs.  Uses NetworkX for graph topology metrics.

Usage:
    uv run python encoding/graph_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

ENTITY_TYPES = ["Construct", "Value", "Stance", "CognitiveStyleMarker"]
VALENCES = ["positive", "negative", "mixed", "ambivalent"]

CACHE_DIR = Path("cache")
STATS_CACHE = CACHE_DIR / "graph_stats.npy"
ID_CACHE = CACHE_DIR / "graph_stats_ids.json"

FEATURE_DIM = 30


def graph_to_features(graph_data: dict, graph_path: Path | None = None) -> np.ndarray:
    """Compute the 30-dimensional feature vector for a single graph.

    Args:
        graph_data: Parsed graph JSON dict.
        graph_path: Source file path (unused; accepted for caller convenience).

    Returns:
        Float32 array of shape ``(30,)``.
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Build NetworkX graph
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for e in edges:
        G.add_edge(e["source"], e["target"], relation=e.get("relation", "?"))

    n_total = max(len(nodes), 1)
    n_edges = G.number_of_edges()

    # ── structural features (7) ────────────────────────────────────
    density = nx.density(G)
    n_components = nx.number_weakly_connected_components(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0.0
    try:
        diameter = nx.diameter(G.to_undirected())
    except (nx.NetworkXError, nx.exception.NetworkXException):
        diameter = -1.0

    # ── node type distribution (6) ─────────────────────────────────
    type_counts = {t: 0 for t in ENTITY_TYPES}
    for n in nodes:
        ntype = n.get("type", "")
        if ntype in type_counts:
            type_counts[ntype] += 1

    n_construct = type_counts["Construct"]
    n_value = type_counts["Value"]
    n_stance = type_counts["Stance"]
    n_csm = type_counts["CognitiveStyleMarker"]

    construct_value_ratio = n_construct / max(n_value, 1)
    stance_construct_ratio = n_stance / max(n_construct, 1)

    # ── construct quality (3) ──────────────────────────────────────
    constructs = [n for n in nodes if n.get("type") == "Construct"]
    bipolarity_score = (
        np.mean([1.0 if n.get("bipolarity_complete") else 0.5 for n in constructs])
        if constructs
        else 0.0
    )
    construct_degrees = [G.degree(n["id"]) for n in constructs]
    mean_construct_degree = np.mean(construct_degrees) if construct_degrees else 0.0
    max_construct_degree = max(construct_degrees) if construct_degrees else 0.0

    # ── stance valence (8) ─────────────────────────────────────────
    stances = [n for n in nodes if n.get("type") == "Stance"]
    valence_counts = {v: 0 for v in VALENCES}
    for s in stances:
        v = s.get("valence", "ambivalent")
        if v in valence_counts:
            valence_counts[v] += 1

    n_stance_safe = max(n_stance, 1)
    dominant_valence = max(valence_counts, key=valence_counts.__getitem__) if stances else "absent"
    valence_onehot = [1.0 if dominant_valence == v else 0.0 for v in VALENCES]
    valence_onehot.append(1.0 if dominant_valence == "absent" else 0.0)

    # ── centrality (3) ─────────────────────────────────────────────
    try:
        betweenness = nx.betweenness_centrality(G)
        bc_values = list(betweenness.values())
        max_bc = max(bc_values) if bc_values else 0.0
        mean_bc = np.mean(bc_values) if bc_values else 0.0
        value_nodes = [n["id"] for n in nodes if n.get("type") == "Value"]
        max_value_bc = max(betweenness.get(v, 0.0) for v in value_nodes) if value_nodes else 0.0
    except Exception:
        max_bc = mean_bc = max_value_bc = 0.0

    # ── cognitive style (2) ────────────────────────────────────────
    csm_present = float(n_csm > 0)
    csm_count_clipped = min(n_csm, 2) / 2.0

    features = np.array(
        [
            # structural (7)
            n_total / 15.0,
            n_edges / 20.0,
            density,
            n_components / n_total,
            avg_degree / 5.0,
            max_degree / 10.0,
            (diameter + 1) / 10.0,
            # node type distribution (6)
            n_construct / n_total,
            n_value / n_total,
            n_stance / n_total,
            n_csm / n_total,
            construct_value_ratio / 5.0,
            stance_construct_ratio / 3.0,
            # construct quality (3)
            bipolarity_score,
            mean_construct_degree / 5.0,
            max_construct_degree / 10.0,
            # stance valence (8)
            valence_counts["positive"] / n_stance_safe,
            valence_counts["negative"] / n_stance_safe,
            valence_counts["mixed"] / n_stance_safe,
            valence_counts["ambivalent"] / n_stance_safe,
            *valence_onehot,  # 5 values
            # centrality (3)
            max_bc,
            mean_bc,
            max_value_bc,
            # cognitive style (2)
            csm_present,
            csm_count_clipped,
        ],
        dtype=np.float32,
    )

    return features


def compute_all_stats(
    graph_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute graph statistics for all cached graphs.

    Returns:
        (stats_matrix, transcript_ids) where stats_matrix has shape (N, 36).
    """
    if graph_dir is None:
        # v4 corpus (P6.6): all Phase 6 tests run on v4_think only. Stats use node
        # types + valence + structure (label-independent), so canonical==free_text here.
        graph_dir = Path("s1_data/graphs/v4_think/canonical")

    if STATS_CACHE.exists() and ID_CACHE.exists():
        print("loading cached graph statistics")
        stats = np.load(STATS_CACHE)
        ids = json.loads(ID_CACHE.read_text(encoding="utf-8"))
        return stats, ids

    graph_paths = sorted(graph_dir.glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError(f"no graph files found in {graph_dir}")

    stats_list: list[np.ndarray] = []
    ids: list[str] = []

    for gp in graph_paths:
        g = json.loads(gp.read_text(encoding="utf-8"))
        fv = graph_to_features(g, gp)
        stats_list.append(fv)
        ids.append(g["transcript_id"])

    stats = np.stack(stats_list)  # shape: (N, 36)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(STATS_CACHE, stats)
    ID_CACHE.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"cached {len(ids)} graph stat vectors → {STATS_CACHE}")

    return stats, ids


if __name__ == "__main__":
    stats, ids = compute_all_stats()
    print(f"Done. Shape: {stats.shape}, IDs: {len(ids)}")
    print("Sample features (first graph):")
    print(f"  structural: {stats[0, :7]}")
    print(f"  type dist:  {stats[0, 7:13]}")
    print(f"  construct:  {stats[0, 13:16]}")
