"""Build the canonical vocabulary by clustering free-text node labels.

Loads all extracted graphs, collects unique labels per entity type, embeds
them with ``all-MiniLM-L6-v2``, clusters with agglomerative clustering
(cosine distance, average linkage), and selects the label closest to each
cluster centroid as the canonical label.

Output: ``canonicalisation/canonical_map.json`` — a dict mapping every
free-text label to its canonical equivalent.

Usage:
    uv run python canonicalisation/clusterer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

GRAPH_DIR = Path("data/graphs/free_text")
OUT_PATH = Path("canonicalisation/canonical_map.json")

ENTITY_TYPES = ["Construct", "Value", "Stance", "CognitiveStyleMarker"]
MODEL_NAME = "all-MiniLM-L6-v2"
DISTANCE_THRESHOLD = 0.35
RANDOM_SEED = 42


def load_all_labels(graph_dir: Path = GRAPH_DIR) -> dict[str, set[str]]:
    """Collect all unique free-text labels per entity type from the graph corpus.

    Returns:
        Dict mapping entity type → set of unique label strings.
    """
    labels_by_type: dict[str, set[str]] = {t: set() for t in ENTITY_TYPES}

    for path in sorted(graph_dir.glob("*.json")):
        g = json.loads(path.read_text(encoding="utf-8"))
        for node in g.get("nodes", []):
            ntype = node.get("type", "")
            label = node.get("label", "")
            if ntype in labels_by_type and label:
                labels_by_type[ntype].add(label)

    return labels_by_type


def cluster_labels(
    labels: list[str],
    model: SentenceTransformer,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> dict[str, str]:
    """Cluster a list of labels and return a mapping to canonical labels.

    Uses cosine distance with agglomerative clustering.  The canonical
    label for each cluster is the one closest to the cluster centroid.

    Args:
        labels: List of free-text label strings.
        model: A loaded SentenceTransformer.
        distance_threshold: Agglomerative clustering distance threshold.

    Returns:
        Dict mapping each original label → canonical label.
    """
    if not labels:
        return {}

    # Identical labels always map to themselves — no embedding needed
    unique_labels = list(dict.fromkeys(labels))  # preserve order, deduplicate
    if len(unique_labels) == 1:
        return {unique_labels[0]: unique_labels[0]}

    embeddings = model.encode(unique_labels, normalize_embeddings=True)

    # With normalized embeddings, cosine distance = 1 - cosine_similarity
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    cluster_ids = clustering.fit_predict(embeddings)

    # For each cluster, pick the label closest to the centroid
    label_map: dict[str, str] = {}
    for cluster_id in set(cluster_ids):
        mask = cluster_ids == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_labels = [unique_labels[i] for i, m in enumerate(mask) if m]

        centroid = cluster_embeddings.mean(axis=0)
        # Normalize centroid for cosine comparison
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        distances = 1.0 - np.dot(cluster_embeddings, centroid_norm)
        canonical_idx = distances.argmin()
        canonical = cluster_labels[canonical_idx]

        for label in cluster_labels:
            label_map[label] = canonical

    return label_map


def build_canonical_map(
    graph_dir: Path = GRAPH_DIR,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> dict[str, dict[str, str]]:
    """Build the full canonical map for all entity types.

    Returns:
        Nested dict: ``{entity_type: {free_text_label: canonical_label}}``.
    """
    print(f"Loading labels from {graph_dir}...")
    labels_by_type = load_all_labels(graph_dir)

    for etype in ENTITY_TYPES:
        n_unique = len(labels_by_type[etype])
        print(f"  {etype}: {n_unique} unique labels")

    print(f"\nLoading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    canonical_map: dict[str, dict[str, str]] = {}
    for etype in ENTITY_TYPES:
        labels = sorted(labels_by_type[etype])
        print(f"\nClustering {etype} ({len(labels)} labels, threshold={distance_threshold})...")
        mapping = cluster_labels(labels, model, distance_threshold)
        n_clusters = len(set(mapping.values()))
        print(f"  → {n_clusters} clusters")
        canonical_map[etype] = mapping

        # Print sample clusters for inspection
        _print_sample_clusters(mapping, etype, max_clusters=5)

    return canonical_map


def _print_sample_clusters(
    mapping: dict[str, str], entity_type: str, max_clusters: int = 5
) -> None:
    """Print a sample of clusters for manual inspection."""
    # Group by canonical label
    clusters: dict[str, list[str]] = {}
    for label, canonical in mapping.items():
        clusters.setdefault(canonical, []).append(label)

    print(f"  Sample clusters for {entity_type}:")
    for i, (canonical, members) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))):
        if i >= max_clusters:
            break
        print(f"    [{canonical}]")
        for m in members[:5]:
            marker = " ← canonical" if m == canonical else ""
            print(f"      {m}{marker}")
        if len(members) > 5:
            print(f"      ... and {len(members) - 5} more")
        print()


def main() -> None:
    canonical_map = build_canonical_map()

    # Write output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(canonical_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCanonical map written to {OUT_PATH}")

    # Summary
    print("\nVocabulary sizes:")
    for etype in ENTITY_TYPES:
        n_clusters = len(set(canonical_map[etype].values()))
        n_labels = len(canonical_map[etype])
        print(f"  {etype}: {n_clusters} canonical from {n_labels} free-text labels")


if __name__ == "__main__":
    main()
