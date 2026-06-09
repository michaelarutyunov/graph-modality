# ADR 0002: Canonical Vocabulary Lock

- **Status:** Accepted
- **Date:** 2026-06-08
- **Phase:** 2 (Canonicalisation)

## Context

Extraction produces free-text node labels (e.g., "AI reliability and accuracy concerns", "concerns about AI dependability"). To use these as features, labels must be normalised into a shared vocabulary. The canonical vocabulary is the foundation for all downstream encoding — every node type count, graph statistic, and GNN node feature depends on it.

The vocabulary must be locked (immutable) before any encoding or classification begins. Changes after locking require full pipeline re-run.

## Decision

**AgglomerativeClustering with cosine distance, linkage=average, threshold=0.35.**

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | AgglomerativeClustering | Deterministic, no need to specify cluster count |
| Metric | Cosine distance | Robust to label length variation |
| Linkage | Average | Balances single-link chaining vs complete-link over-merging |
| Threshold | 0.35 | Increased from default 0.3 based on cluster quality inspection |
| Embedding model | `all-MiniLM-L6-v2` | Fast, good for short phrase similarity; also used for GNN node features |

### Results

| Entity Type | Free-text Labels | Canonical Labels | Reduction |
|---|---|---|---|
| Construct | 1,331 | ~400 | ~70% |
| Value | 765 | ~250 | ~67% |
| Stance | 1,017 | ~350 | ~66% |
| CSM | 888 | ~270 | ~70% |
| **Total** | **15,753** | **1,271** | **~92%** |

Coverage: 100% — all 18,662 nodes mapped. Mean intra-cluster similarity: 0.72–0.75.

## Rationale

1. **Threshold choice (0.35 vs 0.30):** At 0.30, clusters were too fine-grained — near-synonyms like "AI reliability concerns" and "worries about AI dependability" remained separate. At 0.35, these merge correctly while still distinguishing genuinely different concepts.

2. **Deterministic algorithm:** AgglomerativeClustering produces identical results on re-run — critical for reproducibility.

3. **Coverage guarantee:** Every free-text label maps to exactly one canonical label. No unmapped nodes.

## Consequences

- `canonical_map.json` is locked and must never be modified
- If vocabulary changes are needed, the full pipeline (canonicalisation → encoding → classification) must be re-run
- GNN node features use canonical labels — choice affects graph embedding quality
- The 0.35 threshold trades label richness against noise resistance

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Manual mapping | Infeasible at 15,753 labels |
| DBSCAN | Requires density parameter; less predictable cluster sizes |
| Threshold 0.30 | Over-segmented; near-synonyms remained separate |
| Threshold 0.40 | Under-segmented; distinct concepts merged |
| LLM-based normalisation | Non-deterministic; hard to reproduce |
