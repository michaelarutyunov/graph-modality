# ADR 0003: Target-Agnostic Encoder Architecture

- **Status:** Accepted
- **Date:** 2026-06-09
- **Phase:** 5 (Target-Agnostic Modality Fusion)

## Context

Phase 3 used task-supervised GNN training — the GIN encoder was trained end-to-end with a classification head, jointly optimising graph representation and cohort prediction. This conflates encoding with classification: the resulting graph embeddings encode what the graph predicts (cohort labels), not what the graph IS (topology, node types, connectivity patterns).

This conflation makes it impossible to cleanly measure modality complementarity. If the fused classifier improves, we can't tell whether the graph modality genuinely adds signal or the encoder simply memorised classification labels during pretraining.

## Decision

**Self-supervised GIN autoencoder with frozen inference.** The encoder is trained on ALL 1,250 graphs with a node-type reconstruction objective — no classification labels involved. After training, the encoder is frozen and produces 128-dim graph embeddings. Classifiers consume these fixed embeddings.

### Architecture

```
GIN Encoder (GINEncoder)
├── GINConv Layer 1: 388 → 256 (with BatchNorm)
├── GINConv Layer 2: 256 → 128 (with BatchNorm)
└── global_mean_pool → 128-dim graph embedding

GIN Autoencoder (GINAutoencoder)
├── encoder: GINEncoder (shared)
└── node_type_head: Linear(128 → 4) — reconstruct entity types
```

- **Input:** 388-dim node features (4 type one-hot + 384 MiniLM label embedding)
- **Output:** 128-dim graph embedding (frozen after training)
- **Training objective:** Cross-entropy on node type reconstruction (4 classes)
- **Training data:** All 1,250 graphs (no train/test split — no labels needed)
- **Result:** 100% node type reconstruction accuracy

### Pipeline separation

```
Phase 3 (deprecated):              Phase 5 (current):
┌──────────────┐                   ┌──────────────┐
│ GIN Encoder  │ ← cohort labels   │ GIN Encoder  │ ← node types only
│ + Classifier │                   │ (autoencoder) │
└──────┬───────┘                   └──────┬───────┘
       │ frozen? no                       │ frozen? YES
       ▼                                  ▼
  [text + graph_emb]              [text + graph_emb]
       │                                  │
  single classifier                N classifiers (zoo)
  (conflated)                      (clean measurement)
```

## Rationale

1. **Clean complementarity measurement:** Frozen embeddings encode graph structure, not task-specific features. If classification improves, it's because graph topology genuinely complements text.

2. **Target-agnostic:** Same graph embedding works for any classification target (cohort, AI adoption, career stage) without retraining the encoder.

3. **Architectural precedent:** This is the graph equivalent of what SBERT does for text — a fixed representation that encodes what the data IS, not what it predicts.

4. **Reproducibility:** Deterministic training (seed=42, fixed architecture) produces identical embeddings across runs.

## Consequences

- Old task-supervised GIN moved to `s4_encoding/_archived/` — deprecated but preserved
- The GIN autoencoder achieves 100% node type reconstruction — this means the 128-dim bottleneck preserves entity type information perfectly
- Adding a new modality requires: (1) produce cached .npy + _ids.json, (2) add one line to `build_dataset.py`
- Phase 5 results: graph modality adds small but genuine complementary signal (+0.001–0.006 F1). Node-type reconstruction preserves less cohort-relevant structure than task-supervised pretraining.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Keep task-supervised GIN | Conflates encoding with classification; can't measure complementarity |
| Graph contrastive learning | More complex; no clear benefit over reconstruction for small graphs |
| Denoising autoencoder | Node features are continuous (MiniLM embeddings); masking is less natural than type reconstruction |
| Variational autoencoder | Adds KL divergence complexity; no evidence it improves downstream quality |
