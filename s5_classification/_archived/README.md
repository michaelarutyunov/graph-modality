# Deprecated: Route 3 — Task-Supervised GIN (Phase 3)

`route3.py` is **deprecated** as of 2026-06-09.

## Why deprecated

Route 3 trained a GIN encoder end-to-end with classification loss — the
graph encoder and classifier were a single module. This conflated graph
representation learning with task-specific learning. The embedding was
not a pure modality representation; it was optimized for a specific target.

## What replaced it

- **`s5_classification/fusion/models.py`** — Classifier zoo with 4 architectures
  (single, stacked, gated, late) — all consume frozen modality embeddings.

- **`s5_classification/fusion/run.py`** — Config-driven experiment runner.
  Any architecture × any modality combination × any target.

- **`s4_encoding/gnn/autoencoder.py`** — Self-supervised GIN encoder.
  Produces 128-dim graph embeddings without classification labels.

The separation of encoder (frozen, target-agnostic) and classifier (task-specific)
enables clean measurement of modality complementarity.

## Archive date

2026-06-09 — Phase 5 implementation.
