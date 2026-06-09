# Deprecated: Task-Supervised GIN (Phase 3)

These files (`model.py`, `train.py`) are **deprecated** as of 2026-06-09.

## Why deprecated

The old GIN was trained end-to-end with classification loss — the encoder
and classifier were a single module (`GraphEncoder`). This conflated graph
representation learning with task-specific learning, making it impossible
to cleanly measure whether graph structure carries complementary signal
independent of text.

## What replaced them

- **`s4_encoding/gnn/autoencoder.py`** — Self-supervised GIN autoencoder.
  Trained on ALL 1,250 graphs with node type reconstruction loss.
  Produces a target-agnostic 128-dim graph encoder.

- **`s4_encoding/gnn/encode.py`** — Frozen encoder inference.
  Loads the trained encoder, produces 128-dim embeddings for any graph set,
  caches to `cache/gin_embeddings.npy`.

- **`s5_classification/fusion/`** — Task-specific classifier zoo.
  Classifiers consume frozen modality embeddings. Only the classifier
  learns per-target — the encoder never sees classification labels.

## Architecture comparison

```
OLD (Phase 3):                    NEW (Phase 5):
graph → GIN → 128-dim ┐           graph → GIN autoencoder → 128-dim (frozen)
                       ├─ concat ─→ classifier                ↓
text → SBERT → 768-dim ┘                           classifier (task-specific)
                       ↑                                      ↑
                BOTH trained with                    ONLY classifier trained
                classification loss                  with classification loss
```

## Archive date

2026-06-09 — Phase 5 implementation begins.
