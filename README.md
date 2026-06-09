# cdt-graph-modality

> Research prototype testing whether concept graphs extracted from interview transcripts constitute a structurally distinct modality for consumer digital twin (CDT) representation.

## What it does

This project extracts structured concept graphs from the [Anthropic Interviewer dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer) (1,250 transcripts across workforce, creative, and scientist cohorts), then tests whether graph topology carries predictive signal that text embeddings alone cannot recover.

The pipeline is linear: **download → extract → canonicalise → encode → classify → analyse**. Each stage is cache-aware — expensive operations (API extraction, SBERT encoding, GNN training) run once and write to disk.

## Architecture

The system uses **target-agnostic modality encoders**: each modality (text, graph statistics, GIN graph embedding) produces a frozen vector representation. Downstream classifiers consume these fixed embeddings. This separation allows clean measurement of whether graph modalities add complementary signal over text alone.

| Modality | Encoder | Dimensions |
|---|---|---|
| Text | SBERT (`all-mpnet-base-v2`) | 768 |
| Graph statistics | NetworkX-derived features | 30 |
| Graph structure | GIN autoencoder (self-supervised) | 128 |

Classifiers: 4 PyTorch architectures (single, stacked, gated, late) + sklearn backend (logistic regression, SVM, random forest, gradient boosting).

## Quick start

```bash
# Setup
uv sync

# Pipeline (each step is idempotent and caches results)
uv run python s1_data/download.py                     # Download dataset
uv run python s2_extraction/extractor.py              # Extract graphs (requires API key)
uv run python s3_canonicalisation/clusterer.py        # Build canonical vocabulary
uv run python s3_canonicalisation/apply_canonical.py  # Apply canonical labels
uv run python s4_encoding/text_encoder.py             # Text embeddings (768-dim)
uv run python s4_encoding/graph_stats_encoder.py      # Graph statistics (30-dim)
uv run python s4_encoding/graph_gnn_encoder.py        # Train GIN autoencoder
uv run python s4_encoding/graph_gnn_encoder.py --encode  # Frozen GIN inference (128-dim)
uv run python s4_encoding/build_dataset.py            # Package as .npz
uv run python s5_classification/train_run.py          # Run experiments

# Tests
uv run pytest
```

## Key results

| Route | Modalities | Test macro-F1 |
|---|---|---|
| Text-only baseline | text | 0.823 |
| Text + graph statistics | text + stats | 0.839 |
| Text + GIN embedding | text + graph | 0.837 |
| GIN-only | graph | 0.843 |

GIN-only beats text-only — graph structure alone is a strong signal for cohort classification.

## Documentation

| Document | Description |
|---|---|
| [`docs/CHARTER.md`](docs/CHARTER.md) | Research questions, ontology, evaluation philosophy |
| [`docs/ENGINEERING.md`](docs/ENGINEERING.md) | Full technical specification |
| [`docs/PLAN.md`](docs/PLAN.md) | Implementation plan with phase history |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records |

## Stack

Python 3.11+ · PyTorch (CPU-only) · PyTorch Geometric · Polars · scikit-learn · NetworkX · Sentence-Transformers · Marimo · uv
