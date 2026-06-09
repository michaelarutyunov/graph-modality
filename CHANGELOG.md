# Changelog

All notable changes to this project. Phases correspond to the implementation plan in `docs/PLAN.md`.

## Phase 5 — Target-Agnostic Modality Fusion (2026-06-09)

Self-supervised GIN autoencoder, frozen embedding pipeline, classifier zoo, sklearn backend.

### Added
- Self-supervised GIN autoencoder (`s4_encoding/graph_gnn_encoder.py`) — node type reconstruction objective, no classification labels
- Frozen embedding dataset (`s4_encoding/build_dataset.py`) — packages text/stats/GIN as .npz per split/target
- Classifier zoo: 4 PyTorch architectures (single, stacked, gated, late) + sklearn backend (4 classifiers)
- Config-driven experiment runner (`s5_classification/train_run.py`) — 90 experiments via parameter sweep
- Complementarity analysis notebook (`s6_notebooks/05_fusion_analysis.py`)
- sklearn classifier wrapper (`s5_classification/sklearn_classifier.py`)
- Demographic classification targets (AI adoption, career stage) — `.claude/context/demographic-variables.md`

### Changed
- **Directory restructuring:** flattened `s4_encoding/gnn/` → `s4_encoding/`, `s5_classification/fusion/` → `s5_classification/`
- Split monolithic `models.py` into per-classifier files (mlp_single/stacked/gated/late)
- Merged `gnn/autoencoder.py` + `gnn/encode.py` → `graph_gnn_encoder.py`
- Moved `CHARTER.md`, `ENGINEERING.md`, `PLAN.md` → `docs/`

### Removed
- 5 broken Phase 3 evaluation scripts (route3b, demographic_tasks, eval_all_routes, test_evaluation)
- Deprecated task-supervised GIN (moved to `s4_encoding/_archived/`)

## Phase 4 — Structural Analysis (2026-06-09)

RQ2 confirmatory analysis, hypothesis testing.

### Added
- H1–H4 statistical tests in `s6_notebooks/04_structural_analysis.py`
- AI adoption exploratory analysis
- Permutation feature importance (`s5_classification/analysis_feature_importance.py`)
- Stats-only per-class report (`s5_classification/analysis_stats.py`)

### Key findings
- H1 (scientist hub-and-spoke): significant but reversed — scientists have lower C:V ratio
- H2 (creative negative valence): supported — creatives highest negative-stance fraction
- H3 (workforce bipolarity): not significant — ceiling effect from ontology
- H4 (scientist cognitive style): not significant — CSM count ceiling (max 2)

## Phase 3 — Encoding + Classification (2026-06-08)

Three routes evaluated on held-out test set.

### Added
- Text embeddings: 1,250 × 768-dim (`all-mpnet-base-v2`, human-only turns)
- Graph statistics: 30-dim NetworkX-derived features
- Train/val/test split: 875/187/188 stratified (seed=42)
- Route 1 (text-only): test macro-F1 = 0.823
- Route 2 (text+stats): test macro-F1 = 0.839 (Δ = +0.016)
- Route 3 (text+GIN): test macro-F1 = 0.837 (Δ = +0.014)

### Fixed
- Interviewer confound: AI turns stripped (cohort-specific openings leaked labels)

## Phase 2 — Canonicalisation (2026-06-08)

Locked canonical vocabulary from free-text labels.

### Added
- 1,271 canonical labels from 15,753 free-text labels across 4 entity types
- AgglomerativeClustering, cosine distance, threshold=0.35
- 100% coverage: all 18,662 nodes mapped
- `canonical_map.json` locked and immutable

## Phase 1 — Extraction (2026-06-07)

Graph extraction from interview transcripts.

### Added
- 1,250 transcripts extracted with DeepSeek (deepseek-chat, OpenAI-compatible endpoint, JSON mode)
- Prompt v3 (two-shot examples: workforce + scientist)
- Speaker tagger handling three prefixes (Assistant/AI/User)
- Graph validator enforcing 6 structural constraints
- 3-model comparison (Claude/DeepSeek/Agnes)

### Key metrics
- 0 failures, 0.3% violation rate
- Mean 14.9 nodes / 13.6 edges per graph

## Phase 0 — Environment Setup (2026-06-07)

### Added
- `pyproject.toml` with CPU-only torch index
- Full directory tree (`s1_data/`, `s2_extraction/`, `s3_canonicalisation/`, `s4_encoding/`, `s5_classification/`, `s6_notebooks/`, `tests/`)
- `.gitignore` with project-specific rules
- `.env.example` template for API keys
- 5 specialist agents (schema-guardian, extraction, canonicalisation, encoding, analysis)
