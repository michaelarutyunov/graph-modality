# results-log.md

> **Purpose:** canonical record of all experiment results. Write once per experiment, never overwrite. This is the single source of truth for classification outcomes, route comparisons, and statistical tests.

---

## Phase 1 — Extraction

| metric | value |
|---|---|
| transcripts processed | 1,250 |
| extraction model | deepseek-chat (OpenAI-compatible endpoint, JSON mode) |
| prompt version | v3 (two-shot examples) |
| successful extractions | 1,250 (100%) |
| failed extractions | 0 |
| mean nodes per graph | 14.9 |
| mean edges per graph | 13.6 |
| violation rate | 0.3% (4 graphs) |

### Model Comparison (10 fixed transcripts)

| model | successful | mean nodes | mean edges | violations | notes |
|---|---|---|---|---|---|
| Claude (claude-sonnet-4-6) | 10/10 | 15.1 | 15.8 | 0 | Selected as baseline reference |
| DeepSeek (deepseek-v4-pro, Anthropic endpoint) | 0/10 | — | — | — | Failed: thinking blocks + JSON truncation |
| DeepSeek (deepseek-chat, OpenAI endpoint + JSON mode) | 3/3 (validation) | 14-18 | 13-19 | 0 | Correct config: works reliably |
| Agnes (agnes-2.0-flash) | 3/3 (validation) | 11-15 | 10-13 | 0-1 | Leaner graphs, works reliably |

---

## Phase 2 — Canonicalisation

| metric | value |
|---|---|
| free-text labels | 15,753 |
| canonical vocabulary | 1,271 (Construct 1,331, Value 765, Stance 1,017, CSM 888) |
| distance threshold | 0.35 cosine |
| embedding model | all-MiniLM-L6-v2 |
| coverage | 100% (0 unmapped labels out of 18,662 nodes) |
| mean intra-cluster similarity | 0.72-0.75 |
| stability (100-transcript holdout) | 19.3% reassignment rate |
| canonical_map.json locked | 2026-06-08 |

---

## Phase 3 — Encoding + Classification

### Split

| metric | value |
|---|---|
| total transcripts | 1,250 |
| split ratios | 70/15/15 stratified |
| seed | 42 |
| train / val / test | 875 / 187 / 188 |
| class distribution (train) | workforce=700, creatives=87, scientists=88 |
| class distribution (val) | workforce=150, creatives=19, scientists=18 |
| class distribution (test) | workforce=150, creatives=19, scientists=19 |

### Ceiling effect (key finding)

The text-only baseline (768-dim `all-mpnet-base-v2` embeddings → logistic regression) achieves **perfect val macro-F1 = 1.0**. The three cohorts (workforce / creatives / scientists) are trivially separable from language alone. Routes 2 and 3 cannot beat Route 1 on F1 — they can only match it. The research question shifts from "do graphs add predictive power?" to "do graph features provide equivalent signal with fewer dimensions, better robustness, or interpretable structure?" The test set evaluation (bead `ico`) is critical: the ceiling may not hold on held-out data.

### Route 1 — Text-only baseline

| metric | value |
|---|---|
| classifier | LogisticRegression (C=1.0, class_weight=balanced, max_iter=1000) |
| text encoder | all-mpnet-base-v2 (768-dim) |
| val macro-F1 | **1.0000** |
| test macro-F1 | — (held out) |

### Route 2 — Text + graph statistics

| metric | value |
|---|---|
| classifier | LogisticRegression (C=1.0, class_weight=balanced, max_iter=1000) |
| feature dim | 798 (768 text + 30 graph stats) |
| graph stats source | canonicalised graphs via `encoding/graph_stats.py` |
| val macro-F1 | **1.0000** (Δ = +0.0000 vs baseline) |
| test macro-F1 | — (held out) |
| permutation importance | All zero (model already perfect on val; cannot improve) |
| top permutation features | N/A (ceiling effect) |

### Route 3 — Text + GIN

| metric | value |
|---|---|
| architecture | 2-layer GIN (388-dim input → 256 hidden → 128 output) + MLP head (896 → 256 → 3) |
| node features | 4-dim type one-hot + 384-dim MiniLM label embedding = 388 |
| training config | Adam (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau, early stopping patience=10 |
| epochs run | 26 (early stopped) |
| best epoch | 16 |
| val macro-F1 | **0.9797** (Δ = -0.0203 vs baseline) |
| test macro-F1 | — (held out) |

### Route comparison

| route | val macro-F1 | test macro-F1 | Δ vs baseline |
|---|---|---|---|
| Baseline (text-only) | 1.0000 | — | — |
| Route 2 (text + stats) | 1.0000 | — | +0.0000 |
| Route 3 (text + GIN) | 0.9797 | — | -0.0203 |

---

## Phase 4 — Analysis

> *To be populated during Phase 4.*

### Structural analysis (RQ2)

| hypothesis | test statistic | p-value | significant | interpretation |
|---|---|---|---|---|
| H1 — Scientist hub-and-spoke | — | — | — | — |
| H2 — Creative negative valence | — | — | — | — |
| H3 — Workforce bipolarity | — | — | — | — |
| H4 — Scientist cognitive style | — | — | — | — |
