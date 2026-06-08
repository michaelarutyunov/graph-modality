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

> *To be populated during Phase 3.*

### Route 1 — Text-only baseline

| metric | value |
|---|---|
| classifier | — |
| val macro-F1 | — |
| test macro-F1 | — |

### Route 2 — Text + graph statistics

| metric | value |
|---|---|
| classifier | — |
| val macro-F1 | — |
| test macro-F1 | — |
| top permutation features | — |

### Route 3 — Text + GIN

| metric | value |
|---|---|
| architecture | — |
| val macro-F1 | — |
| test macro-F1 | — |
| best epoch | — |

### Route comparison

| route | val macro-F1 | test macro-F1 | Δ vs baseline |
|---|---|---|---|
| Baseline (text-only) | — | — | — |
| Route 2 (text + stats) | — | — | — |
| Route 3 (text + GIN) | — | — | — |

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
