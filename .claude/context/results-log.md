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

### Interviewer confound (key finding)

Initial results showed a **perfect val macro-F1 = 1.0** ceiling effect across all routes. Investigation revealed an interviewer confound: the AI interviewer uses cohort-specific opening scripts ("...using AI in your **creative** work" vs "...using AI in your **scientific** work"), injecting the cohort label into every transcript's `formatted` field. Since the text encoder embeds the full conversation, the classifier trivially separates cohorts by the interviewer's word choices, not the interviewee's cognitive patterns.

**Fix (2026-06-08):** `text_encoder.py` now defaults to **human-only** encoding (`speaker_filter="Human"`), stripping AI turns. All embeddings, models, and results below reflect this fix. The human-only task is genuinely challenging: classify professional cohort from how people describe their work, without the interviewer's label leakage.

### Route 1 — Text-only baseline

| metric | value |
|---|---|
| classifier | LogisticRegression (C=1.0, class_weight=balanced, max_iter=1000) |
| text encoder | all-mpnet-base-v2 (768-dim), human-only turns |
| val macro-F1 | **0.9032** |
| test macro-F1 | **0.8228** |

### Route 2 — Text + graph statistics

| metric | value |
|---|---|
| classifier | LogisticRegression (C=1.0, class_weight=balanced, max_iter=1000) |
| feature dim | 798 (768 text + 30 graph stats) |
| graph stats source | canonicalised graphs via `encoding/graph_stats.py` |
| val macro-F1 | **0.9122** (Δ = +0.0090 vs baseline) |
| test macro-F1 | **0.8390** (Δ = +0.0161 vs baseline) |
| top permutation features | diameter_norm (0.0092), component_ratio (0.0036), mixed_stance_frac (0.0034), max_degree_norm (0.0023) |

### Route 3 — Text + GIN

| metric | value |
|---|---|
| architecture | 2-layer GIN (388-dim input → 256 hidden → 128 output) + MLP head (896 → 256 → 3) |
| node features | 4-dim type one-hot + 384-dim MiniLM label embedding = 388 |
| training config | Adam (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau, early stopping patience=10 |
| epochs run | 20 (early stopped) |
| best epoch | 10 |
| val macro-F1 | **0.9249** (Δ = +0.0217 vs baseline) |
| test macro-F1 | **0.8368** (Δ = +0.0139 vs baseline) |

### Route 2b — Graph statistics ONLY (no text)

| metric | value |
|---|---|
| classifier | LogisticRegression (C=1.0, class_weight=balanced, max_iter=2000) |
| feature dim | 30 (graph statistics only, no text concatenated) |
| val macro-F1 | **0.3796** |
| test macro-F1 | **0.4891** |
| notes | Above chance (0.333) but weak — hand-crafted features capture modest structural signal. Scientists most graph-distinctive (15/18 val). |

### Route 3b — GIN graph embedding ONLY (no text)

| metric | value |
|---|---|
| architecture | 2-layer GIN (388-dim input → 256 hidden → 128 output) + MLP head (128 → 256 → 3) |
| node features | 4-dim type one-hot + 384-dim MiniLM label embedding = 388 |
| training config | Adam (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau, early stopping patience=10 |
| epochs run | 14 (early stopped) |
| best epoch | 4 |
| val macro-F1 | **0.8621** |
| test macro-F1 | **0.8434** (Δ = +0.0206 vs R1 baseline) |
| key finding | **GIN-only BEATS text-only on test set** (+2.06 pp). Graph topology alone is a stronger predictor of professional cohort than 768-dim sentence embeddings of human speech. |

### Route comparison

| route | modalities | val macro-F1 | test macro-F1 | Δ vs R1 (test) |
|---|---|---|---|---|
| Route 1 (text-only) | text | 0.9032 | 0.8228 | — |
| Route 2 (text + stats) | text+stats | 0.9122 | 0.8390 | +0.0161 |
| Route 3 (text + GIN) | text+gin | 0.9249 | 0.8368 | +0.0139 |
| Route 2b (stats only) | stats | 0.3796 | 0.4891 | -0.3338 |
| **Route 3b (GIN only)** | **gin** | **0.8621** | **0.8434** | **+0.0206** |

### Per-class breakdown (test set, all 5 routes)

| class | R1 (text) | R2 (text+stats) | R3 (text+gin) | R2b (stats) | R3b (gin) | support |
|---|---|---|---|---|---|---|
| workforce | 0.9301 | 0.9371 | 0.9347 | 0.6609 | 0.9508 | 150 |
| creatives | 0.6667 | 0.6531 | 0.6809 | 0.3529 | 0.7222 | 19 |
| scientists | 0.8718 | 0.9268 | 0.8947 | 0.4533 | 0.8571 | 19 |

**Key insight:** GIN-only (R3b) outperforms text-only (R1) on workforce AND creatives. Scientists are slightly better with text. The concept graph modality is not just complementary — for some cohorts it's the stronger signal. The text+GIN fusion (R3) underperforms GIN-only, suggesting the simple concatenation fusion is suboptimal.

---

## Alternative targets — Demographic classification (2026-06-09)

Demographic attributes extracted from human-only transcripts via DeepSeek (prompt v2, bead `lxk` + `sfz`). Two classification targets tested.

### AI Adoption — binary (tool_user vs integrated)

n=1,224 (novice=21, power_user=5 dropped). Well-balanced: 602 tool_user / 622 integrated.

| route | test macro-F1 | Δ vs text |
|---|---|---|
| R1 (text) | 0.6736 | — |
| **R2 (text+stats)** | **0.7169** | **+0.0433** |
| R3 (text+GIN) | 0.6319 | -0.0416 |
| R2b (stats only) | 0.6249 | -0.0487 |
| R3b (GIN only) | 0.6682 | -0.0053 |

**Key finding:** Text alone achieves a moderate 0.67 F1 — genuine headroom for graphs. Route 2 (text+stats) adds +4.3pp, the largest graph contribution across any target. GIN underperforms across the board (both R3 and R3b), suggesting the GIN architecture overfits on the subtle tool_user/integrated distinction. R3b (GIN-only, 0.668) nearly matches text-only — graph structure alone captures AI adoption patterns almost as well as language.

Per-class: tool_user 0.73 (R2), integrated 0.70 (R2). No class is systematically harder.

### Career Stage — 3-class (early/mid/late)

n=430 (uncertain=820 dropped). Imbalanced: 105 early / 264 mid / 61 late.

| route | test macro-F1 | Δ vs text |
|---|---|---|
| **R1 (text)** | **0.4501** | — |
| R2 (text+stats) | 0.3161 | -0.1340 |
| R3 (text+GIN) | 0.3539 | -0.0962 |
| R2b (stats only) | 0.2956 | -0.1545 |
| R3b (GIN only) | 0.3632 | -0.0869 |

**Key finding:** Task is hard (chance=0.33, text=0.45). ALL graph routes hurt — adding graph features introduces noise, not signal. The "late" class is essentially unpredictable (F1=0.11-0.29, only 9 test samples). Small sample + class imbalance make this target unreliable. **Verdict: career stage is not viable as a classification target with n=430.** Would need a larger labeled dataset or a semi-supervised approach leveraging the 820 uncertains.

### Cross-target comparison

AI adoption is the better demographic target: well-powered (n=1,224), balanced, moderate text baseline (0.67) with genuine graph upside (+4.3pp from stats). Career stage suffers from small sample and the 65% uncertain rate making supervised classification impractical.

### Per-class breakdown (test set)

| class | Route 1 | Route 2 | Route 3 | support |
|---|---|---|---|---|
| workforce | 0.9301 | 0.9371 | 0.9347 | 150 |
| creatives | 0.6667 | 0.6531 | 0.6809 | 19 |
| scientists | 0.8718 | 0.9268 | 0.8947 | 19 |

The "creatives" cohort is hardest to classify across all routes (F1 ≈ 0.65-0.68), likely due to small sample size (n=19) and high within-cohort diversity (composers, designers, visual artists). Scientists benefit most from graph features (+0.055 F1 from Route 1→2).

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
