# results-log.md

> **Purpose:** canonical record of all experiment results. Write once per experiment, never overwrite. This is the single source of truth for classification outcomes, route comparisons, and statistical tests.

> **2026-06-10 update:** under repeated-seed evaluation (Method-Review Phase 1, see bottom),
> the Phase 5 headline fusion gain is **not supported** for the GIN graph embedding —
> text+graph and text+stats+graph deltas on the primary `ai_adoption` target have CIs
> that include 0. The only delta surviving the pre-registered criterion is **text+stats**
> (+0.0142, CI excludes 0). Treat the Phase 5 "Architecture Comparison Summary" and
> "Synthesis" below as superseded point estimates, not confirmed effects.

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
| graph stats source | canonicalised graphs via `s4_encoding/graph_stats_encoder.py` |
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

## Phase 4 — Structural Analysis (RQ2)

*Analysis date: 2026-06-09. All tests run on 1,250 canonical graphs. Omnibus: Kruskal-Wallis.
Post-hoc: pairwise Mann-Whitney U with Bonferroni correction (α=0.05/3=0.0167). Effect sizes:
eta-squared (η²) for 3-group comparisons, Cliff's delta (δ) for pairwise.*

### H1–H4 Confirmatory Results

| hypothesis | Kruskal-Wallis H | p-value | η² | significant (α=0.05) | interpretation |
|---|---|---|---|---|---|
| H1 — Scientist hub-and-spoke (C:V ratio) | 23.902 | 0.000006 | 0.024 | **yes** | Significant but **opposite direction**: scientists have *lower* C:V ratio (1.44) than workforce (1.60). Scientists have fewer constructs per terminal value — values are more concentrated, not more differentiated. The "hub-and-spoke" pattern holds structurally (fewer constructs radiating from each value) but not in the predicted direction. |
| H2 — Creative negative valence | 11.129 | 0.003831 | 0.009 | **yes** | **Supported.** Creatives have highest negative-valence stance fraction (0.396) vs scientists (0.341) and workforce (0.385). Post-hoc: creatives > scientists (p=0.002, δ=0.179), creatives > workforce (p=0.104, n.s.). The dual satisfaction/anxiety pattern is structurally visible. |
| H3 — Workforce bipolarity | 5.525 | 0.063130 | 0.004 | no | **Not significant.** Bipolarity is near ceiling across all cohorts: workforce 0.998, scientists 0.998, creatives 0.986. The ontology constraint (both poles required) and strong prompt compliance leave almost no variance for differentiation. Smallest η² of all hypotheses. |
| H4 — Scientist cognitive style (CSM count) | 0.500 | 0.778645 | 0.001 | no | **Not significant.** 99.8% of graphs have exactly 2 CSMs (the ontology ceiling). Only 2 graphs (both workforce) have 1 CSM. The CSM ceiling constraint makes count-based differentiation impossible. Verification-oriented CSM fraction shows no significant cohort difference (p=0.221). |

### H1 Detail: Construct:Value Ratio

| cohort | mean C:V ratio | mean n_construct | mean n_value | median C:V |
|---|---|---|---|---|
| workforce | 1.604 | 10.89 | 7.51 | 1.500 |
| creatives | 1.572 | 10.78 | 7.55 | 1.429 |
| scientists | 1.439 | 9.70 | 7.22 | 1.333 |

**Interpretation:** Scientists have the *lowest* C:V ratio — they extract fewer constructs relative to values. This is the opposite of the pre-registered prediction. Rather than scientists having more differentiated evaluation frames, they appear to have more *concentrated* value structures: fewer constructs serving each terminal value. The significant pairwise differences are scientists < workforce (p=0.000003, δ=-0.182) and scientists < creatives (p=0.026, δ=-0.093). Workforce and creatives do not differ significantly (p=1.0).

Possible explanation: The scientist cohort (n=125) discusses AI adoption in narrower epistemic terms (data integrity, verification, rigour), producing fewer distinct constructs per value. Workforce respondents (n=1,000) have broader, more varied AI interactions, producing more differentiated evaluation frames.

### H2 Detail: Negative Stance Valence

| cohort | mean neg_frac | mean n_stance | mean n_negative | pairwise |
|---|---|---|---|---|
| workforce | 0.385 | 11.35 | 4.43 | — |
| creatives | 0.396 | 11.46 | 4.66 | > scientists (p=0.002) |
| scientists | 0.341 | 10.76 | 3.71 | < creatives (p=0.002), < workforce (p=0.042) |

**Interpretation:** The creative cohort's higher negative-valence fraction is consistent with the dual satisfaction/anxiety pattern documented in the Anthropic research. Creatives report 97% productivity gains but also pervasive identity anxiety — this ambivalence shows up structurally as a higher proportion of negatively-valenced stances. Scientists are the least negative, consistent with a more instrumental relationship to AI (tool for verification, not a threat to identity).

### H3 Detail: Bipolarity Completeness

| cohort | mean complete_frac | mean bipolarity_score | % 100% complete |
|---|---|---|---|
| workforce | 0.998 | 0.999 | 99.7% |
| creatives | 0.986 | 0.993 | 96.8% |
| scientists | 0.998 | 0.999 | 99.2% |

**Interpretation:** The near-ceiling bipolarity scores (all >0.98) reflect strong prompt compliance and the ontology's requirement that constructs have both poles. Creatives show slightly lower completeness (0.986 vs 0.998), consistent with H3's directional prediction of more ambivalent/unresolved constructs, but the difference does not reach significance after correction. The metric is effectively saturated — future work should use a continuous bipolarity measure (e.g., pole specificity rating) rather than binary complete/incomplete.

### H4 Detail: CSM Prevalence

| cohort | mean CSM count | % with CSM | mean verify_frac |
|---|---|---|---|
| workforce | 1.998 | 100% | 0.439 |
| creatives | 2.000 | 100% | 0.400 |
| scientists | 2.000 | 100% | 0.402 |

**Interpretation:** The CSM ceiling (max 2 per transcript) is universally enforced by the extractor — only 2 of 1,250 graphs have fewer than 2 CSMs. This makes H4's count-based prediction untestable. The verification-oriented fraction is slightly higher in workforce (0.439) than scientists (0.402), opposite to the predicted direction, but not significant (Kruskal-Wallis p=0.221). Scientists do not show higher verification-orientation in their CSM subtypes.

### Methodological Note: Ceiling Effects

Two of four pre-registered hypotheses (H3, H4) were undermined by ceiling effects from the extraction ontology itself. When the prompt enforces "both poles required" and "max 2 CSMs," those constraints become the dominant source of variance (or lack thereof). Future pre-registration should verify metric variance on pilot data before locking hypotheses.

### Exploratory: Structural Analysis by AI Adoption

Same metrics regrouped by AI adoption (`tool_user` n=602 vs `integrated` n=622; novice=21, power_user=5 excluded). Mann-Whitney U with Cliff's delta.

| metric | mean tool_user | mean integrated | Cliff's δ | p-value | significant |
|---|---|---|---|---|---|
| Construct:Value ratio | 1.556 | 1.612 | -0.107 | 0.000985 | **yes** |
| Graph size (nodes) | 14.85 | 14.82 | 0.016 | 0.610 | no |
| Edge count | 13.48 | 13.62 | -0.010 | 0.758 | no |
| Negative stance fraction | 0.386 | 0.376 | 0.037 | 0.256 | no |
| Bipolarity completeness | 0.997 | 0.997 | -0.005 | 0.371 | no |
| CSM count | 1.998 | 1.998 | -0.000 | 0.982 | no |
| Verification CSM fraction | 0.437 | 0.430 | 0.008 | 0.755 | no |
| Conflict prevalence | 0.246 | 0.243 | 0.006 | 0.839 | no |

**Key finding:** Integrated users have a significantly *higher* Construct:Value ratio (1.61 vs 1.56, p=0.001, δ=-0.107). Integrated users articulate more differentiated evaluation frames — more constructs per terminal value — than tool_users. This is the only metric that differentiates AI adoption groups, and it's the same metric that differentiates cohorts (H1). The effect size is small (δ=-0.107) but robust.

No other metric reaches significance. AI adoption does not produce dramatically different graph topology beyond the C:V ratio shift — the structural fingerprint of AI adoption is subtler than the structural fingerprint of professional cohort.

### Cross-Target Comparison

| metric | cohort η² | cohort p | AI adoption |δ| | AI adoption p | stronger signal |
|---|---|---|---|---|---|---|
| C:V ratio | 0.024 | 0.000006 | 0.107 | 0.000985 | cohort |
| Neg stance fraction | 0.009 | 0.003831 | 0.037 | 0.256 | cohort |
| Bipolarity completeness | 0.004 | 0.063130 | 0.005 | 0.371 | neither (ceiling) |
| CSM count | 0.001 | 0.778645 | 0.000 | 0.982 | neither (ceiling) |

**Verdict:** Professional cohort is a stronger structural differentiator than AI adoption level. Where structural differentiation exists (C:V ratio, negative valence), it is more pronounced across cohorts than across AI adoption groups. This is consistent with the R1 classification results: cohort classification achieves F1 ≈ 0.82-0.84, while AI adoption classification peaks at F1 ≈ 0.72.

### Phase 4 Synthesis

**Which hypotheses were supported?**
- H2 (creative negative valence): **Supported** — clear structural fingerprint of the dual satisfaction/anxiety pattern.
- H1 (scientist hub-and-spoke): **Significant but reversed** — scientists have lower, not higher, C:V ratio. The structural pattern exists but in the opposite direction.
- H3 (workforce bipolarity): **Not supported** — ceiling effect from ontology constraint.
- H4 (scientist cognitive style): **Not supported** — ceiling effect from CSM max constraint.

**Does graph topology differentiate by cohort?**
Yes, modestly. H1 (C:V ratio, η²=0.024) and H2 (negative valence, η²=0.009) show significant cohort differences with small-to-medium effect sizes. However, both significant effects are small in absolute terms. Graph topology carries genuine but limited cohort signal — consistent with R2b (stats-only classification: F1=0.49, above chance but far below text-only).

**Does graph topology differentiate by AI adoption?**
Only through C:V ratio (δ=-0.107, p=0.001). All other metrics are non-significant. AI adoption leaves a fainter structural trace than professional cohort.

**What does this mean for the modality hypothesis?**
The concept graph modality carries some structural signal that differentiates groups (RQ2), but the signal is modest and concentrated in a few metrics (C:V ratio, stance valence). The classification results (Phase 3) showed that graph structure alone can classify cohort at F1=0.49 (stats) to F1=0.84 (GIN), substantially above chance. But the interpretable, hypothesis-driven metrics (H1-H4) capture only a fraction of whatever signal the GIN is exploiting. This gap between interpretable metrics and GNN performance suggests that the GIN is picking up on structural patterns not captured by hand-crafted features — possibly subgraph motifs, edge-type interactions, or label-semantic patterns that the simple count-based metrics miss.

**Open questions:**
1. What is the GIN actually learning? The gap between interpretable metrics (η²≈0.01-0.02) and GIN-only classification (F1=0.84) is large. Attribution techniques (GNNExplainer, integrated gradients) could probe what structural patterns the GIN exploits.
2. Would a larger scientist cohort (n=125 currently) change H1 direction? The reversed H1 finding might reflect the small scientist sample or genuine differences in how scientists discuss AI in short interviews.
3. Can we design continuous metrics that avoid ceiling effects? Binary bipolarity and capped CSM counts are inherently low-variance.

---

## Phase 5 — Target-Agnostic Modality Fusion

*Experiment date: 2026-06-09. 42 experiments (21 per target). Encoders frozen — only classifiers learn per-task.*

### Architecture Principle

The key design change from Phase 3: **encoders produce fixed vectors; only classifiers learn per-target.** The GIN autoencoder is self-supervised (node type reconstruction on all 1,250 graphs), never sees classification labels. This is the graph equivalent of SBERT for text — a frozen representation encoding what the data IS, not what it predicts. The old task-supervised GIN (`s4_encoding/_archived/model.py`, `s4_encoding/_archived/train.py`) conflated encoding with classification, making it impossible to cleanly measure modality complementarity.

### GIN Autoencoder

| metric | value |
|---|---|
| training graphs | 1,250 (all, no split — self-supervised) |
| architecture | 2-layer GINConv (388→256→128) + global mean pool |
| decoder head | Linear(128, 4) — node type reconstruction |
| loss | cross-entropy on 4-class node type prediction |
| training config | Adam (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau |
| epochs run | 59 (early stopped at patience=15) |
| best epoch | 44 |
| best loss | 0.0002 |
| **final node type accuracy** | **1.0000** (100%) |
| node features | 4-dim type one-hot + 384-dim MiniLM label embedding = 388 |
| encoder weights | `cache/gin_encoder_canonical.pt` (1.07 MB) |
| embedding dim | 128 |
| embedding cache | `cache/gin_embeddings_canonical.npy` (1,250 × 128) |

**Training note:** The autoencoder converges rapidly — 95.2% node type accuracy after 1 epoch, 99.99% by epoch 2, reaching 100% at epoch 4. The node type reconstruction task is easy (node features already contain the type one-hot), but the 128-dim bottleneck forces the encoder to compress graph topology into a compact vector. The decoder is discarded after training; only the encoder is saved.

### Frozen Embedding Dataset

Package: `cache/modality_dataset/`

| file | shapes | contents |
|---|---|---|
| `cohort_train.npz` | (875,) | text (768), stats (30), graph (128), labels (0/1/2) |
| `cohort_val.npz` | (187,) | text (768), stats (30), graph (128), labels (0/1/2) |
| `cohort_test.npz` | (188,) | text (768), stats (30), graph (128), labels (0/1/2) |
| `ai_adoption_train.npz` | (860,) | text (768), stats (30), graph (128), labels (0/1) |
| `ai_adoption_val.npz` | (181,) | text (768), stats (30), graph (128), labels (0/1) |
| `ai_adoption_test.npz` | (183,) | text (768), stats (30), graph (128), labels (0/1) |

Label distribution (AI adoption): 602 tool_user / 622 integrated (26 excluded: 21 novice, 5 power_user).
Label distribution (cohort): workforce=700/150/150, creatives=87/19/19, scientists=88/18/19 (train/val/test).

### Classifier Architectures

Four architectures in `s5_classification/models.py` (split into single/stacked/gated/late), all consuming frozen modality embeddings:

| architecture | description | parameters (text+graph) |
|---|---|---|
| **Single** | MLP on one modality — baselines | ~200K |
| **Stacked** | Concat all modalities → MLP (old R2/R3 pattern) | ~230K |
| **Gated** | Learn per-example softmax attention over modalities → MLP | ~360K |
| **Late** | Separate MLP per modality → average logits (ensemble) | ~230K |

### AI Adoption Results (binary, n=1,224)

**Best model: `late_text-graph` — test F1 = 0.6446**

#### Single-modality baselines

| architecture | modality | test macro-F1 | val F1 | epochs |
|---|---|---|---|---|
| single | text | 0.6118 | 0.7174 | 17 |
| stacked | text | 0.6391 | 0.7182 | 20 |
| gated | text | 0.6222 | 0.7123 | 18 |
| late | text | 0.6174 | 0.7014 | 16 |
| single | stats | 0.5964 | 0.6098 | 23 |
| stacked | stats | 0.6295 | 0.6060 | 33 |
| gated | stats | 0.5949 | 0.6090 | 24 |
| late | stats | 0.5898 | 0.6181 | 19 |
| single | graph | 0.5832 | 0.5635 | 11 |
| stacked | graph | 0.5929 | 0.6016 | 34 |
| gated | graph | 0.6065 | 0.5520 | 11 |
| late | graph | 0.6011 | 0.5844 | 19 |

#### Fusion results (text + graph modalities)

| architecture | modalities | test macro-F1 | Δ vs best text-only | val F1 |
|---|---|---|---|---|
| stacked | text+stats | 0.6444 | +0.0053 | 0.7458 |
| **late** | **text+graph** | **0.6446** | **+0.0055** | **0.7235** |
| stacked | text | 0.6391 | 0.0 | 0.7182 |
| stacked | text+stats+graph | 0.6385 | -0.0006 | 0.7458 |
| gated | text+stats | 0.6337 | -0.0054 | 0.7179 |
| late | text+stats | 0.6335 | -0.0056 | 0.7233 |
| gated | text+graph | 0.6332 | -0.0059 | 0.7177 |
| late | text+stats+graph | 0.6332 | -0.0059 | 0.7341 |
| gated | text+stats+graph | 0.6222 | -0.0169 | 0.7231 |
| stacked | text+graph | 0.6124 | -0.0267 | 0.7125 |

**Pattern:** Graph modality contribution is small but positive for AI adoption. Late fusion (ensemble) of text+graph yields the best result (0.6446), but the improvement over text-only stacked (0.6391) is only +0.0055. The GRAPH-UNIQUE fraction (examples where text+graph is correct but text-only is wrong) ranges 3-8%. The graph signal is complementary but weak — adding graph features helps on a small subset of edge cases.

### Cohort Results (3-class, n=1,250)

**Best model: `gated_text-graph` — test F1 = 0.8629**

#### Single-modality baselines

| architecture | modality | test macro-F1 | val F1 | notes |
|---|---|---|---|---|
| stacked | text | 0.7584 | 0.9111 | |
| gated | text | 0.7584 | 0.9111 | |
| single | text | 0.7584 | 0.9111 | |
| **late** | **text** | **0.8616** | **0.9065** | best text-only |
| *all stats-only* | stats | 0.2959 | 0.2967 | *majority-class baseline* |
| *all graph-only* | graph | 0.2959 | 0.2967 | *majority-class baseline* |

**Critical finding:** Stats-only and graph-only models collapse to majority-class prediction (F1=0.296 ≈ always predicting workforce, 80% of data). This is the single biggest difference from Phase 3 — in Phase 3, GIN-only achieved 0.8434 on cohort. The frozen autoencoder embedding, stripped of classification signal, cannot discriminate cohorts alone. This is **by design** — it confirms the encoder is target-agnostic. But it also means the autoencoder objective (node type reconstruction) may not preserve cohort-relevant structural patterns.

#### Fusion results (text + graph modalities)

| architecture | modalities | test macro-F1 | Δ vs best text-only | val F1 |
|---|---|---|---|---|
| **gated** | **text+graph** | **0.8629** | **+0.0013** | **0.9494** |
| late | text | 0.8616 | 0.0 | 0.9065 |
| gated | text+stats+graph | 0.8269 | -0.0347 | 0.9292 |
| stacked | text+graph | 0.8186 | -0.0430 | 0.9150 |
| late | text+stats | 0.8158 | -0.0458 | 0.9022 |
| stacked | text+stats+graph | 0.8088 | -0.0528 | 0.8943 |
| late | text+graph | 0.8049 | -0.0567 | 0.9111 |
| stacked | text+stats | 0.7947 | -0.0669 | 0.9111 |
| late | text+stats+graph | 0.7777 | -0.0839 | 0.8974 |

**Pattern:** Graph contribution is extremely small (+0.0013) and only with gated fusion. Most fusion combinations *hurt* performance vs late-fusion text-only. The gated mechanism's softmax attention appears to help suppress noisy graph features — but the benefit is negligible. For cohort classification with frozen embeddings, text dominates completely.

### Per-Class Breakdown (best fusion models)

#### AI Adoption — late_text-graph

| class | text-only acc | fusion acc | Δ | support |
|---|---|---|---|---|
| tool_user (0) | 0.652 | 0.652 | 0.000 | 89 |
| integrated (1) | 0.637 | 0.637 | 0.000 | 94 |

Both classes gain zero from graph in the best fusion model — the late fusion ensemble essentially weights text ≈1.0 and graph ≈0.0.

#### Cohort — gated_text-graph

| class | text-only acc | fusion acc | Δ | support |
|---|---|---|---|---|
| workforce (0) | 0.960 | 0.947 | -0.013 | 150 |
| creatives (1) | 0.579 | 0.632 | +0.053 | 19 |
| scientists (2) | 0.737 | 0.842 | +0.105 | 19 |

The gated fusion helps **creatives (+5.3pp) and scientists (+10.5pp)** while slightly hurting workforce (-1.3pp). This is the complementarity story: graph structure helps disambiguate the minority classes (creatives, scientists) where text alone struggles. The gate mechanism learns to attend more to graph features for structurally distinctive cohorts.

### Complementarity Analysis

For each fusion experiment, we compute the 2×2 complementarity matrix: text-only correct/wrong vs. (text+graph) correct/wrong. The **GRAPH-UNIQUE** cell counts examples where fusion succeeds but text-only fails.

#### AI Adoption — GRAPH-UNIQUE fractions

| architecture | modalities | GRAPH-UNIQUE | TEXT-UNIQUE | OVERLAP | NEITHER |
|---|---|---|---|---|---|
| late | text+graph | 0.082 | 0.082 | 0.563 | 0.273 |
| stacked | text+graph | 0.055 | 0.082 | 0.557 | 0.306 |
| gated | text+graph | 0.077 | 0.066 | 0.557 | 0.301 |
| late | text+stats | 0.071 | 0.060 | 0.563 | 0.306 |
| stacked | text+stats | 0.082 | 0.082 | 0.563 | 0.273 |

For AI adoption, GRAPH-UNIQUE ranges 5.5-8.2% — about 10-15 examples out of 183 in the test set. The graph modality contributes complementary signal on a small but real subset. Late fusion captures the most GRAPH-UNIQUE examples (8.2%).

#### Cohort — GRAPH-UNIQUE fractions

| architecture | modalities | GRAPH-UNIQUE | TEXT-UNIQUE | OVERLAP | NEITHER |
|---|---|---|---|---|---|
| gated | text+graph | 0.117 | 0.090 | 0.745 | 0.048 |
| stacked | text+graph | 0.069 | 0.069 | 0.750 | 0.112 |
| late | text+graph | 0.069 | 0.080 | 0.734 | 0.117 |

For cohort, gated fusion achieves the highest GRAPH-UNIQUE fraction (11.7% — 22 of 188 test examples). This is consistent with gated text+graph being the best cohort model. The gating mechanism captures ~5pp more GRAPH-UNIQUE examples than stacked or late fusion. These are primarily creatives and scientists — the minority classes where text alone struggles.

### Comparison with Phase 3 (Conflated GIN)

| aspect | Phase 3 (conflated) | Phase 5 (frozen) | interpretation |
|---|---|---|---|
| GIN-only cohort F1 | 0.8434 | 0.2959 | Frozen encoder loses all cohort signal — confirms encoder is target-agnostic |
| Text+GIN cohort F1 | 0.8368 | 0.8629 | Frozen architecture *exceeds* conflated — better classifier design compensates |
| Graph contribution | mixed (sometimes hurts) | small but positive | Clean separation reveals genuine but weak complementarity |
| GIN training | task-supervised (cohort labels) | self-supervised (node types) | Frozen encoder cannot exploit label leakage |

**Key insight:** The Phase 3 GIN-only result (0.8434) was driven by task-supervised training — the encoder learned cohort-specific features. The frozen autoencoder (0.2959) proves this: without classification labels, graph structure alone cannot separate cohorts above chance. This validates the target-agnostic design — but also reveals that node type reconstruction may be too weak a self-supervised objective to preserve cohort-relevant structural patterns.

### Gate Questions — Answered

**(a) With target-agnostic encoders, does graph modality add signal over text alone?**

**Yes, but the effect is small.** For AI adoption: +0.006 F1 (late fusion). For cohort: +0.001 F1 (gated fusion). The GRAPH-UNIQUE complementarity cell is populated (3-12% of test examples), confirming that frozen GIN embeddings capture structural patterns not present in text. However, the magnitude is much smaller than Phase 3 suggested — the conflated GIN's strong performance was driven by task-supervised training, not by inherent graph complementarity.

**(b) Does gated fusion outperform stacked concatenation?**

**For cohort, yes.** Gated fusion captures 11.7% GRAPH-UNIQUE vs 6.9% for stacked — the attention mechanism helps suppress noisy graph features and amplify structural signal for minority classes. For AI adoption, **late fusion (ensemble) is best** (8.2% GRAPH-UNIQUE). The optimal architecture depends on target: gated for multi-class where graph helps specific classes, late for balanced binary where ensemble averaging reduces variance.

**(c) Does the answer differ by target (AI adoption vs cohort)?**

**Yes, dramatically.** For AI adoption, graph contribution is small but consistent across architectures (+0.003-0.006 F1). For cohort, graph-only models collapse to chance (0.296 F1), and only gated fusion extracts any benefit (+0.001 F1). The graph modality carries less inherent cohort signal after freezing — the autoencoder's node-type objective preserves structural patterns relevant to AI adoption (node type distributions, connectivity) better than patterns relevant to professional cohort (C:V ratios, stance valence distributions). This is a limitation of the node-type reconstruction objective.

### Architecture Comparison Summary

| target | best architecture | best modalities | test F1 | GRAPH-UNIQUE |
|---|---|---|---|---|
| AI adoption | late | text+graph | 0.6446 | 0.082 |
| AI adoption | stacked | text+stats | 0.6444 | 0.082 |
| Cohort | gated | text+graph | 0.8629 | 0.117 |
| Cohort | late | text | 0.8616 | — |

### Phase 5 Synthesis

**What we learned:**

1. **Target-agnostic encoding works as designed.** The frozen autoencoder produces a pure structural representation, stripped of task-specific signal. Graph-only classification collapses to chance, confirming no label leakage.

2. **Graph complementarity is real but weaker than Phase 3 suggested.** The conflated GIN's strong performance (0.84 F1) was due to task-supervised training, not inherent graph structure. With frozen encoders, graph adds 0.001-0.006 F1 — genuine but modest.

3. **The self-supervised objective matters.** Node type reconstruction (100% accuracy) may be too easy — the encoder learns to preserve entity type information but not necessarily the structural patterns (C:V ratios, subgraph motifs, edge-type interactions) that differentiate cohorts. A harder pretext task (contrastive learning, graph isomorphism, motif prediction) might preserve more cohort-relevant structure.

4. **Fusion architecture matters for cohort but not AI adoption.** Gated fusion helps cohort (minority classes benefit from selective graph attention) but late fusion (simple ensemble) is equally good for AI adoption. The architecture choice is target-dependent.

5. **The gap between handcrafted metrics (Phase 4) and GNN performance persists.** Phase 4 found interpretable structural differences (H1 C:V ratio η²=0.024, H2 negative valence η²=0.009) but Phase 5's graph-only GIN collapses to chance. The GIN is not capturing the same patterns as the handcrafted metrics — and without task supervision, it captures almost nothing at all.

**Recommendations:**
- For CDT modality architecture: use frozen SBERT for text; graph modality with node-type autoencoder adds marginal value. Consider contrastive pre-training as a stronger self-supervised objective.
- Future work: graph contrastive learning (Option C from design doc) — corrupt graph structure and train encoder to distinguish original vs corrupted. This forces the encoder to learn structure-preserving representations and may preserve more cohort-relevant patterns than node type prediction.
- Attribution: apply GNNExplainer to the *conflated* GIN to understand what structural patterns it exploited, then design self-supervised objectives that preserve those patterns.

---

## Method-Review Phase 1 — Repeated-Evaluation Re-scoring (2026-06-10)

The Phase 5 headline numbers above were a **single fixed split (seed=42)**, selected as
the max over a 90-config sweep — exactly the practice `docs/METHOD_REVIEW.md` flagged as
unsupported. P1.3 (`s5_classification/repeated_run.py`) re-ran the full
`build_sweep() + build_sklearn_sweep()` matrix (90 configs) across the 10 frozen
protocol seeds (`docs/method-review/00-evaluation-protocol.md`), 900 runs total.
Selection per `(target, modality_combo)` is now the config with the highest **mean
validation macro-F1 across seeds** — never the test-set max.

Raw per-run rows: `results/method_review/phase1/runs.jsonl` (gitignored).
Aggregate: `results/method_review/phase1/summary.json` (gitignored).

### Selected configs (validation-selected, 10-seed test mean ± 95% CI)

| target | modality combo | arch/backend | mean val F1 | mean test F1 | 95% CI |
|---|---|---|---|---|---|
| ai_adoption | text | late/torch | 0.6749 | 0.6444 | [0.624, 0.665] |
| ai_adoption | stats | gated/torch | 0.6127 | 0.5680 | [0.537, 0.599] |
| ai_adoption | graph | stacked/torch | 0.5888 | 0.5934 | [0.571, 0.616] |
| ai_adoption | text+stats | late/torch | 0.6911 | 0.6587 | [0.638, 0.680] |
| ai_adoption | text+graph | gated/torch | 0.6798 | 0.6358 | [0.607, 0.665] |
| ai_adoption | text+stats+graph | late/torch | 0.6814 | 0.6507 | [0.629, 0.672] |
| cohort | text | gated/torch | 0.8885 | 0.8788 | [0.853, 0.904] |
| cohort | stats | logistic/sklearn | 0.3948 | 0.3985 | [0.373, 0.424] |
| cohort | graph | svm/sklearn | 0.4865 | 0.4893 | [0.458, 0.521] |
| cohort | text+stats | gated/torch | 0.8871 | 0.8851 | [0.859, 0.911] |
| cohort | text+graph | gated/torch | 0.9003 | 0.8763 | [0.854, 0.899] |
| cohort | text+stats+graph | stacked/torch | 0.8889 | 0.8787 | [0.848, 0.910] |

### Paired text-vs-fusion deltas (primary target = ai_adoption)

| target | fusion combo | mean Δ | 95% CI | real_effect (CI excl. 0 & Δ≥0.01) | McNemar p (seed=0) |
|---|---|---|---|---|---|
| ai_adoption | text+stats | **+0.0142** | [0.0031, 0.0253] | **True** | 0.581 |
| ai_adoption | text+graph | -0.0087 | [-0.0208, 0.0035] | False | 0.824 |
| ai_adoption | text+stats+graph | +0.0062 | [-0.0092, 0.0217] | False | 1.000 |
| cohort | text+stats | +0.0063 | [-0.0031, 0.0157] | False | 1.000 |
| cohort | text+graph | -0.0025 | [-0.0198, 0.0149] | False | 0.688 |
| cohort | text+stats+graph | -0.0001 | [-0.0269, 0.0267] | False | 1.000 |

### Chance baseline (majority-class macro-F1, mean over 10 seeds)

- ai_adoption: 0.3367 (CI ≈ [0.3366, 0.3368])
- cohort: 0.2959 (constant — class proportions stable across seeds)

### Verdict (per the frozen protocol, §"What 'supported' means downstream")

**Do any text-vs-fusion deltas survive repeated evaluation (CI excludes 0 AND mean Δ ≥ +0.01)?**

- **ai_adoption (primary): YES for text+stats only.** Δ=+0.0142, CI=[0.0031, 0.0253] —
  excludes 0 and clears the +0.01 threshold. text+graph (Δ=-0.0087, CI=[-0.0208, 0.0035])
  and text+stats+graph (Δ=+0.0062, CI=[-0.0092, 0.0217]) do NOT survive — both CIs
  include 0.
- **cohort (secondary, confounded): NO for any combo.** All three deltas
  (text+stats +0.0063, text+graph -0.0025, text+stats+graph -0.0001) have CIs that
  include 0.

**Fusion does NOT add signal over text on the primary target (ai_adoption) by the §7
criterion** — except via graph-stats. The only "real" effect is **text+stats**
(Δ=+0.0142, CI excludes 0, ≥+0.01) — i.e. **graph-stats**, not the GIN embedding, is the
complementary modality. Both
text+graph and text+stats+graph deltas have CIs that include 0 on ai_adoption — under
repeated evaluation, the previously reported text+graph gains do not survive.

The cohort target (sanity-only, confounded) shows no real effects in either direction —
consistent with all combos hovering near the gated/text baseline (0.876-0.885), within
noise of each other.

**Implication for Phase 2 (`n70`):** the kill-criterion question — does GIN topology
(full vs structure-only vs label-bag) add signal — is now sharper: full-GIN fusion
(text+graph) is *not* a real effect on ai_adoption even before the label-bag ablation.
Phase 2 should determine whether structure-only GIN clears the chance baseline at all;
if not, combined with this result, the topology hypothesis is dead on this dataset and
graph-stats (not GNN-derived graph structure) is the dataset's actual complementary
signal.

---

## Method-Review Phase 2 — Graph-vs-Labels Disentanglement (2026-06-10)

The decisive ablation (review concerns #2, #4; epic kill-criterion). Five single-modality
embeddings were each probed with a fresh, fixed-capacity logistic regression
(`s5_classification/ablation_probe.py::probe_variant`) across the same 10 protocol seeds
and split procedure as Phase 1. Results: `results/method_review/phase2/summary.json`.

| Variant | Source | Isolates |
|---|---|---|
| text | SBERT 768-d | raw-transcript text |
| (c) label_bag | mean-pooled MiniLM label embeddings, no edges (P2.2) | pooled label semantics |
| (b) structure_only | GIN trained on type one-hot + degree only (P2.1) | topology only |
| (a) full_gin | GIN trained on type + label embeddings (Phase 5 default) | topology + label semantics |
| (a') masked_gin | GIN with masked node-type objective (P2.3) | topology + labels, non-trivial objective |

### Single-modality probe results (mean test macro-F1 ± 95% CI, 10 seeds)

| target | variant | mean F1 | 95% CI |
|---|---|---|---|
| ai_adoption | text | 0.6369 | [0.618, 0.655] |
| ai_adoption | label_bag (c) | 0.6814 | [0.658, 0.705] |
| ai_adoption | structure_only (b) | 0.5658 | [0.536, 0.596] |
| ai_adoption | full_gin (a) | 0.5714 | [0.556, 0.587] |
| ai_adoption | masked_gin (a') | 0.5787 | [0.544, 0.613] |
| cohort | text | 0.8593 | [0.849, 0.870] |
| cohort | label_bag (c) | 0.7740 | [0.745, 0.803] |
| cohort | structure_only (b) | 0.4230 | [0.394, 0.452] |
| cohort | full_gin (a) | 0.4700 | [0.437, 0.503] |
| cohort | masked_gin (a') | 0.5694 | [0.555, 0.584] |

### Chance baseline (majority-class macro-F1, mean over 10 seeds)

- ai_adoption: 0.3367
- cohort: 0.2959

### Paired deltas

| target | comparison | mean Δ | 95% CI | real_effect |
|---|---|---|---|---|
| ai_adoption | (a) full_gin − (c) label_bag | **-0.1100** | [-0.135, -0.085] | False (negative) |
| ai_adoption | (b) structure_only − chance | **+0.2291** | [0.199, 0.260] | **True** |
| ai_adoption | (a') masked_gin − (a) full_gin | +0.0073 | [-0.029, 0.044] | False |
| cohort | (a) full_gin − (c) label_bag | **-0.3040** | [-0.357, -0.251] | False (negative) |
| cohort | (b) structure_only − chance | **+0.1272** | [0.098, 0.156] | **True** |
| cohort | (a') masked_gin − (a) full_gin | **+0.0994** | [0.071, 0.128] | **True** |

(`real_effect` here follows the protocol's directional definition: CI excludes 0 AND
mean Δ ≥ +0.01. A significant *negative* delta — as for `(a)−(c)` on both targets — is
reported as `real_effect=False` by that definition but is itself a strong, CI-excludes-0
finding, called out explicitly below.)

### Kill-criterion verdict

The epic's pre-registered kill-criterion is: *if `(a)≈(c)` (delta CI includes 0) AND
structure-only is at chance, the topology hypothesis is dead.*

**Neither half of that conjunction holds, but not in the direction the criterion
anticipated:**

1. **`(a) ≈ (c)` is FALSE — but `(a) < (c)`, significantly.** On both targets, full-GIN
   embeddings (topology + labels) score *significantly lower* than the label-bag baseline
   (topology-free pooled labels): ai_adoption Δ=-0.110 [-0.135,-0.085], cohort
   Δ=-0.304 [-0.357,-0.251], both CIs tightly excluding 0. The GIN is not "approximately
   the label-bag" — message-passing through the autoencoder's node-type-reconstruction
   objective actively *degrades* the pooled-label signal relative to simply averaging it.

2. **`(b) − chance` is FALSE for the kill-criterion (structure is NOT at chance) — it's a
   real positive effect on both targets.** structure_only clears chance by +0.229
   (ai_adoption) and +0.127 (cohort), both CIs excluding 0 and ≥+0.01. Topology alone (4-d
   type one-hot + degree, 128-d after the GIN) carries real classification signal — just
   much less than text or even label_bag.

**Verdict: the kill-criterion as literally specified is NOT met (both legs point away from
"topology is dead"), but the epic's underlying concern is sharpened, not resolved
favourably for the current GIN.** Topology *does* carry signal above chance (point 2), so
"graph modality = pooled-label semantics in disguise" is too strong. But the *current*
full-GIN encoding (node-type-reconstruction objective, type+label input) is **strictly
worse than its own label-bag input** on both targets — the GIN is actively destroying
information relative to a trivial mean-pool. This means the Phase 5 / Phase 1 "full GIN"
embeddings used in the production fusion sweep are a poor representation of both topology
and labels simultaneously.

3. **`(a') − (a)` (masked objective vs default):** no real effect on ai_adoption
   (CI includes 0), but a real **positive** effect on cohort (+0.0994, CI=[0.071,0.128]).
   The masked node-type objective (P2.3, held-out reconstruction acc=0.978) produces a
   meaningfully better encoder for cohort than the trivial pass-through objective — though
   still below label_bag and text on cohort, and still not better than label_bag on
   ai_adoption.

### Implication for the deferred Phases 3-5 epic

- The single biggest lever is **not** more data or a different fusion architecture — it's
  the **GIN's self-supervised objective and/or input features**. The masked objective
  (a') already recovers some of the gap on cohort; further work (e.g. training (a') with
  structure_only inputs, or a contrastive objective) is the highest-leverage next step
  before any new extraction.
- Given finding 1, Phase 1's "text+stats is the only real fusion gain" (text+graph CI
  includes 0) is now explained: the `graph` modality fed into Phase 1's fusion sweep was
  `full_gin`, which this bead shows underperforms even a training-free label-bag — so its
  near-zero fusion contribution is consistent, not surprising.

### Null-Ladder Edge Test (v4) --- PASS

**Date:** 2026-06-12
**Target:** cohort (workforce/creatives/scientists)
**Protocol:** 10-seed frozen CI (42-51)

**Arms:**
- Null: 11-dim bag-of-types histogram
  (node type + edge type frequencies + mean degree) -> LogisticRegression
- Alternative: GINEConv(typed) 128-dim frozen embeddings -> LogisticRegression

**Results:**
- GINEConv mean macro-F1: 0.3335 +/- 0.0360
- Histogram mean macro-F1: 0.3048 +/- 0.0435
- Mean delta: +0.0287
- 95% CI: [+0.0094, +0.0480]
- CI excludes 0: True
- Mean delta >= +0.01: True

**Per-seed:**
| Seed | GINE F1 | Hist F1 | delta |
|------|---------|---------|-------|
| 42 | 0.3141 | 0.2895 | +0.0246 |
| 43 | 0.3417 | 0.2907 | +0.0510 |
| 44 | 0.3649 | 0.3421 | +0.0229 |
| 45 | 0.4002 | 0.3625 | +0.0378 |
| 46 | 0.3348 | 0.3118 | +0.0230 |
| 47 | 0.3522 | 0.3285 | +0.0237 |
| 48 | 0.2688 | 0.2730 | -0.0043 |
| 49 | 0.3403 | 0.3575 | -0.0172 |
| 50 | 0.3132 | 0.2636 | +0.0496 |
| 51 | 0.3049 | 0.2289 | +0.0761 |

**Verdict:** PASS
**Interpretation:** Typed relational structure carries signal beyond
bag-of-types -- edge-type-aware GNN beats the no-wiring null under
the 10-seed CI protocol. The mean delta (+0.0287 macro-F1) exceeds
the +0.01 threshold and the 95% CI [+0.0094, +0.0480] excludes zero.
8 of 10 seeds show positive delta. Both arms operate near chance
(~0.30-0.33, 3-class) with structure_only features -- the absolute
performance is low because label semantics are excluded by design,
but the RELATIVE advantage of typed topology is reliable.

---

### Null-Ladder Edge Test (v4, ai_adoption) --- FAIL

**Date:** 2026-06-12
**Target:** ai_adoption (tool_user/integrated)
**Protocol:** 10-seed frozen CI (42-51)

**Arms:**
- Null: 11-dim bag-of-types histogram
  (node type + edge type frequencies + mean degree) -> LogisticRegression
- Alternative: GINEConv(typed) 128-dim frozen embeddings -> LogisticRegression

**Results:**
- Chance baseline: 0.3367
- GINEConv mean macro-F1: 0.5373
- Histogram mean macro-F1: 0.5345
- Mean delta: +0.0028
- 95% CI: [-0.0172, +0.0228]
- CI excludes 0: False
- Mean delta >= +0.01: False

**Per-seed:**
| Seed | GINE F1 | Hist F1 | delta |
|------|---------|---------|-------|
| 42 | 0.5163 | 0.5163 | +0.0000 |
| 43 | 0.5156 | 0.5634 | -0.0478 |
| 44 | 0.5217 | 0.4999 | +0.0218 |
| 45 | 0.5591 | 0.5486 | +0.0106 |
| 46 | 0.5921 | 0.6350 | -0.0429 |
| 47 | 0.5047 | 0.4833 | +0.0214 |
| 48 | 0.5080 | 0.5037 | +0.0044 |
| 49 | 0.5426 | 0.5357 | +0.0069 |
| 50 | 0.5434 | 0.5317 | +0.0117 |
| 51 | 0.5691 | 0.5272 | +0.0420 |

**Verdict:** FAIL
**Interpretation:** Typed relational structure does NOT reliably beat the bag-of-types null under the CI criterion -- the edge-type hypothesis is not supported.

---

### Phase 2.6 — v4 structure_only > chance

**Date:** 2026-06-12 | **Protocol:** 10-seed frozen CI (42-51)

**Target: ai_adoption (tool_user/integrated)**

- Mean F1: 0.5389 +/- 0.0376
- Chance: 0.3367
- Mean delta: +0.2022
- 95% CI: [+0.1753, +0.2290]
- CI excludes 0: True
- delta >= +0.01: True
- **Verdict: PASS**

| Seed | F1 | delta vs chance |
|------|----|----|
| 42 | 0.4818 | +0.1451 |
| 43 | 0.5787 | +0.2420 |
| 44 | 0.5270 | +0.1903 |
| 45 | 0.5535 | +0.2168 |
| 46 | 0.5868 | +0.2501 |
| 47 | 0.4999 | +0.1632 |
| 48 | 0.5159 | +0.1792 |
| 49 | 0.5255 | +0.1888 |
| 50 | 0.5923 | +0.2556 |
| 51 | 0.5270 | +0.1903 |

**Target: cohort (workforce/creatives/scientists)**

- Mean F1: 0.3382 +/- 0.0424
- Chance: 0.2959
- Mean delta: +0.0423
- 95% CI: [+0.0120, +0.0726]
- CI excludes 0: True
- delta >= +0.01: True
- **Verdict: PASS**

| Seed | F1 | delta vs chance |
|------|----|----|
| 42 | 0.3574 | +0.0615 |
| 43 | 0.3045 | +0.0086 |
| 44 | 0.3386 | +0.0427 |
| 45 | 0.4443 | +0.1484 |
| 46 | 0.3131 | +0.0172 |
| 47 | 0.3308 | +0.0349 |
| 48 | 0.3104 | +0.0145 |
| 49 | 0.3572 | +0.0613 |
| 50 | 0.3256 | +0.0297 |
| 51 | 0.3002 | +0.0043 |

---
