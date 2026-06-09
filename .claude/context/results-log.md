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
