# External Methodological Review Prompt

You are an expert in research methodology, computational social science, and machine learning evaluation. I am asking you to perform an **adversarial methodological review** of a research project. Your job is to find weaknesses, threats to validity, and indefensible choices — not to confirm that the work is sound.

---

## Research Question

**Does concept graph topology extracted from interview transcripts carry predictive signal that text embeddings alone cannot recover?**

The broader claim is that concept graphs constitute a "structurally distinct modality" for representing interview respondents — one that captures relational cognitive structure (which concepts connect to which, how, and with what valence) that flat text embeddings miss.

The domain is consumer digital twins (CDTs) — AI agents conditioned on interview data to simulate a person's responses. The project tests whether adding graph representations to text representations improves classification of respondents.

---

## Dataset

**Source:** Anthropic Interviewer dataset (publicly available on HuggingFace).
- 1,250 interview transcripts across 3 professional cohorts: workforce (1,000), creatives (125), scientists (125).
- Each transcript is a multi-turn conversation between an AI interviewer and a human respondent about AI usage in their professional work.
- Transcripts average ~14k characters, ~10 human turns.

**Classification targets:**
1. **Cohort** (3-class: workforce/creatives/scientists) — n=1,250
2. **AI adoption** (binary: tool_user/integrated) — n=1,224 (26 excluded as novice/power_user)

---

## Methodology

### Stage 1: Graph Extraction
- An LLM (DeepSeek) parses each transcript into a structured JSON concept graph.
- The ontology has 4 entity types (Construct, Value, Stance, CognitiveStyleMarker) and 4 relation types (SERVES, EXPRESSED_VIA, MODULATED_BY, CONFLICTS_WITH).
- Constructs are bipolar (positive pole + negative pole). Stances have valence (positive/negative/mixed/ambivalent). CognitiveStyleMarkers are capped at 2 per transcript.
- Only human turns are extracted; AI turns used as context only.
- Every node requires a grounding_span (verbatim quote from the transcript).

### Stage 2: Canonicalisation
- 15,753 free-text node labels are clustered into 1,271 canonical labels using AgglomerativeClustering (cosine distance, threshold=0.35, linkage=average).
- Label embeddings from `all-MiniLM-L6-v2`.
- The canonical map is locked before any downstream modelling — it is never modified between experiments.

### Stage 3: Encoding (Target-Agnostic)
Three modality encoders produce frozen vector representations:

| Modality | Encoder | Dimensions | Training data |
|---|---|---|---|
| Text | SBERT (`all-mpnet-base-v2`) | 768 | Pre-trained, frozen. Human-only turns. |
| Graph statistics | NetworkX-derived structural features | 30 | Deterministic, no training. |
| Graph structure | GIN autoencoder (self-supervised) | 128 | ALL 1,250 graphs. Node-type reconstruction. |

**Critical design choice:** Encoders never see classification labels. Only downstream classifiers are task-specific. This means the same graph embedding is used for cohort classification and AI adoption classification without retraining the encoder.

**GIN autoencoder details:** 2-layer GIN with BatchNorm, 388-dim node features (4 type one-hot + 384 MiniLM label embedding), trained to reconstruct node entity types from 128-dim graph embedding. Achieves 100% node-type reconstruction accuracy on all 1,250 graphs. No train/test split for encoder training (no labels needed).

### Stage 4: Classification
- Fixed stratified 70/15/15 split (seed=42): train=875, val=187, test=188.
- Test set held out until final evaluation — no hyperparameter decisions on test performance.
- Classifiers: 4 PyTorch MLP architectures (single-modality, stacked concat, gated attention, late ensemble) + sklearn backend (logistic regression, SVM, random forest, gradient boosting).
- All consume the same frozen embeddings from Stage 3.

### Stage 5: Analysis
- **RQ1 (classification):** Does adding graph features improve over text alone?
- **RQ2 (structural):** Do cohorts differ in graph topology? Pre-registered hypotheses H1–H4 tested with Kruskal-Wallis + Mann-Whitney U with Bonferroni correction, effect sizes as eta-squared and Cliff's delta.
- Permutation feature importance on graph statistics to identify discriminative features.

---

## Key Results

### RQ1 — Classification

| Route | Modalities | Test macro-F1 | Δ vs text-only |
|---|---|---|---|
| Text-only | text (768d) | 0.823 | — |
| Text + graph stats | text + stats (30d) | 0.839 | +0.016 |
| Text + GIN | text + graph (128d) | 0.837 | +0.014 |
| GIN-only | graph (128d) | **0.843** | +0.020 |
| Graph stats-only | stats (30d) | ~0.80 | — |

**Phase 5 results (target-agnostic encoders, frozen):**
- With frozen graph embeddings, adding graph modality to text improves AI adoption F1 by +0.001–0.006 (small but consistent).
- Gated fusion helps minority classes.
- Node-type reconstruction preserves less cohort-relevant structure than task-supervised pretraining (old Phase 3 approach that was replaced).

### RQ2 — Structural Hypotheses

| Hypothesis | Prediction | Result |
|---|---|---|
| H1: Scientist hub-and-spoke | Scientists have higher C:V ratio | **Reversed** — scientists have lower C:V (significant) |
| H2: Creative negative valence | Creatives have more negative stances | **Supported** (significant) |
| H3: Workforce bipolarity | Workforce has higher bipolarity completeness | Not significant (ceiling effect) |
| H4: Scientist cognitive style | Scientists have more verification-oriented CSMs | Not significant (max-2 ceiling) |

**AI adoption exploratory:** Only C:V ratio differentiates tool_user vs integrated. Other structural metrics not significant.

---

## Specific Areas I Want You to Scrutinise

Please evaluate each of these and any other concerns you identify:

1. **LLM-as-extractor circularity.** The concept graphs are extracted by an LLM, then fed to ML classifiers. Does this introduce bias? Is the graph an artefact of the extraction model's priors rather than the respondent's cognition?

2. **Sample size vs model complexity.** 1,250 graphs (188 in test set). The GIN has 265K parameters. The 100% reconstruction accuracy on the training set — is this meaningful or a sign of overfitting? Does the 128-dim embedding capture anything beyond node type identity?

3. **The text dominance problem.** Text-only F1 is 0.823. The ceiling is bounded by the interview confound (AI opening turns leaked cohort-specific language, now stripped). Is there enough headroom for graph features to demonstrate incremental value? Does the small Δ (+0.001–0.006 with frozen encoders) constitute evidence for a "distinct modality" or noise?

4. **GIN-only paradox.** GIN-only (0.843) beats text-only (0.823), which suggests graph structure is informative. But the GIN encoder never saw classification labels. What does this mean? Is it a genuine finding or an artefact of the encoding scheme (node type one-hot + label embeddings leak cohort signal)?

5. **Canonicalisation as a design choice.** Locking the vocabulary at threshold=0.35 before any modelling. Does this pre-determine what the graph features can capture? What if the threshold is wrong — is there a risk of circular reasoning (cluster, then classify using cluster-derived features)?

6. **Multiple comparisons.** 90 experiments (42 torch + 48 sklearn) across 2 targets × 6 modality combos × 4+ architectures. How should we interpret the best results given this search? Are the reported test-set F1s cherry-picked?

7. **External validity.** Can findings from this specific dataset (AI usage interviews, 3 cohorts, one extraction prompt, one LLM) generalise to other interview-based CDT contexts? What would need to hold for the "distinct modality" claim to transfer?

8. **The interviewer confound.** AI opening turns contained cohort-specific language (e.g., "As a creative professional..."). This was stripped, but: (a) could residual cohort signal still leak through AI contextual turns that weren't stripped? (b) does the confound invalidate any earlier experiments run before the fix?

9. **Effect size vs statistical significance.** Some structural tests are significant but with tiny effect sizes (eta-squared < 0.01). Is the project conflating statistical significance with practical significance?

10. **Reproducibility.** The extraction uses DeepSeek (proprietary, versioned model). The canonical vocabulary is deterministic once extracted, but re-extraction with a different model or prompt would produce different graphs. Is the pipeline reproducible in any meaningful sense?

---

## Output Format

For each concern (and any additional ones you identify):

1. **Severity:** Critical / Major / Minor / Defensible
2. **Description:** What the issue is, in 2-3 sentences
3. **Impact on claims:** How it affects the specific research conclusions
4. **Suggested mitigation:** What could be done to address it (if anything)

Then provide:
- An overall assessment of methodological soundness (1-10 scale)
- The strongest and weakest aspects of the research design
- Whether the "structurally distinct modality" claim is supported, partially supported, or unsupported by the evidence presented


# Adversarial Methodological Review: cdt-graph-modality

## Concern-by-concern assessment

### 1. LLM-as-extractor circularity — Major

Description: The graphs are not measurements of respondent cognition; they are DeepSeek's interpretation of transcripts, constrained by a hand-designed ontology. Any classifier consuming them is partly classifying DeepSeek's stylistic priors. Crucially, if the extractor produces systematically different graphs for different cohorts for reasons unrelated to cognitive structure (e.g., vocabulary triggers different node-type assignments), the "structural signal" is an extraction artefact.

Impact on claims: The claim should be downgraded from "concept graphs capture relational cognitive structure" to "an LLM-extracted intermediate representation carries cohort-predictive signal." Those are very different claims; only the latter is supported.

Mitigation: Inter-extractor agreement study — extract a subsample with 2–3 different models/prompts and measure (a) graph similarity, (b) whether downstream classification results replicate. The 3-model comparison was run for quality, but not (as far as documented) for downstream-result invariance.
Also: human extraction of a small gold subset.

### 2. Sample size vs GIN complexity — Major

Description: 265K parameters trained on 1,250 graphs with no held-out encoder data, achieving 100% node-type reconstruction. Node type is trivially recoverable from the 4-dim one-hot in the input features — 100% accuracy is near-meaningless as evidence of learned structure; the autoencoder may simply be passing through input information.

Impact on claims: The reconstruction objective is too weak to certify that the 128-dim embedding encodes topology. It may primarily encode a bag-of-label-embeddings summary (mean-pooled MiniLM vectors), which is a text feature in disguise. This directly undermines the "structurally distinct" interpretation of GIN results.

Mitigation: Ablate node features: train a GIN on structure-only inputs (degree or constant features, no label embeddings). If GIN-only F1 collapses toward stats-only or below, the 0.843 was carried by label semantics, not topology. Also use a harder reconstruction target (edge reconstruction / masked-label prediction) and report reconstruction on a held-out graph set.

### 3. Text dominance and tiny deltas — Major

Description: +0.001–0.006 F1 on a test set of 188 is well inside split noise. With n=188, the standard error of macro-F1 is roughly ±0.02–0.03; a single relabeled prediction moves F1 by ~0.005.

Impact on claims: "Small but consistent" cannot be distinguished from zero with a single fixed split. The fusion-improves-over-text claim is unsupported at current evidence strength.

Mitigation: Repeated splits (5×2 CV or 10 seeds) with paired comparisons (McNemar's test or bootstrap CIs on the F1 delta). This is cheap given frozen embeddings — there is no excuse not to do it.

### 4. GIN-only beats text-only — Major (as currently interpreted)

Description: The headline paradox dissolves once you notice the GIN node features include 384-dim MiniLM label embeddings. The GIN-only model is therefore not "graph structure only" — it is a distilled, LLM-curated text representation pooled over a graph. Its 0.843 vs text 0.823 may show that LLM-extracted concept labels are a better text summary than raw-transcript SBERT, which is interesting but is not a topology finding.

Impact on claims: The strongest single result in the project is the most over-interpreted one. Also note the 0.843 figure comes from the Phase 3 task-supervised GIN era table — the frozen Phase 5 encoder admittedly "preserves less cohort-relevant structure," so the headline number doesn't even describe the current architecture.

Mitigation: The structure-only ablation from #2 is the decisive experiment. Additionally compare against a "labels-only bag" baseline: mean MiniLM embedding of node labels, no edges. If that matches GIN-only, topology contributes nothing.

### 5. Canonicalisation lock — Minor / Defensible

Description: Locking before modelling is good hygiene against threshold-shopping. The threshold=0.35 is arbitrary but applied uniformly across cohorts, so it can't manufacture cohort signal by itself. The real risk is subtler: canonical vocabulary was built on all 1,250 transcripts including the test set — a mild transduction leak (test-set labels influenced the cluster structure).

Impact on claims: Negligible for the main conclusions; worth a footnote. The leak is label-free (no targets involved), but a purist would re-fit canonicalisation on train only.

Mitigation: Sensitivity analysis at thresholds 0.25/0.45; acknowledge the transductive vocabulary.

### 6. Multiple comparisons across 90 experiments — Major

Description: 90 configurations evaluated, and the narrative quotes the best ones. Even with val-based selection, reporting the maximum test F1 across a sweep inflates expected performance; with deltas of +0.001–0.006, selection noise exceeds the effect.

Impact on claims: Best-route deltas are upward-biased. Combined with #3, the incremental-value claim is doubly fragile.

Mitigation: Pre-specify one selection rule (best val F1 per modality combo), report only that configuration's test F1, and report the full distribution across the sweep, not the max.

### 7. External validity — Major, but honestly scoped

Description: One dataset, one extraction model, one prompt, one ontology designed for this dataset's domain (AI usage). Cohorts are also confounded with sampling: workforce n=1000 vs 125/125 means "cohort" classification is largely "majority class vs two small distinctive groups." The cohorts were likely recruited/instructed differently, so cohort signal may reflect interview-protocol differences, not professional cognition.

Impact on claims: No generalisation claim is currently defensible. The "distinct modality for CDTs" framing exceeds the evidence by a wide margin; this is a single-dataset feasibility study.

Mitigation: A second dataset is the only real fix. Short of that, frame conclusions as dataset-specific.

### 8. Interviewer confound — Major (residual risk)

Description: Stripping AI opening turns was correct, but (a) human responses echo interviewer framing ("As a creative, I…"), so cohort vocabulary leaks through human turns regardless; (b) the AI's subsequent questions steer topic distributions per cohort, shaping which concepts get extracted into graphs.
The confound contaminates the graph modality too, not just text.

Impact on claims: Cohort classification at 0.82–0.84 may substantially measure interview-protocol fingerprints. The AI-adoption target is the cleaner test — and there the graph deltas are the tiny +0.001–0.006.

Mitigation: Audit for explicit profession self-references in human turns; consider masking profession terms. Weight the AI-adoption results, not cohort, as the primary evidence.

### 9. Significance vs effect size — Minor, currently handled adequately

Description: The RQ2 writeup reports eta-squared and Cliff's delta and acknowledges ceilings, which is correct practice. The danger is narrative: "significant" structural differences with η² < 0.01 explain <1% of variance and cannot support a modality claim. Also H1 was reversed, which under a pre-registration logic is a falsification, not a finding to reinterpret post hoc.

Mitigation: State plainly which effects are practically negligible; treat reversed H1 as hypothesis-generating only.

### 10. Reproducibility — Minor

Description: Pipeline is reproducible conditional on the cached extraction artifacts (which are gitignored). End-to-end reproducibility depends on a proprietary, mutable API model. This is standard for LLM-pipeline research but should be stated.

Mitigation: Archive and publish the extracted graphs + canonical map; pin model version strings in extraction metadata.

Additional concern: class imbalance — Minor/Major

80/10/10 cohort imbalance means macro-F1 on 188 test samples gives ~19 examples per minority class. Minority-class F1 estimates are extremely noisy; "gated fusion helps minority classes" rests on a handful of predictions.

---
## Overall assessment

Methodological soundness: 6/10. Above average engineering discipline (locked vocabulary, frozen encoders, held-out test set, pre-registered structural hypotheses, confound found-and-fixed) attached to evidence that cannot yet support the central claim.

Strongest aspects: the target-agnostic encoder design (a genuinely clean way to pose the complementarity question); the locked canonical map; pre-registered RQ2 hypotheses with effect sizes; the honesty of recording reversed/null results.

Weakest aspects: (1) label embeddings inside GIN node features, which makes "graph-only" a misnomer and contaminates the headline result; (2) single fixed split with deltas an order of magnitude below the noise floor; (3) cohort target confounded with interview protocol and class imbalance.

Verdict on "structurally distinct modality": unsupported as stated; partially supported in a weaker form. The evidence supports: LLM-extracted concept-graph representations carry strong predictive signal, and 30-dim deterministic graph statistics alone reach ~0.80 F1 — the stats-only result is actually the cleanest topology evidence in the whole project, since it contains no label semantics. The evidence does not support: that topology adds information text cannot recover (untested without the structure-only ablation), or that fusion gains are real (within noise).

The two experiments that would settle it: (a) GIN with structure-only node features vs label-bag baseline; (b) 10-seed repeated splits with bootstrap CIs on the text-vs-fusion delta. Both are cheap given the frozen-embedding architecture — and arguably the architecture's whole point was to make exactly these tests easy. They should be run before any external claim is made.


# NEXT STEPS

Given the constraints (no new data, no graph-stats modality, near-zero human review), the strategy has to shift from "more evidence" to "harder-to-fake evidence": ablations, replication-across-extractors, and statistical rigor on what you have.

## Phase 1 — Fix the evaluation foundation first

Do this before any v4 extraction, because it changes how every later result is read.

- Replace the single fixed split with repeated evaluation. 10+ seeds (or repeated stratified CV), report mean ± CI on the text-vs-fusion delta, paired tests (bootstrap or McNemar's) — not point F1s.
- Pre-register the selection rule. One config chosen on val per modality combo; only that config's test score is quotable. Write the rule down before running the sweep.
- Pick the primary target deliberately. AI adoption is the cleaner target (less interviewer/protocol confound, less imbalance); demote cohort to secondary/sanity.

Interim test: re-score the existing Phase 5 results under this protocol. If the current +0.001–0.006 deltas dissolve, you've learned that before spending extraction budget.

Watch-out: repeated splits leak test data into your own decision-making over iterations — freeze the protocol once and don't tune against it.

## Phase 2 — Disentangle "graph" from "labels" (the decisive ablation)

This is the heart of the modality claim, and it needs no new data.

- Three encoder variants on the same graphs: (a) current GIN (types + label embeddings), (b) structure-only GIN (types/degree features, no label embeddings), (c) label-bag baseline (pooled label embeddings, no edges).
- The claim "topology adds signal" is supported only if (a) > (c) and (b) > chance by a margin that survives Phase 1's statistics. If (a) ≈ (c), the graph modality is a text summary in disguise — important to know, and still publishable as a finding.
- Strengthen the self-supervised objective while you're in there: edge reconstruction or masked-label prediction instead of node-type reconstruction (which is trivially solvable from inputs). Report reconstruction on held-out graphs.

Watch-out: don't let the encoder objective drift toward anything label-aware; target-agnosticism is the one architectural asset you have — keep it.

## Phase 3 — Extraction robustness via v4 + multi-model replication

Your replacement for human review: agreement between independent extractors.

- One consolidated v4 prompt carrying the ontology fixes (per-pole grounding, multi-span salience, same-type relations, topic-neutral wording, no CSM cap, edge rationales). One version, not iterative — each iteration is a full re-extraction.
- Extract with 2–3 models (e.g., DeepSeek + Claude + one more) on the full set or a large subsample.
- Two levels of agreement: (1) representation-level — do the models produce similar graphs (node-count correlation, label-embedding similarity, edge overlap)? (2) conclusion-level — the one that matters: does each model's graph corpus, run through Phases 1–2, yield the same sign and rough magnitude of deltas? Conclusion-level replication across extractors is your strongest available substitute for ground truth.
- Automated validity proxies in lieu of humans: grounding-span verbatim verification (string match against transcript — fully automatable), constraint-violation rates, and an LLM-judge spot-check with a different model on a random subsample for "does this node faithfully reflect the span."

Watch-out: v4 means new canonicalisation — treat it as a parallel experiment line, never patch the locked map. Also check node-count vs transcript-length correlation per cohort; if graph size tracks transcript length, you've found a confound, not cognition.

## Phase 4 — Confound stress tests

Cheap, label-free checks that bound how much of the signal is protocol artifact.

- Profession-leak audit: mask explicit profession/self-reference vocabulary in human turns, re-encode text, re-run. The drop quantifies how much "cohort" was vocabulary.
- Graph-side leak check: do the same masking before extraction on a subsample — does the graph modality's signal survive better than text's? If yes, that's actually positive evidence for the structural claim (structure is robust to surface vocabulary).
- Size-controlled comparison: condition or regress out node count / transcript length from classification results.

## Phase 5 — Reframe and report

- Restate the claim at the strength the evidence supports. The defensible ladder: (1) LLM-extracted graphs carry predictive signal → (2) that signal is not reducible to pooled label semantics → (3) it's robust across extractors → (4) it survives vocabulary masking better than text. Each phase above buys one rung; claim only the rungs you've bought.
- Document the single-dataset limitation explicitly; position as feasibility study with a replication protocol others could run.

## Sequencing logic

Phase 1 and 2 first — they use existing artifacts, cost nothing in API spend, and may change what v4 needs to test. Phase 3 is the big spend; only commit once you know whether the current pipeline's deltas are even real. Phase 4 piggybacks on Phase 3's infrastructure. The kill-criterion to set now: if structure-only (Phase 2b) is at chance and the label-bag matches the full GIN, the topology hypothesis is dead on this dataset, and the honest pivot is "LLM concept extraction as a text-distillation modality" — which your data already supports.