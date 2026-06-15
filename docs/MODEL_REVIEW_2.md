# Model Review 2 — From "graphs as a modality" to the human–AI delegation boundary

**Status:** Proposal (construct-validation gate). Gate 0 (grounding read) PASSED with a reframe;
next gate is the graph-free prevalence pass (Gate 1). No graph extraction authorised yet.
**Date:** 2026-06-15
**Author:** Michael (with Claude, brainstorming session)
**Companion docs:** `MODEL_REVIEW_1.md` (round-1 adversarial review), `docs/method-review/00-evaluation-protocol.md` (10-seed protocol), ADR-0006 / ADR-0007 (Phase-6 verdict).

---

## 0. Purpose of this document

This is a design review and high-level plan for a possible **third investigation round**.
It exists because the first two rounds (v3, v4) settled the original hypothesis in the
*negative-relational* direction, and before paying for a third extraction we need a precise,
defensible answer to one question: **what is the right outcome, and what graph would be
meaningful enough to deliver it?**

It documents the reasoning that led from "test whether graphs add predictive value" to
"model a specific cognitive *mechanism*," and the validity design that makes that testable
without repeating the circularity and confound mistakes of rounds 1–2.

---

## 1. Where Phase 6 left us (the settled verdict)

The Phase-6 verdict is **triangulated across three independent methods**, all agreeing:

| Leg | Method | Result |
|---|---|---|
| 1. Predictive (ADR-0006) | 10-seed fusion on `stance_ambivalence` | graph stats 0.433 > text 0.367 (CI excl. 0), but **no fusion beats the best single modality**; `label_bag` (no edges) > `full_gin` (with edges). |
| 2. Relational kill (ADR-0007) | H_edge ladder + RGCN last-look | edge presence *and* type at chance; RGCN with more capacity does not beat untyped. Topology dead. |
| 3. Label-free geometry (bead `graph-modality-z2o`) | RSA / Mantel | text↔graph_stats r = 0.0126 (distinct) **but** split-half reliability 0.488 < 0.50 → **not trustworthy**. |

**Conclusion (unchanged):** the graph modality's usable signal is **distributional
node-attribute** (concept presence + stance-valence balance), **not relational/topological**.
In one line: *the graph behaves like a sophisticated word-count of the extracted concepts,
not a map of how those concepts connect.*

The RSA disagreement cases make this concrete: the graph calls a scientist and a warehouse
worker "twins" because they share node-counts and valence-fractions, while their text is
maximally far apart — i.e. **statistical coincidence in a bag of counts.**

---

## 2. The diagnosis — a representation in search of a justification

The root cause is a **methodological inversion**. The project defined a *generic* concept-graph
ontology first (Construct / Value / Stance / CSM) and then went **label-hunting** to justify it
(cohort → ai_adoption → stance_ambivalence). A multi-purpose ontology produces a
**grab-bag geometry** — which is exactly why split-half reliability landed at 0.49: different
subsets of the 30 stats encode semi-independent things, so the representation has no stable
opinion about who resembles whom.

The corrective principle, surfaced during this session:

> **Question-first, not label-first.** Start from a phenomenon about the world; let the
> question *force* the ontology (so it is coherent and reliable by construction); operationalise
> the answer *independently* of the graph (so the test is not circular); and treat distinctness
> as an **outcome to validate**, never a target to optimise.

(Note the trap in the naive "label-first": designing an ontology to *encode* the label makes
"graph predicts label" tautological — the `ai_adoption` circularity in new clothes.)

---

## 3. Principles established this session

1. **Distinctness is an outcome, not an input.** A graph that genuinely captures something real
   will *turn out* distinct. Optimising for distinctness manufactures artifacts.
2. **Faithfulness is purpose-relative.** A graph can never model a whole mind; "faithful" means
   *captures the aspect the question is about, without distortion* — undefinable until the
   question is fixed.
3. **Reliability is a prerequisite, defined operationally.** If the graph's notion of "who is
   similar to whom" flips depending on which features you read, the signal is an artifact. An
   unreliable measure cannot be valid.
4. **The "graph earns its keep" test.** A graph is the right representation **only** for a
   question where *two people with the same concepts and same valences get a different answer
   purely from how those concepts connect.* If counting suffices, no graph is needed.
5. **Fast filter for candidate questions.** *Relation between two typed things* (good — relational)
   vs *property of one thing* (bad — a disguised count).
6. **Two hard problems must always be controlled:** (a) **circularity** — labels must come from a
   process that never sees the graph; (b) **grounding / data-depth** — you cannot extract
   relational richness that 15-minute interviews do not contain.

---

## 4. Candidate phenomena considered

Six candidates were run through the principles above.

| # | Candidate | Earns a graph? | Verdict |
|---|---|---|---|
| 5 | **Identity–task coupling** | yes — path between task and identity nodes | **Tier 1.** Native to the means-end ontology; theoretically loaded (the displacement-anxiety mechanism). |
| 3 + 6 | **Self↔AI relational schema** (agency direction + relationship type) | yes — edges around self/AI nodes | **Tier 1.** One family: #3 = directionality, #6 = type. Needs new self/AI nodes. |
| 2 | Temporal orientation | no — a node attribute (tag + count) | **Tier 2.** Distributional; a flat horizon-classifier suffices. |
| 1 | Epistemic posture / uncertainty tolerance | no — largely lexical; argument-structure, not means-end | **Tier 2.** Already present as CSM (ceiling effect); modal-verb/hedge leakage → text likely wins. |
| 4 | Engagement depth with the interview | partly (contradiction structure) but meta-level | **Tier 3.** Answers a methods question, not the person's model; needs latency/sequence data the pooled extraction discards. |

**Pattern:** Tier 1 candidates are all "relation between two typed things with direction/path
mattering." Tier 2 fail because they are "property of one thing."

---

## 4.5 Empirical pre-tests (this session) — and how the construct evolved

Two graph-free read-throughs of the transcripts (hand-coded by Claude; ~30 transcripts total,
stratified 5/cohort each, two disjoint samples) were run *before* committing any extraction. They
moved the design substantially:

- **Gate 0 (grounding read, 15 transcripts).** The displacement-anxiety elements (threat,
  identity-coupling, agency, felt-anxiety) are **genuinely in respondents' words, not
  confabulated** (creativity_0025 is a textbook case). BUT: (a) **agency is near-constant** —
  everyone is an actor in *how they use* the tool; (b) **threat, identity-coupling, and
  felt-anxiety cluster in creatives** — the same cohort-leakage confound that killed `cohort`;
  (c) the full anxious-displaced configuration looked **rare** (the ambivalence-`high` power
  problem again).
- **Coupling bake-off (fresh 15 transcripts).** Re-coded for three coupling-targets
  (identity / competence / output) + agency. **Competence-coupling is the cross-cohort winner**
  (10/15, all three cohorts), with real internal variance (*guard / develop / compensate*).
  Identity-coupling is creative-skewed; agency is confirmed weak (constant for tool-use).
  Reframing threat as **competence-erosion** (not job loss) makes it appear in scientists and
  workforce too (science_0010: "without struggle it's difficult to retain… this kills creativity").
- **Anthropic per-cohort objectives.** The interviewer probed creatives on *creativity*,
  scientists on *trust/barriers*, workforce on *relationship/integration* — three surface
  vocabularies for one underlying act. Trust / perceived value as *scalars* are Tier-2 (lexical →
  text wins); as **task-conditional** patterns they are the behavioural surface of coupling.

**Net evolution:** displacement-anxiety (identity, creative-skewed, low-power) → competence-coupling
(cross-cohort) → **the human–AI delegation boundary** — the superordinate that is invariant to the
interviewer's per-cohort protocol. That is the spine below.

---

## 5. The chosen construct — the human–AI delegation boundary

The protocol-invariant phenomenon all three cohorts circle is **where the respondent draws the
line between tasks they retain and tasks they cede to AI, and why.** Trust (scientists), ownership
(creatives), and relationship (workforce) are three *dialects* describing the same partition.
Naming the construct after any one dialect (e.g. "perceived reliability") re-imports that cohort's
bias — and fails on the others (a creative can find AI fully reliable and still retain a task
because it is *theirs*).

### 5.1 Three-level model

| Level | Construct | Nature |
|---|---|---|
| **Driver (latent)** | task ↔ competence / identity **coupling** | *why* a task is retained |
| **Structure (observable)** | the **delegation boundary** — retained vs ceded task partition | *where / what* — the graph |
| **Surface (cohort vocab)** | trust / ownership / relationship | *how they describe it* — not measured |

Core hypothesis: **the coupling structure predicts where the delegation boundary falls** — and
does so beyond a flat trust/sentiment score (the relational claim).

### 5.2 Theoretical grounding (cohort-neutral, not ad hoc)

- **Function allocation / levels of automation** (Sheridan & Verplank 1978; Parasuraman, Sheridan
  & Wickens 2000): the science of which functions humans retain vs delegate to automation — the
  delegation boundary itself, decades deep and occupation-neutral. (Primary anchor.)
- **Means-end chain / laddering** (Gutman 1982): the coupling driver — task (attribute) →
  competence/identity (terminal value).
- **Appraisal theory** (Lazarus): retained ⇄ ceded reflects relevance × controllability — the
  rationale for *why* the boundary sits where it does.

### 5.3 Ontology sketch (question-forced, minimal — designed ONLY if Gate 1 passes)

- **Nodes:** Self · Task/Activity · competence/identity Value · AI.
- **Edges:** Task—`RETAINED_BY`→Self / Task—`CEDED_TO`→AI (the boundary); Task—`SERVES`→Value
  (the coupling rationale).
- **Readouts (computed, not separately labelled):** *delegation breadth* (ceded fraction);
  *boundary–coupling alignment* (are retained tasks the competence/identity-coupled ones?).

Keep it minimal — schema bloat destroys reliability.

---

## 6. How it will be tested (the validity design)

The relational claim is: **the coupling structure predicts the delegation boundary beyond a flat
trust/sentiment score.**

### Part 1 — Convergent validity
The graph's structural readout of the delegation boundary (retained/ceded partition) matches an
**independent holistic coding** of the same boundary (the graph-free labeler of Gate 1, which
never sees the graph). Does the structure capture the construct at all?

### Part 2 — Discriminant / dissociation (what a *difference* means)
Boundary-location and coupling-rationale live in *different substructures* (Task→Self/AI edges vs
Task→Value edges). Ablating the coupling edges should degrade prediction of *why* tasks are
retained while leaving *which* tasks are ceded recoverable, and vice versa. A crossing pattern
proves the graph encodes **separable** information (Campbell & Fiske). **A difference is the
positive result, not an embarrassment.**

### Part 3 — Relational payoff (where the graph beats text, for a stated reason)
Test whether the **coupling structure predicts the delegation boundary** beyond (a) a flat
**trust/sentiment score** and (b) a **matched-reasoning text baseline** (an LLM flat-summary
embedding — the control rounds 1–2 lacked). Two people with the *same overall trust level* but
*different boundary patterns* is exactly the case a scalar cannot represent and a graph can.

### 6.1 The composite trap (explicit)
**Do not** fuse the readouts into a single "delegation index" and call it the answer — that
*defines the answer out of the inputs* (the `ai_adoption` circularity, no external criterion). Any
index is the **estimated output** of a validated test against an independent criterion, never a
constructed input.

### 6.2 Guards
- **Independent outcome:** the delegation boundary is coded by a separate process (the Gate-1
  labeler), never read off the graph.
- **Circularity (coupling ⇄ boundary ⇄ trust):** often stated in one breath; operationalise
  coupling from Task→Value structure and the boundary from explicit retain/cede statements, or
  label them independently.
- **Protocol confound:** trust / ownership / relationship are elicited *differently per cohort* —
  measure the construct, not the dialect, and check boundary-talk *depth* for cohort skew (§7).
- **Reasoning-asymmetry:** controlled by the matched-reasoning text baseline.

### 6.3 Decision tree (every branch is publishable)
- **Boundary not codable / no variance** → stop (Gate 1 catches this first).
- **Coupling does NOT predict boundary beyond sentiment** → the boundary is a scalar attitude →
  distributional story, no graph needed.
- **Coupling predicts boundary beyond sentiment AND matched-reasoning text** → the delegation
  boundary is a genuine relational construct → the clean result.

---

## 7. High-level plan for Round 3

> **GATE 0 — grounding read (15 transcripts): ✅ PASSED, with reframe** (§4.5). The elements are in
> the data; the construct evolved to the delegation boundary.
>
> **GATE 1 (NEXT, MANDATORY, graph-free): prevalence / codability pass.** An LLM coding pass over
> raw human turns — **no graph, no ontology** — to confirm the delegation boundary varies, codes
> reliably, is cross-cohort, and is not protocol-confounded. Kill/go before any ontology work.

### Gate 1 design (the prevalence pass)
- **Sample:** ~150 transcripts, **stratified ≈50 / 50 / 50** (workforce / creatives / scientists —
  *over-sampling the minorities* for cross-cohort power, NOT proportional), seed=42, **human turns
  only** (interviewer stripped, per the Phase-3 confound fix).
- **Method:** reuse the `ambivalence_labeler` harness; **new prompt** `prompts/selfpos_v1.txt`.
  **Dual-model** (Agnes + Haiku — neither is the DeepSeek graph extractor → breaks circularity
  *and* doubles as the future independent outcome labeler). **Report Cohen's κ** as the codability
  check. **No adjudication yet** (that is for final label production, post-gate).
- **Code per transcript:** (a) tasks tagged *retained / ceded / shared*; (b) dominant **coupling
  rationale** for retained tasks (competence / identity / trust-reliability / output-efficiency /
  other); (c) **boundary-talk depth** (1–3 richness rating) — the cohort-confound probe.
- **Gate 1 PASS iff:** boundary varies; competence-rationale appears **across all three cohorts**;
  κ ≥ ~0.4 (codable); and boundary-talk depth is **not strongly cohort-confounded** (or the skew
  is measurable and controllable). Fail on any → revisit construct or invoke the data-ceiling branch.

### Subsequent steps (ONLY if Gate 1 passes)
| Step | Work | Output |
|---|---|---|
| 2 | Ontology design (minimal: Self/Task/Value/AI; retain-cede + serves) + validator | ADR-0008, schema revision |
| 3 | **Matched-reasoning text baseline** spec (the missing control) | LLM flat-summary → SBERT arm |
| 4 | Round-3 extraction (single-purpose, DeepSeek) | `prompts/selfpos_extract_v1.txt`, new graph dir |
| 5 | Final independent boundary labels (dual-model + adjudication, ambivalence protocol) | label `.jsonl` + κ |
| 6 | Analysis: convergent validity, dissociation, relational payoff — 10-seed protocol | results-log section |
| 7 | Write-up (every decision-tree branch reportable) | paper draft |

**Budget note:** Gate 1 + extraction use cheap backends (Agnes free; Haiku/DeepSeek cheap; short
calls). The **≤30-transcript cap** (`phase3-extraction-budget`) binds only expensive multi-model
*graph* re-extraction (Sonnet/Claude), not these.

---

## 8. Risks & open questions

1. **Data-depth ceiling.** 15-minute interviews may lack the relational detail needed. *Reduced*
   by Gate 0 (the boundary is richly stated) but not eliminated; Gate 1 quantifies it.
2. **Protocol elicitation-depth confound.** The interviewer probed each cohort differently, so
   *how much* boundary-talk a transcript contains may correlate with cohort even if the construct
   is shared. Measured directly by Gate 1's depth rating.
3. **Circularity (coupling ⇄ boundary ⇄ trust).** Often stated in one breath; operationalise from
   different structures or label independently.
4. **Boundary edges into the lexical.** "I do it myself" is partly surface text; the relational
   test (coupling structure predicts the boundary *beyond* a trust-keyword baseline) settles
   whether a graph is needed.
5. **Schema complexity vs reliability.** Keep the ontology minimal.

---

## 9. What success looks like (pre-registered, high level)

Evaluation under the existing protocol (`docs/method-review/00-evaluation-protocol.md`):
10-seed frozen CI, PASS = CI excludes 0 **and** mean Δ ≥ +0.01.

- **Convergent validity:** the structural delegation-boundary readout matches its independent coding.
- **Dissociation:** boundary-location and coupling-rationale are separably recoverable.
- **Relational payoff:** the coupling structure predicts the delegation boundary beyond a flat
  trust/sentiment score **and** a matched-reasoning text baseline.

All three branches of the §6.3 decision tree are honest, publishable outcomes — the design wins
either way, which is the property rounds 1–2 lacked.

---

## Appendix — references

- Sheridan, T. B., & Verplank, W. L. (1978). *Human and Computer Control of Undersea Teleoperators.* (levels of automation / function allocation — the delegation boundary)
- Parasuraman, R., Sheridan, T. B., & Wickens, C. D. (2000). A model for types and levels of human interaction with automation. *IEEE SMC.*
- Gutman, J. (1982). A means-end chain model based on consumer categorization processes. *JM.* (coupling driver)
- Lazarus, R. S. (1991). *Emotion and Adaptation.* (appraisal theory; why the boundary sits where it does)
- Rotter, J. B. (1966). Generalized expectancies for internal vs external control. *Psych. Monographs.*
- Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation by the
  multitrait-multimethod matrix. *Psych. Bulletin.* (dissociation logic)
- Kelly, G. A. (1955). *The Psychology of Personal Constructs.* (construct systems)
- Project: ADR-0006, ADR-0007; `MODEL_REVIEW_1.md`; `.claude/context/results-log.md`
  (Phase-6 + RSA sections); `.claude/context/ambivalence-target.md` (label protocol).
