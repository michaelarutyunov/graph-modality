# Model Review 2 — From "graphs as a modality" to a mechanism

**Status:** Proposal (pre-grounding-gate). No extraction authorised yet.
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

## 5. The chosen design — unified self-positioning ontology

The two Tier-1 candidates are **not two questions**; they are facets of one causal structure:

> AI **threatens** a *task* → that task is (tightly/loosely) **coupled** to *identity* → and the
> *self* is positioned as **agent or patient** toward the AI.

That composition **is** the AI-displacement-anxiety mechanism:
**anxiety = threat × identity-coupling × low-agency** (a conjunction, not a sum).

### 5.1 Theoretical grounding (not ad hoc)

- **Appraisal theory of stress** (Lazarus): threat appraisal is *multiplicative* —
  relevance × controllability. Coupling = relevance; agency = controllability. Appraisal theory
  *predicts the interaction form.* (Primary citation.)
- **Means-end chain / laddering** (Gutman 1982): identity-task coupling is a ladder from task
  (attribute) → identity (terminal value); coupling strength = ladder connectivity.
- **Locus of control** (Rotter 1966): the agency axis.

### 5.2 Ontology sketch (question-forced, single-purpose)

- **Nodes:** Self · Identity/terminal-Value · Task/Activity · AI (with role).
- **Edges (typed, directed):**
  - Self —`IDENTIFIES_WITH`→ Task (coupling)
  - Task —`SERVES`/`CONSTITUTES`→ Identity
  - Self ↔ AI agency relation (`ACTS_ON` vs `IS_ACTED_ON_BY`)
  - AI —`THREATENS`/`AUGMENTS`→ Task
- **Three deterministic structural readouts** (computed, not separately labelled):
  - *coupling* = path strength Task→Identity
  - *agency* = direction of the Self↔AI relation
  - *threat* = displacing AI→Task edge with negative valence

Keep the schema **minimal** — schema bloat is what destroys reliability. One phenomenon, the
smallest ontology that expresses it.

---

## 6. How it will be tested (the validity design)

**"One graph, two (or three) readouts" is not redundancy — it is a dissociation design.**

### Part 1 — Convergent validity (per construct)
Each structural readout is compared against an **independent holistic rating** of the same
construct (ambivalence-style labeler that never sees the graph). Does the structure capture the
construct at all?

### Part 2 — Double dissociation (what a *difference* means)
The readouts come from *different substructures*. Ablate:
- remove Task↔Identity edges → coupling-prediction collapses, agency-prediction survives;
- remove Self↔AI edges → agency-prediction collapses, coupling-prediction survives.

A crossing pattern proves the graph encodes **separable** constructs (Campbell & Fiske
multitrait-multimethod logic; neuropsychological dissociation). **A difference is the positive
result, not an embarrassment.** If the readouts do *not* dissociate, they are one construct →
do not unify.

### Part 3 — Compositional payoff (where the graph beats text, for a stated reason)
Fit the **interaction** (threat × coupling × agency) to an **independent felt-anxiety outcome**
and test whether it beats (a) an additive model and (b) a **matched-reasoning text baseline**
(an LLM flat-summary embedding — the control round-1/2 lacked). The graph's irreducible
contribution is the *explicit conjunction*; the win, if any, is **explicitness and
sample-efficiency**, sharpest on the rare high-anxiety cell where a text model has too few
examples to learn the interaction implicitly.

### 6.1 The composite-coefficient trap (explicit)
**Do not** fuse threat × coupling × agency into a "displacement coefficient" and call that the
answer — that *defines the answer out of the inputs* (the `ai_adoption` circularity again, with
no external criterion). The coefficient is the **estimated output** of the validated interaction
test, never a constructed input.

### 6.2 Guards
- **Outcome independence:** the anxiety label rates *expressed affect/distress*, sourced
  separately, and must be *conceptually downstream* (one can have all three structural conditions
  and feel no anxiety — resigned/reframed/stoic — and vice versa). This separability is what
  makes it a non-circular criterion. It is a **different** construct than `stance_ambivalence`
  (valence-balance) → a fresh label is likely required.
- **Circularity & reasoning-asymmetry both controlled** by independent labels + the
  matched-reasoning text baseline.

### 6.3 Decision tree (every branch is publishable)
- **No dissociation** → one construct, no graph needed → stop.
- **Dissociation but no interaction** → two genuinely separate questions that don't compose →
  the unified ontology bought nothing → report; treat as separate descriptive constructs.
- **Dissociation *and* interaction beats additive + text** → the unified ontology is
  *theoretically earned* and the graph beats text *for a mechanistic reason* → the clean result.

---

## 7. High-level plan for Round 3

> **GATE 0 (MANDATORY, before any extraction call): grounding read.**
> Hand-read ~15 transcripts spanning cohorts; confirm that threat, identity-coupling,
> self-positioning/agency, and felt-anxiety are **actually stated in respondents' own words**,
> not confabulable by the LLM. **Kill/go decision.** If thin, no schema rescues it — stop here,
> having spent an afternoon instead of an extraction budget.

| Step | Work | Output |
|---|---|---|
| 0 | **Grounding read** (gate) | go/no-go note appended here |
| 1 | Ontology design (question-forced, minimal) + validator update | ADR-0008, `graph-schema` revision |
| 2 | **Matched-reasoning text baseline** spec (the missing control) | LLM flat-summary → SBERT arm |
| 3 | Round-3 extraction (single-purpose self-positioning ontology, DeepSeek) + validator | new prompt `v5.txt`, new graph dir |
| 4 | Independent labels: convergent ratings for coupling & agency; **felt-anxiety outcome** (dual-model + adjudication, ambivalence protocol) | label `.jsonl` + κ |
| 5 | Analysis: convergent validity, double dissociation (ablation), compositional interaction — under the 10-seed protocol | results-log section |
| 6 | Write-up (every decision-tree branch is reportable) | paper draft |

**Budget note:** the main extraction uses DeepSeek (cheap); short label calls are affordable on
the full corpus. The **≤30-transcript cap** (memory `phase3-extraction-budget`) applies only to
*expensive multi-model graph re-extraction* (Sonnet/Claude), not to DeepSeek extraction or short
label calls.

---

## 8. Risks & open questions

1. **Data-depth ceiling (the big one).** 15-minute interviews may not contain the relational
   detail this design needs. Gate 0 is the test; do not skip it.
2. **Outcome circularity** if the anxiety labeler silently re-derives the antecedents. Mitigate by
   instructing it to rate *expressed affect only*.
3. **Schema complexity vs reliability.** Every added node/edge type risks re-creating the
   grab-bag. Keep the ontology minimal.
4. **Interaction learnable by text given enough data.** The claim is explicitness/sample-efficiency,
   not impossibility — hence the rare-cell focus and the matched-reasoning baseline.
5. **Fresh anxiety label feasibility/cost** — confirm before committing Step 4.

---

## 9. What success looks like (pre-registered, high level)

Evaluation under the existing protocol (`docs/method-review/00-evaluation-protocol.md`):
10-seed frozen CI, PASS = CI excludes 0 **and** mean Δ ≥ +0.01.

- **Convergent validity:** each structural readout correlates with its independent rating.
- **Double dissociation:** the crossing ablation pattern holds.
- **Compositional payoff:** interaction(threat × coupling × agency) → felt-anxiety beats additive
  **and** matched-reasoning text, especially on the rare high-anxiety cell.

All three branches of the §6.3 decision tree are honest, publishable outcomes — the design wins
either way, which is the property rounds 1–2 lacked.

---

## Appendix — references

- Lazarus, R. S. (1991). *Emotion and Adaptation.* (appraisal theory; conjunctive threat)
- Gutman, J. (1982). A means-end chain model based on consumer categorization processes. *JM.*
- Rotter, J. B. (1966). Generalized expectancies for internal vs external control. *Psych. Monographs.*
- Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation by the
  multitrait-multimethod matrix. *Psych. Bulletin.* (dissociation logic)
- Kelly, G. A. (1955). *The Psychology of Personal Constructs.* (construct systems)
- Project: ADR-0006, ADR-0007; `MODEL_REVIEW_1.md`; `.claude/context/results-log.md`
  (Phase-6 + RSA sections); `.claude/context/ambivalence-target.md` (label protocol).
