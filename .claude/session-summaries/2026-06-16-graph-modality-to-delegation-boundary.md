> **What this session is about:** the full reasoning trail of how this project pivoted from "are
> concept graphs a distinct modality?" (answer: yes but only *distributionally*, not relationally)
> to a new, falsifiable construct — the **human–AI delegation boundary** — and then executed
> Round-3 data labeling through a **Gate-1 GO**. Written as source material for a future article;
> emphasis is on the *why* and the dead-ends, not just the conclusions.

# Session summary — from "graphs as a modality" to the delegation boundary

**Date:** 2026-06-15 → 2026-06-16 · **Participants:** Michael + Claude (Opus) · **Status:** Round-3
epic `graph-modality-5ch` at 4/8; Gate-1 GO; independent labels produced; next = B4 (ontology).

---

## 0. The opening question

Michael's framing: in qualitative research you work raw text to extract meaning; he wanted to
*quantify that extra work* by comparing predictive power of **text alone vs graph alone vs
text+graph**. The results only "partially" confirmed the hypothesis, and he wanted to know if the
experiment could have been designed differently.

## 1. Where Phase 6 had actually landed (the real verdict)

Reading the project's own record made the verdict sharper than "partial":
- **Modality-distinctness SUPPORTED** — graph *stats* beat text on the fair target
  (`stance_ambivalence`, 0.433 > 0.367, CI excludes 0).
- **Complementarity NOT supported** — no fusion beat the best single modality.
- **Relational/topology DEAD** — edges add nothing; `label_bag` (no edges) > `full_gin` (with
  edges); RGCN with more capacity doesn't beat untyped (ADR-0006/0007).

One-line: *the graph behaves like a sophisticated word-count of the extracted concepts, not a map
of how they connect.*

## 2. The diagnosis (the pivot's seed)

The deep problem wasn't the fusion math — it was that the project was a **representation in search
of a justification**. A generic concept-graph ontology was built first, then targets were
hunted to validate it (cohort → ai_adoption → ambivalence). A multi-purpose ontology yields a
**grab-bag geometry** — which is exactly why the later RSA reliability came out low.

Michael's own correction, mid-conversation: *"I should have started with the label, then designed
the ontology around it."* I refined this to **question-first, not label-first** (designing the
ontology to *encode* the label just manufactures circularity — the `ai_adoption` failure mode).

## 3. The label-free geometry check (bead `graph-modality-z2o`)

Before any new extraction, we asked the most basic question — *is the graph even geometrically
distinct from text?* — via RSA / Mantel, needing no label. Result: text↔graph_stats r ≈ 0.01
(distinct) BUT split-half reliability 0.488 < the pre-registered 0.50 → **DISTINCT-but-not-
trustworthy.** The disagreement cases showed the graph calling a scientist and a warehouse worker
"twins" on matching counts — statistical coincidence in a bag of features. This *triangulated* the
distributional verdict from a third, independent angle.

## 4. Principles established (reusable, article-worthy)

1. **Distinctness is an outcome, not a design target** — optimizing for it manufactures artifacts.
2. **Faithfulness is purpose-relative** — undefinable until the question is fixed.
3. **Reliability is a prerequisite** — a measure that disagrees with itself can't be valid.
4. **The "graph earns its keep" test** — only worth a graph if *two people with the same concepts
   and same valences get different answers from how those concepts connect*. If counting suffices,
   no graph.
5. **Fast filter** — *relation between two typed things* (good) vs *property of one thing* (a
   disguised count).
6. **Two hard problems always controlled** — circularity (labels from a process that never sees
   the graph) and grounding/data-depth (can't extract richness 15-min interviews don't contain).

## 5. Finding the construct (the heart of the pivot)

Six candidate phenomena were run through the filters. Tier-1 (genuinely relational): identity-task
coupling; the self↔AI relational schema. Tier-2 (disguised counts): temporal orientation,
epistemic posture. We nearly committed to a unified "displacement-anxiety mechanism"
(anxiety = threat × identity-coupling × low-agency, grounded in appraisal theory).

Then two course-corrections, both from confronting the data:
- **Gate-0 grounding read (15 transcripts):** the elements are *real*, not confabulated — but
  agency is near-constant, and threat/identity/anxiety **cluster in creatives** (the cohort
  confound again), and the full anxious configuration is rare.
- **Coupling bake-off (fresh 15):** **competence-coupling is the cross-cohort winner**; reframing
  threat as *competence-erosion* (not job loss) makes it appear in scientists/workforce too.

Then Michael brought in Anthropic's per-cohort interview objectives (creatives→creativity,
scientists→trust/barriers, workforce→relationship) and asked for a *higher-level construct* that
escapes the per-occupation protocol bias. That produced the synthesis: those are three **dialects**
for one underlying act — **where a person draws the line between tasks they retain and tasks they
cede to AI, and why.** The **delegation boundary** (function-allocation theory; Sheridan/
Parasuraman) is the protocol-invariant superordinate. Trust/ownership/relationship are surface
vocab; competence/identity coupling is the latent driver; the boundary partition is the observable.
(Reject "perceived reliability" as a name — it re-imports the scientist pole and fails on creatives
who retain work that AI *could* do, because it's *theirs*.)

Captured in `docs/MODEL_REVIEW_2.md` (reframed from the earlier displacement-anxiety draft).

## 6. The validity design (so a result would mean something)

- "One graph, two readouts" is a **dissociation design**, not redundancy — a *difference* between
  readouts is the positive result (Campbell & Fiske).
- The relational payoff is whether the coupling **structure** predicts the boundary beyond a flat
  trust/sentiment score AND a **matched-reasoning text baseline** (the control rounds 1–2 lacked).
- **The composite trap:** never fuse the components into a "delegation index" and call it the
  answer — that defines the answer out of the inputs (the `ai_adoption` circularity).

## 7. Round-3 execution (epic `graph-modality-5ch`)

- **B1 — rubric dogfood.** Hand-coded 10 transcripts against `selfpos_v1`. Found two real rubric
  gaps: the self-vs-AI tie-breaker (competence/identity vs trust_reliability) and the
  competence *guard* vs *compensate* split. Decided to split.
- **B2 — model calibration.** Cheap pair Agnes+Haiku vs a human reference. First pass looked like a
  fail; the catch was a **metric artifact** — full multi-label Jaccard was tanked by
  `output_efficiency` over-tagging, while the *core* categories agreed well. Switched the rubric to
  multi-label + banned `output_efficiency` co-tagging, scored on core categories → Agnes 0.65 /
  Haiku 0.633, inter-model 0.717. **Cheap pair confirmed; Kimi dropped, no Sonnet.**
  (Kimi gotcha discovered here: k2.6 thinking is ON by default → times out + temperature locked.)
- **B3 — Gate 1 (full corpus, 1,238 dual-coded).** 3/4 criteria pass cleanly; the OVERALL came up
  NO-GO on **one number** — competence inter-coder Cohen κ 0.326. That's the **kappa paradox**:
  competence is ~80% prevalent, so κ deflates despite 78% raw agreement. Under the correct
  high-prevalence statistic (**Gwet's AC1 = 0.675**, PABAK 0.561) codability passes. Amended the
  criterion (Cohen κ → Gwet AC1, both recorded) → **GO.** Key design finding: competence-coupling
  is near-ubiquitous, so discrimination lives in the **boundary partition + rationale mix**, not
  competence-presence.
- **B7 — independent labels.** Consensus from Agnes+Haiku (breadth agreement 83.2%); all 208
  breadth disputes were **single ordinal steps** (low↔med / med↔high). Michael chose Kimi
  adjudication (ambivalence protocol); using the *proven* thinking-disabled config it resolved all
  208, 0 failures, Kimi siding Agnes 149 / Haiku 59 (Agnes the more accurate breadth coder, judged
  independently). User spot-checked 10 — sound, transcript-grounded. Final labels:
  `cache/selfpos_boundary.jsonl` (1,250), breadth medium 740 / low 470 / high 40.

## 8. Methodological lessons (the article's "how to do this honestly" spine)

- **Pick the right agreement statistic for the base rate.** Cohen's κ lies at high prevalence;
  Gwet's AC1 / PABAK don't. We almost killed a valid construct on a κ artifact.
- **Multi-label Jaccard punishes over-tagging.** Score on the load-bearing categories; ban
  throwaway labels in the rubric instead of letting them pollute the metric.
- **Cheap models were enough.** Agnes (free) + Haiku cleared the bar; the expensive thinking model
  (Kimi) wasn't needed for coding and was unreliable for it — reserved as a (cheap-config)
  adjudicator only.
- **Gate discipline.** Every gate (B2 selection rule, B3 PASS criteria) was pinned to byte-
  identical thresholds *before* running, then honestly amended only on a documented statistical
  grounds — not goalpost-moving.
- **Dead-ends are cheap when gated.** Two graph-free read-throughs (Gate 0, bake-off) reshaped the
  whole construct for the cost of an afternoon, before any extraction spend.
- **The deepest reframe was free.** "Question-first, not label-first" and the delegation-boundary
  superordinate came from *thinking*, not compute.

## 9. Where it stands / next

- **Done:** B1, B2, B3 (GO), B7. Epic 4/8.
- **Next:** **B4** (`graph-modality-17v`) — design the minimal delegation-boundary ontology +
  ADR-0008 + graph-schema revision + validator. Then B5 (extraction) → B6 (canonicalise) → B8
  (package encoding-ready dataset). Handoff prompt for B4 is in the conversation.
- **Caveat carried forward:** the ontology must encode the *boundary partition*, not just rationale
  flags (competence is ~80% ubiquitous).

## Artifact index

- **Design/specs:** `docs/MODEL_REVIEW_2.md` (delegation-boundary spec), `docs/MODEL_REVIEW_1.md`
  (round-1 adversarial review, renamed this session), `docs/adr/0006`, `0007`.
- **Results:** `.claude/context/results-log.md` — "Geometry RSA" + "Gate 1" sections.
- **Beads:** epic `graph-modality-5ch`; `z2o` (RSA), `8hy`/`zb9`/`nu2`/`2wf` (B1/B2/B3/B7 closed),
  `17v`/`sfh`/`ghm`/`5p4` (B4–B8 open).
- **Code:** `s2_extraction/selfpos_{calibrate,labeler,gate1,consensus,adjudicator}.py`,
  `s2_extraction/prompts/selfpos_v1.txt`, `s2_extraction/README.md`.
- **Data:** `cache/selfpos_{agnes,haiku}.jsonl` (raw), `cache/selfpos_boundary.jsonl` (consensus
  labels), `cache/selfpos_adjudications.jsonl` (audit trail),
  `results/method_review/selfpos_{calib,gate1,consensus}/`.
- **Memory:** `bd memories delegation` (cross-session state + Kimi config gotcha).
