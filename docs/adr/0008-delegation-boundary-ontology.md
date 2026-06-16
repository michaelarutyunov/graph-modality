# 0008 — Round-3 delegation-boundary graph ontology (`selfpos` / v5)

- **Status:** Accepted (design; extraction is bead `graph-modality-sfh`/B5, not yet run)
- **Date:** 2026-06-16
- **Supersedes:** none (new corpus; v4 ontology in `graph-schema.md` retained for provenance)
- **Related:** ADR-0004 (edge-signal validity, the determinism trap), ADR-0006/0007
  (Phase-6 distributional-not-relational verdict); `docs/MODEL_REVIEW_2.md` §5–6;
  results-log "Gate 1 — delegation-boundary prevalence/codability"; bead `graph-modality-17v` (B4),
  epic `graph-modality-5ch`.

## Context

The Phase-6 verdict (ADR-0006/0007) was that the concept-graph modality's signal is
**distributional node-attribute** (label semantics / stance valence), **not relational/topological**:
edges added nothing, and `label_bag` (no edges) beat `full_gin` (with edges). MODEL_REVIEW_2 then
reframed the program around a single protocol-invariant construct — the **human–AI delegation
boundary**: *where a respondent draws the line between tasks they retain and tasks they cede to AI,
and why* (function allocation, Sheridan/Parasuraman; means-end coupling, Gutman; appraisal, Lazarus).

**Gate 1 passed (GO)** on a graph-free coding pass over the full corpus (results-log; bead
`graph-modality-nu2`). The independent outcome labels already exist — `cache/selfpos_boundary.jsonl`,
1,250 transcripts, dual-coded (Agnes + Haiku, neither the DeepSeek graph extractor → breaks
circularity), 208 disputes Kimi-adjudicated, user spot-checked. Each record carries
`delegation_breadth ∈ {low, med, high}`, `rationales_present` (multi-label over
`{competence, identity, trust_reliability, output_efficiency, competence_compensate, other}`), and
`boundary_talk_depth ∈ {1,2,3}`. **These labels are the independent criterion the downstream tests
predict; the graph's deterministic readouts must be checkable against them.**

Two Gate-1 findings bind this design:

1. **Competence-coupling is ~ubiquitous, hence non-discriminative by presence.** `trust_reliability`
   is the *most* common consensus rationale (980/1,250 = 78%), `competence` 859, `identity` 713. The
   construct's discriminative power therefore lives in the **boundary partition** (which tasks
   retained vs ceded; breadth) and the **rationale mix** — *not* in the presence of any single
   rationale. The ontology must richly encode the self/task/AI partition, not just rationale flags.
2. **Class imbalance.** Breadth is med 740 / low 470 / high 40 — effectively two usable classes;
   `high` (3.2%) is low-power. Convergent validity on breadth is also near-circular (the graph
   extractor and the Gate-1 labeler both read disposition from the same statements) — it is closer
   to a reliability check than a discovery. The *informative* tests are the alignment readout, the
   dissociation, and the relational-payoff vs a matched-reasoning text baseline (MODEL_REVIEW_2 §6).

### The design tension

Rounds 1–2 and v4 died of **schema bloat**: more node/edge types → coder/LLM divergence → reliability
collapse, with no relational payoff (ADR-0004's determinism trap: edge type was a function of
endpoint node types, so a node-type histogram reconstructed it). The discipline this time is
**minimal and structural**. But a thin graph risks being too thin to out-predict text. The resolution
(below) is to keep the ontology thin and make *lexicon* an encoding-time dial, so the question
"is this modality structural or a bag-of-words in disguise?" is answered **by ablation, not by guess**.

## Decision

A thin, structural ontology — schema id **`selfpos`** (prompt `v5`/`selfpos_v1`). New corpus and new
canonical map (`canonical_map_selfpos`, bead `ghm`): a new experiment, **not** a patch to v4. The v4
schema, validator path, and on-disk graphs are retained untouched for provenance.

### Nodes (4 types)

| type | cardinality | grounding | attrs |
|---|---|---|---|
| `Self` | exactly 1 (id `self`) | **exempt** — structural anchor, extractor-injected | — |
| `AI` | 0–1 (id `ai`) | **exempt** — structural anchor, extractor-injected | — |
| `Task` | 0..n | required `grounding_span` (verbatim) | `label` |
| `Value` | 0..n | required `grounding_span` (verbatim) | `label`, `value_type ∈ {competence, identity}` |

`Self`/`AI` are **injected deterministically by the extractor**, not emitted by the LLM. The LLM emits
only `Task`/`Value` nodes and edges referencing the literal ids `self`/`ai`. This was the round-1/2
reliability lesson: fewer free-form nodes → fewer hallucinated anchors. They are grounding-exempt
because they are structural poles, not extracted claims; requiring a verbatim span for "Self" would be
artificial. `AI` is present iff at least one `CEDED_TO`/`SHARED_WITH` edge exists (no orphan anchor).

`type` is the node's **entity category** (4 values). `value_type` is a sub-attribute carried **only by
`Value` nodes** (the means-end distinction: skill/judgment vs who-they-are/authorship). In the
purified type-only encoding a node's feature is its `type` one-hot, with `Value` split by `value_type`
→ five abstract categories: `Self`, `AI`, `Task`, `Value:competence`, `Value:identity`.

### Edges (4 types) — every edge carries a required verbatim `grounding_span`

| relation | source→target | role | extra attr |
|---|---|---|---|
| `RETAINED_BY` | Task→Self | boundary partition (which side) | `rationale_tags` |
| `CEDED_TO` | Task→AI | boundary partition | `rationale_tags` |
| `SHARED_WITH` | Task→AI | boundary partition (control retained) | `rationale_tags` |
| `SERVES` | Task→Value | **coupling** (the ablatable "why-retained") | — |

- **Partition invariant:** every `Task` has **exactly one** boundary edge → a clean retained/ceded/
  shared partition. `shared` is a single `SHARED_WITH`→AI edge (distinct relation from `CEDED_TO`; the
  AI's involvement is what separates shared from retained), preserving the one-boundary-edge invariant.
- **`rationale_tags`** (boundary edges only) is a multi-label list ⊆
  `{trust_reliability, output_efficiency, competence_compensate, other}` (may be empty). The
  competence/identity coupling is carried by `SERVES`→`Value`, **not** by tags. Rationale placement is
  thus split, and principled: only competence/identity are means-end *terminal self-values* (first-class
  `Value` nodes / the dissociation's ablation target); trust/efficiency/compensate are
  appraisal/practical decision reasons that qualify the disposition (edge attributes). The labeler's
  `rationales_present` is reconstructed as `{Value.value_type} ∪ {boundary rationale_tags}`.

### Dropped from v4 (deliberate minimalism)

`Construct`/`Stance`/`CognitiveStyleMarker` node types; bipolarity; the `explicit`/`inferred` grounding
enum; the free-text edge `rationale`. The **verbatim `grounding_span` is the single audit unit**, and
forcing it on every edge also improves extraction reliability (the LLM must commit to evidence,
reducing confabulated dispositions).

### Two deterministic readouts (checkable against `selfpos_boundary.jsonl`)

1. **Delegation breadth** = `(|CEDED_TO| + 0.5·|SHARED_WITH|) / |Task|`, thresholded to low/med/high;
   convergent-validated against `delegation_breadth`. (Secondary: reconstruct `rationales_present` and
   compare to the label.)
2. **Boundary–coupling alignment** = the φ coefficient of the 2×2 table
   `{retained vs ceded} × {has SERVES→Value vs not}` over `Task` nodes. Positive φ ⇒ retained tasks are
   the competence/identity-coupled ones (the core hypothesis). The boundary substructure (Task→Self/AI)
   and coupling substructure (Task→Value) are independently ablatable → this is what the dissociation
   test (MODEL_REVIEW_2 §6 Part 2) requires.

### Lexicon is an encoding-time dial, not baked-in node bloat

The ontology preserves **both** lexical and structural information losslessly so the encoding stage can
dial lexicon from full → zero, on nodes and edges **independently**:

| dial | OFF (purified) | ON (lexical) | stored by |
|---|---|---|---|
| node lexicon | `type` (+`value_type`) only | + canonical-label embedding | free-text + canonical node forms |
| edge lexicon | 4-way type only | + `grounding_span` embedding | required edge `grounding_span` |
| edge structure | no edges / untyped GIN | typed GINE + the alignment binding | edge types + Task↔Value |

The **headline relational test uses the purified type-only configuration**: an abstract graph (no
lexicon) beating the matched-reasoning text baseline is an unambiguously *structural* claim, since text
has all the lexicon and the graph has none. The lexical cells are the "how much is just words?" control.
This is the construction that answers the bag-of-words concern. Canonicalisation (bead `ghm`) is run
because the graph-stats route and the lexical `label_bag` cell need comparable labels — but it does
**not** by itself de-lexicalise the graph; only the type-only encoding does.

## Consequences

- **Edge `grounding_span` is dual-purpose:** auditability + reliability *and* an optional edge feature
  for the edge-lexicon ablation cell. **Edge affective valence ("reluctantly"/"happily"/"confidently"
  ceding) is subsumed here, not a separate field** — the affective cue rides in the span text when
  present, with zero forced-judgment/confabulation cost when absent (the v4 `Stance.valence` failure
  mode). Packaging (bead `5p4`) **must preserve the edge `grounding_span`**, or the cell is impossible.
- **Known limitation — edge-type ↔ node-type redundancy (the ADR-0004 echo):** edge type is nearly
  determined by endpoints (`RETAINED_BY`→Self, `SERVES`→Value, `CEDED/SHARED`→AI). A bag-of-(node-type,
  target) histogram reconstructs most edge types. The load-bearing *edge* information is therefore
  (a) the alignment **binding** (which Task↔which Value), (b) ceded-vs-shared (one bit, both target AI),
  and (c) optionally the edge-evidence span — **not** the bare edge type. We expect, a priori, that bare
  typed-edge wiring again adds little; the binding and the span are where any relational payoff must come
  from.
- **The graph's fair-baseline edge is thin (≈ the alignment scalar).** As designed, the graph is likely
  **under-powered to out-predict a matched-reasoning text baseline on raw F1.** This is accepted: per
  MODEL_REVIEW_2 §6.3 every branch is publishable, and the graph's defensible contributions are the
  **dissociation/interpretability** result and the **alignment-beyond-sentiment** test, not necessarily
  prediction accuracy. A thin graph *losing* to text is itself informative (see next point).
- **The text-gap is the explanatory-power meter.** The matched-reasoning text baseline sees the *entire*
  transcript — everything the thin ontology discards. The gap **(text − graph)** directly measures how
  much explanatory power lives outside the ontology; error analysis on a large gap identifies *which*
  facet is missing. Ontology expansion is therefore **evidence-gated, not speculative**.
- **Validator/test impact:** `s2_extraction/validator.py` gains `validate_selfpos_graph()` +
  `is_valid_selfpos()` (new path); the v4 `validate_graph()` and its tests are untouched. New unit tests
  in `tests/test_validator_selfpos.py`. `graph-schema.md` carries v5 as the active contract with v4
  preserved as superseded.
- **Downstream contracts:** B5 extraction (`graph-modality-sfh`) consumes this as its contract; B6
  canonicalisation (`ghm`) clusters `Value`/`Task` free-text labels; packaging (`5p4`) must preserve
  edge `grounding_span` and both node label forms.

## Alternatives considered

- **Rationale as a `Rationale` node type spanning all 6 (symmetric).** Rejected: muddies the dissociation
  (trust/efficiency would share the "why" substructure with coupling, so ablating coupling no longer
  isolates competence/identity) and drifts back toward round-1/2 bloat.
- **Drop trust/efficiency entirely (maximally pure).** Rejected: discards the *most prevalent* rationale
  (trust 78%) and blocks convergent validity on `rationales_present`.
- **Shared as two edges (Task→Self AND Task→AI).** Rejected: richer degree-2 topology but breaks the
  single-boundary-edge partition invariant and complicates breadth.
- **Edge affective valence as a categorical field** (nullable/grounded-only). Rejected for v1: sparse,
  ~90% would be "neutral", reintroducing the forced-categorical reliability trap; subsumed into the edge
  `grounding_span` instead.
- **Deeper — means-end laddering** (Task→consequence→…→Value paths). The only change that buys genuine
  *path* structure a prose embedding can't bind, and theory-grounded (Gutman). Deferred: laddering is
  notoriously unreliable to code (confabulation risk), and the Phase-6 prior is against added structure
  helping. **Pre-registered as an evidence-gated round-4 move** if the text-gap is large.
- **Wider — extra facets** (general AI-affect, trajectory, condition, norm/decision-locus, consequence).
  Mostly rejected: general AI-affect would encode the very sentiment baseline the graph must *beat*;
  consequence is redundant with `trust_reliability`; trajectory/condition are sparse and unreliable.
  `decision_locus` (self-chosen vs imposed) has real theory merit (function allocation is about *who*
  allocates) but is sparse and cohort-correlated → **deferred** as the most likely evidence-gated round-4
  width addition (a boundary-edge attribute, not a new node type).
- **Task canonicalisation by functional type** (a small task taxonomy). Rejected: correlates with
  cohort/profession → reintroduces the Phase-3 vocabulary-leakage confound, and the construct is
  deliberately content-agnostic. Type-only abstraction (discard task content) is cleaner and
  theory-aligned.
