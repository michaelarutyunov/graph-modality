# graph-schema.md

> **This is the data contract.** All modules — extraction, validation, canonicalisation, encoding — derive from this document. Any schema change requires simultaneous updates to `s2_extraction/validator.py` and all affected encoding modules. Never modify downstream code without updating this document first.

> **Two ontologies are live.** The repo now hosts two non-interoperable schemas:
> - **`selfpos` / v5 (ACTIVE for round-3 work)** — the delegation-boundary ontology (Self/AI/Task/Value).
>   Validated by `validate_selfpos_graph()`. See **§ ROUND-3 (`selfpos` / v5)** immediately below.
>   Design rationale: **ADR-0008**; construct: `docs/MODEL_REVIEW_2.md` §5–6.
> - **v4 (SUPERSEDED for round-3; retained for provenance)** — the concept-graph ontology
>   (Construct/Value/Stance/CSM). Validated by `validate_graph()`. The v4 corpus and Phase-6 verdict are
>   final (ADR-0006/0007); the v4 spec is preserved unchanged below the round-3 section.
>
> The two share no node/edge types. Pick the schema by `prompt_version` (`v5`/`selfpos_v1` → selfpos;
> `v4`/`v3`/… → v4).

---

## version history

| version | date | key changes |
|---|---|---|
| v1 | 2026-06-02 | initial extraction ontology |
| v2 | 2026-06-03 | one-shot example added |
| v3 | 2026-06-05 | two-shot examples; `CONFLICTS_WITH` added; CSM cap at 2 |
| v4 | 2026-06-10 | per-pole grounding spans; multi-span salience; new relations (SUBSUMES, IMPLIES); CSM recurrence (no cap); edge rationales; topic-neutral domain; valence definitions |
| **v5 / `selfpos`** | **2026-06-16** | **NEW ONTOLOGY (not a v4 increment). Delegation-boundary: Self/AI/Task/Value nodes; RETAINED_BY/CEDED_TO/SHARED_WITH/SERVES edges; extractor-injected anchors; per-edge verbatim grounding_span; boundary partition invariant; two deterministic readouts (breadth, alignment). ADR-0008.** |

---

## ROUND-3 (`selfpos` / v5) — delegation-boundary ontology  ·  **ACTIVE**

> Validated by `validate_selfpos_graph()` in `s2_extraction/validator.py`. Design record: **ADR-0008**.
> Construct + validity design: `docs/MODEL_REVIEW_2.md` §5–6. Independent criterion labels:
> `cache/selfpos_boundary.jsonl` (`delegation_breadth`, `rationales_present`, `boundary_talk_depth`).

### what this ontology represents

The **human–AI delegation boundary**: which work tasks a respondent **retains**, which they **cede** to
AI, which they **share**, and *why they retain what they retain* (competence/identity coupling). It is
deliberately **thin and structural** — schema bloat destroyed reliability in rounds 1–2 (see ADR-0008
and the v4 section below). Lexical content is preserved but is an **encoding-time dial**, not baked-in
node bloat: the headline relational test runs a purified type-only encoding so a graph win is
unambiguously structural rather than bag-of-words.

### entity types (`selfpos`)

Exactly four node types: `Self`, `AI`, `Task`, `Value`.

**`Self`** — structural anchor for the respondent. Singleton, id `self`. **Injected by the extractor**
(not emitted by the LLM). Grounding-exempt. Exactly one per graph.

**`AI`** — structural anchor for the AI tool(s). Singleton, id `ai`. Injected by the extractor.
Grounding-exempt. Present **iff** ≥1 `CEDED_TO`/`SHARED_WITH` edge exists.

**`Task`** — a discrete work task/activity the respondent discusses in relation to AI. LLM-emitted.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | unique within graph, e.g. `"t1"` |
| `type` | `"Task"` | yes | literal |
| `label` | string | yes | short task description |
| `grounding_span` | string | yes | verbatim respondent phrase naming/describing the task |

**`Value`** — a terminal self-value the task is coupled to (the means-end driver of retention).
LLM-emitted.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | e.g. `"v1"` |
| `type` | `"Value"` | yes | literal |
| `label` | string | yes | free-text value label (canonicalised downstream) |
| `value_type` | enum | yes | `"competence"` (skill/judgment/learning) or `"identity"` (who-they-are/authorship) |
| `grounding_span` | string | yes | verbatim respondent phrase supporting the value |

> `type` is the **entity category** (4 values). `value_type` is a sub-attribute carried **only** by
> `Value` nodes. In the purified type-only encoding, node features are the `type` one-hot with `Value`
> split by `value_type` → five categories: `Self`, `AI`, `Task`, `Value:competence`, `Value:identity`.

### relation types (`selfpos`)

All directed source → target. **Every edge carries a required verbatim `grounding_span`.**

| relation | source → target | role | extra attr |
|---|---|---|---|
| `RETAINED_BY` | Task → Self | boundary partition (kept) | `rationale_tags` |
| `CEDED_TO` | Task → AI | boundary partition (delegated) | `rationale_tags` |
| `SHARED_WITH` | Task → AI | boundary partition (collaborative; control kept) | `rationale_tags` |
| `SERVES` | Task → Value | **coupling** — why the task is retained (the ablatable "why") | — |

**`rationale_tags`** (boundary edges only) — multi-label list ⊆
`{"trust_reliability", "output_efficiency", "competence_compensate", "other"}`; may be empty. These are
the non-coupling appraisal/practical reasons. Competence/identity coupling is expressed by `SERVES` →
`Value`, **not** by tags. The labeler's `rationales_present` is reconstructed as
`{Value.value_type of SERVES targets} ∪ {boundary rationale_tags}`.

`SERVES` edges have **no** `rationale_tags`.

> **Edge `grounding_span` is dual-purpose** (ADR-0008): auditability/reliability + an optional edge
> feature for the edge-lexicon ablation. **Edge affective valence** ("reluctantly"/"happily" ceding) is
> *subsumed* here — it rides in the span text when present; there is no separate `valence` field.

### structural constraints (`selfpos`) — enforced by `validate_selfpos_graph()`

| id | constraint |
|---|---|
| S1 | node `type` ∈ {`Self`, `AI`, `Task`, `Value`} |
| S2 | exactly one `Self` node, id `self` |
| S3 | at most one `AI` node, id `ai`; `AI` present iff ≥1 `CEDED_TO`/`SHARED_WITH` edge exists |
| S4 | every `Task` has a non-empty `label` and `grounding_span` |
| S5 | every `Value` has `label`, `grounding_span`, and `value_type` ∈ {`competence`, `identity`} |
| S6 | edge `relation` ∈ {`RETAINED_BY`, `CEDED_TO`, `SHARED_WITH`, `SERVES`} |
| S7 | relation type signatures: `RETAINED_BY` (Task→Self), `CEDED_TO` (Task→AI), `SHARED_WITH` (Task→AI), `SERVES` (Task→Value) |
| S8 | **partition invariant** — every `Task` has **exactly one** boundary edge (`RETAINED_BY`\|`CEDED_TO`\|`SHARED_WITH`) |
| S9 | edge `source`/`target` ids exist in `nodes` |
| S10 | every edge has a non-empty `grounding_span` |
| S11 | `rationale_tags` only on boundary edges, ⊆ {`trust_reliability`, `output_efficiency`, `competence_compensate`, `other`}; `SERVES` carries none |
| S12 | graph has a `domain` field |

### deterministic readouts (checkable against `selfpos_boundary.jsonl`)

1. **Delegation breadth** = `(|CEDED_TO| + 0.5·|SHARED_WITH|) / |Task|`, thresholded to low/med/high;
   convergent-validated against `delegation_breadth`.
2. **Boundary–coupling alignment** = φ of the 2×2 `{retained vs ceded} × {has SERVES→Value vs not}`
   over `Task` nodes. Positive φ ⇒ retained tasks are the competence/identity-coupled ones. The boundary
   (Task→Self/AI) and coupling (Task→Value) substructures are independently ablatable (dissociation test).

### complete example (`selfpos` / v5)

```json
{
  "transcript_id": "creativity_0001",
  "domain": "AI's role in professional work",
  "split": "creatives",
  "extraction_model": "deepseek-chat",
  "prompt_version": "v5",
  "node_count": 6,
  "edge_count": 5,
  "validation_violations": [],
  "nodes": [
    {"id": "self", "type": "Self"},
    {"id": "ai", "type": "AI"},
    {"id": "t1", "type": "Task", "label": "final creative decisions",
     "grounding_span": "the creative decisions are ultimately mine"},
    {"id": "t2", "type": "Task", "label": "drafting boilerplate copy",
     "grounding_span": "I let it write the first pass of the boring stuff"},
    {"id": "v1", "type": "Value", "label": "creative authorship", "value_type": "identity",
     "grounding_span": "this is my voice, it's who I am as a writer"},
    {"id": "v2", "type": "Value", "label": "craft skill", "value_type": "competence",
     "grounding_span": "I don't want to lose the muscle of writing"}
  ],
  "edges": [
    {"source": "t1", "target": "self", "relation": "RETAINED_BY", "rationale_tags": [],
     "grounding_span": "the creative decisions are ultimately mine"},
    {"source": "t1", "target": "v1", "relation": "SERVES",
     "grounding_span": "this is my voice, it's who I am as a writer"},
    {"source": "t1", "target": "v2", "relation": "SERVES",
     "grounding_span": "I don't want to lose the muscle of writing"},
    {"source": "t2", "target": "ai", "relation": "SHARED_WITH",
     "rationale_tags": ["output_efficiency", "trust_reliability"],
     "grounding_span": "I let it write the first pass of the boring stuff but I always rewrite it"}
  ]
}
```

> `t2` is **shared** (AI drafts, the respondent keeps final control via the rewrite), so it carries a
> single `SHARED_WITH`→`ai` edge — satisfying the S8 partition invariant (every Task has exactly one
> boundary edge). `t1` is retained and coupled to both an identity and a competence `Value` (two `SERVES`
> edges). Readouts on this graph: breadth = `(0 + 0.5·1)/2 = 0.25` (→ low); alignment is positive (the
> retained task `t1` is the coupled one, the shared task `t2` is not).

### allowed type values (exhaustive, `selfpos`)

```
node.type:            Self | AI | Task | Value
value.value_type:     competence | identity
edge.relation:        RETAINED_BY | CEDED_TO | SHARED_WITH | SERVES
boundary.rationale_tags ⊆  trust_reliability | output_efficiency | competence_compensate | other
graph.split:          workforce | creatives | scientists
```

---

# v4 (SUPERSEDED for round-3; retained for provenance)

> The sections below describe the **v4 concept-graph ontology**, the corpus of record for Phase 6
> (ADR-0006/0007). It is **not** used for round-3 work and is preserved unchanged for provenance. Use
> `validate_graph()` (not `validate_selfpos_graph()`) for v4 graphs.

---

## overview

Each interview transcript produces exactly one graph. Graphs exist in two forms derived from the same extraction:

| form | label style | primary use |
|---|---|---|
| free-text | labels as extracted by the LLM | GIN encoding (route 3); qualitative inspection |
| canonical | labels mapped via `canonical_map.json` | graph statistics (route 2); cross-respondent comparison |

Both forms share identical structure. Only node label strings differ.

---

## entity types

### Construct

A bipolar cognitive dimension used to evaluate the domain under discussion. Each pole must be independently grounded in the transcript. `bipolarity_complete` is `true` **only** when both `grounding_spans_positive` and `grounding_spans_negative` are non-empty. Do not invent the opposite pole — if only one pole has evidence, leave the other list empty and set `bipolarity_complete` to `false`.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | unique within graph, e.g. `"n1"` |
| `type` | `"Construct"` | yes | literal |
| `label` | string | yes | positive pole label |
| `label_negative` | string \| null | yes | negative pole label; null only if genuinely unrecoverable |
| `bipolarity_complete` | bool | yes | `true` iff BOTH `grounding_spans_positive` and `grounding_spans_negative` are non-empty |
| `grounding_spans_positive` | list[string] | yes | verbatim [Human] phrases supporting the positive pole |
| `grounding_spans_negative` | list[string] | yes | verbatim [Human] phrases supporting the negative pole; empty list if ungrounded |

Total salience for a Construct = `len(grounding_spans_positive) + len(grounding_spans_negative)`.

### Value

A terminal motivational state that constructs serve. High-abstraction anchor. Functions as a hub node in well-formed graphs. Salience is reflected by the number of distinct grounding spans.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | |
| `type` | `"Value"` | yes | literal |
| `label` | string | yes | |
| `label_negative` | — | no | omit entirely |
| `grounding_spans` | list[string] | yes | verbatim [Human] phrases; single-element list if mentioned once |

### Stance

A valenced attitude position toward some aspect of the domain. Encodes affective register, not content.

**Valence taxonomy (operational definitions):**

| value | definition |
|---|---|
| `positive` | favourable disposition toward the object |
| `negative` | unfavourable disposition toward the object |
| `mixed` | different valence toward different aspects of the same object |
| `ambivalent` | simultaneous conflicting valence toward the same aspect |

When uncertain, prefer `"mixed"` over `"ambivalent"` — ambivalent requires evidence of simultaneous conflict.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | |
| `type` | `"Stance"` | yes | literal |
| `label` | string | yes | |
| `valence` | enum | yes | one of: `"positive"`, `"negative"`, `"mixed"`, `"ambivalent"` |
| `grounding_spans` | list[string] | yes | verbatim [Human] phrases; longer lists = higher salience |

### CognitiveStyleMarker

A stable processing tendency — **how** the person reasons, not **what** they care about. To qualify as a CSM, the pattern must appear across at least **two different [Human] turns** (recurrence as evidence of "stable tendency"). Each CSM's `grounding_spans` must include at least one span from each of two distinct turns. **There is no fixed ceiling on CSM count** — extract every pattern that meets the recurrence test (v4 change from v3's max-2 cap).

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | |
| `type` | `"CognitiveStyleMarker"` | yes | literal |
| `label` | string | yes | |
| `grounding_spans` | list[string] | yes | ≥2 spans from ≥2 different [Human] turns |

---

## relation types

All relations are directed source → target unless noted. Every edge requires a `rationale` field (one-line explanation). For `CONFLICTS_WITH`, the rationale must quote the [Human] span showing the conflict.

| relation | source type | target type | directionality | meaning |
|---|---|---|---|---|
| `SERVES` | Construct | Value | directed | this construct is instrumental to this terminal state |
| `EXPRESSED_VIA` | Stance | Construct | directed | this valenced position is expressed through this construct |
| `MODULATED_BY` | Construct **or** Stance | CognitiveStyleMarker | directed | this construct/stance is shaped by a cognitive processing tendency |
| `CONFLICTS_WITH` | Construct | Construct | undirected | these constructs are in explicit tension; rationale must quote the conflict span |
| `SUBSUMES` | Value | Value | directed (specific → broader) | the source value is a specific instance or component of the target value |
| `IMPLIES` | Construct | Construct | directed | the source construct logically entails or presupposes the target construct |

`MODULATED_BY` was restricted to Construct → CSM in v3. v4 extends it to Stance → CSM as well.

`SUBSUMES` and `IMPLIES` are new in v4. They create genuine topological variance between nodes of the same type, breaking the tripartite-star determinism of v3 where every relation had a fixed source/target type signature.

### edge attributes

Every edge carries two required fields beyond `source`/`target`/`relation`:

| field | type | notes |
|---|---|---|
| `rationale` | string | one-line explanation of why the relation holds |
| `grounding` | enum | `"explicit"` or `"inferred"` — the evidentiary basis of the edge |

**`grounding` taxonomy (operational):**

| value | definition | rationale requirement |
|---|---|---|
| `explicit` | the relation itself is stated or directly quotable in a [Human] span | rationale quotes that span |
| `inferred` | the relation is an inference the analyst draws, licensed by the endpoint nodes' grounding spans | rationale must cite the grounding spans of **both** endpoint nodes that license the inference |

`grounding` is the **ablation lever** for the mental-model question (ADR-0004): downstream encoders can include distributional features only, then add `explicit` edges, then add `inferred` edges, measuring at which rung signal stops increasing. It also bounds the confabulation risk of the inferential relations — `SUBSUMES` and `IMPLIES` are almost always `inferred` and must therefore cite both endpoint spans, making each such edge auditable against the transcript.

`CONFLICTS_WITH` is `explicit` by construction (its rationale already quotes the conflict span).

---

## structural constraints

These are enforced by `s2_extraction/validator.py`. Violations are logged and the graph is flagged; extraction continues.

| id | constraint | enforcement |
|---|---|---|
| C1 | every node must have grounding spans: Constructs use per-pole lists; other types use `grounding_spans` list | validator checks fields non-empty |
| C2 | `bipolarity_complete` must be consistent: `true` only when both pole span lists are non-empty | validator flags inconsistency |
| C3 | CSM nodes must have ≥2 `grounding_spans` from different [Human] turns; no ceiling on total CSM count | validator checks `len(grounding_spans) >= 2`; removed v3 ceiling |
| C4 | no direct Stance → Value edges | validator checks all edges; rejects this source/target type combination |
| C5 | all edge `source` and `target` ids must exist in `nodes` | validator checks referential integrity |
| C6 | `valence` on Stance must be one of the four allowed values | validator checks enum membership |
| C7 | every edge must have a non-empty `rationale` | validator flags missing rationale |
| C10 | every edge must have `grounding` ∈ {`explicit`, `inferred`} | validator checks field present and enum membership |
| C8 | relation type signatures enforced: `SERVES` (Construct→Value), `EXPRESSED_VIA` (Stance→Construct), `MODULATED_BY` (Construct\|Stance→CSM), `SUBSUMES` (Value→Value), `IMPLIES` (Construct→Construct) | validator checks source/target types per relation |
| C9 | graph must have a `domain` field | validator flags missing domain |

---

## metadata block

Required on every graph, populated by `extractor.py`.

| field | type | notes |
|---|---|---|
| `transcript_id` | string | matches source CSV `transcript_id` column |
| `domain` | string | domain under discussion (e.g. "AI's role in professional work"); new in v4 |
| `split` | string | one of: `"workforce"`, `"creatives"`, `"scientists"` |
| `extraction_model` | string | e.g. `"claude-sonnet-4-6"`, `"deepseek-chat"` |
| `prompt_version` | string | e.g. `"v4"` — matches filename in `s2_extraction/prompts/` |
| `node_count` | int | total nodes |
| `edge_count` | int | total edges |
| `bipolarity_score` | float | mean of per-Construct bipolarity scores (1.0 complete, 0.5 incomplete); null if no Constructs |
| `speaker_turns_human` | int | number of Human turns in the source transcript |
| `speaker_turns_ai` | int | number of AI turns in the source transcript |
| `validation_violations` | list[string] | empty list if clean; violation strings from validator |

---

## complete example (v4)

```json
{
  "transcript_id": "work_0981",
  "domain": "AI's role in professional work",
  "split": "workforce",
  "extraction_model": "deepseek-chat",
  "prompt_version": "v4",
  "node_count": 6,
  "edge_count": 5,
  "bipolarity_score": 0.75,
  "speaker_turns_human": 8,
  "speaker_turns_ai": 9,
  "validation_violations": [],
  "nodes": [
    {
      "id": "n1",
      "type": "Construct",
      "label": "AI as comprehension aid for complex content",
      "label_negative": "Independent struggle with difficult material",
      "bipolarity_complete": true,
      "grounding_spans_positive": [
        "Reading research papers and summarizing contents that i don't quite understand"
      ],
      "grounding_spans_negative": [
        "sometimes I just have to sit there and work through it on my own"
      ]
    },
    {
      "id": "n2",
      "type": "Construct",
      "label": "Workplace-sanctioned AI use",
      "label_negative": null,
      "bipolarity_complete": false,
      "grounding_spans_positive": [
        "my colleagues have asked me to use AI if i'm struggling on a section during a meeting"
      ],
      "grounding_spans_negative": []
    },
    {
      "id": "n3",
      "type": "Value",
      "label": "Epistemic understanding",
      "grounding_spans": [
        "ask it to resummarize it for me",
        "ask targeted questions to figure it out"
      ]
    },
    {
      "id": "n4",
      "type": "Value",
      "label": "Career success and professional standing",
      "grounding_spans": [
        "it matters that my colleagues see my work as reliable and careful"
      ]
    },
    {
      "id": "n5",
      "type": "Stance",
      "label": "Wariness toward AI due to hallucination experience",
      "valence": "negative",
      "grounding_spans": [
        "AI has also hallucinated citations for an entire paragraph"
      ]
    },
    {
      "id": "n6",
      "type": "CognitiveStyleMarker",
      "label": "Iterative targeted questioning",
      "grounding_spans": [
        "If that summary still doesn't help me understand, I'll ask targeted questions",
        "I keep refining my question until I get what I need"
      ]
    }
  ],
  "edges": [
    {
      "source": "n1", "target": "n3", "relation": "SERVES", "grounding": "inferred",
      "rationale": "Using AI to understand complex content ('summarizing contents that i don't quite understand') serves epistemic understanding ('ask targeted questions to figure it out')"
    },
    {
      "source": "n5", "target": "n1", "relation": "EXPRESSED_VIA", "grounding": "inferred",
      "rationale": "Wariness about hallucinations ('hallucinated citations for an entire paragraph') is expressed through the comprehension construct"
    },
    {
      "source": "n1", "target": "n6", "relation": "MODULATED_BY", "grounding": "inferred",
      "rationale": "Comprehension strategy is shaped by iterative questioning style ('keep refining my question until I get what I need')"
    },
    {
      "source": "n3", "target": "n4", "relation": "SUBSUMES", "grounding": "inferred",
      "rationale": "Epistemic understanding ('ask targeted questions to figure it out') is a component of broader career success ('it matters that my colleagues see my work as reliable and careful') — both endpoint spans link understanding to professional standing"
    },
    {
      "source": "n2", "target": "n1", "relation": "IMPLIES", "grounding": "inferred",
      "rationale": "Workplace sanction of AI use ('my colleagues have asked me to use AI if i'm struggling') entails using AI for comprehension ('summarizing contents that i don't quite understand') — sanctioned use presupposes the comprehension task"
    }
  ]
}
```

---

## allowed type values (exhaustive)

```
node.type:       Construct | Value | Stance | CognitiveStyleMarker
edge.relation:   SERVES | EXPRESSED_VIA | MODULATED_BY | CONFLICTS_WITH | SUBSUMES | IMPLIES
stance.valence:  positive | negative | mixed | ambivalent
graph.split:     workforce | creatives | scientists
```

Any value outside these sets is a validation error.

---

## v3 → v4 migration summary

| aspect | v3 | v4 |
|---|---|---|
| Construct grounding | `grounding_span` (single string) | `grounding_spans_positive` + `grounding_spans_negative` (lists) |
| Value/Stance/CSM grounding | `grounding_span` (single string) | `grounding_spans` (list, salience = count) |
| CSM limit | max 2 per graph | no ceiling; ≥2 spans from different turns required |
| Bipolarity | LLM infers opposite pole | both poles must have independent grounding spans |
| CONFLICTS_WITH | "use sparingly" | explicit tension evidence required; rationale must quote span |
| Edges | no rationale | `rationale` + `grounding` (explicit\|inferred) required on every edge |
| New relations | — | `SUBSUMES` (Value→Value), `IMPLIES` (Construct→Construct) |
| MODULATED_BY source | Construct only | Construct or Stance |
| Domain | hardcoded "AI's role" in ontology | topic-neutral definitions; `{domain}` parameter + `domain` output field |
| Valence | enum labels only | operational definitions for mixed vs ambivalent |

---

## change protocol

1. Update this document
2. Update `s2_extraction/validator.py` to enforce or relax the changed constraint
3. Update the extraction prompt in `s2_extraction/prompts/` (increment version)
4. If node fields change: update `s4_encoding/graph_dataset.py` (node feature construction)
5. If the change affects canonicalised graphs: re-run `s3_canonicalisation/clusterer.py` and treat as a new experiment
6. Record the change and rationale in `.claude/context/extraction-log.md`
