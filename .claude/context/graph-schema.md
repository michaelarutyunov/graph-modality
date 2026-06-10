# graph-schema.md — v4

> **This is the data contract.** All modules — extraction, validation, canonicalisation, encoding — derive from this document. Any schema change requires simultaneous updates to `s2_extraction/validator.py` and all affected encoding modules. Never modify downstream code without updating this document first.

---

## version history

| version | date | key changes |
|---|---|---|
| v1 | 2026-06-02 | initial extraction ontology |
| v2 | 2026-06-03 | one-shot example added |
| v3 | 2026-06-05 | two-shot examples; `CONFLICTS_WITH` added; CSM cap at 2 |
| **v4** | **2026-06-10** | **per-pole grounding spans; multi-span salience; new relations (SUBSUMES, IMPLIES); CSM recurrence (no cap); edge rationales; topic-neutral domain; valence definitions** |

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
