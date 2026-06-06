# graph-schema.md

> **This is the data contract.** All modules — extraction, validation, canonicalisation, encoding — derive from this document. Any schema change requires simultaneous updates to `extraction/validator.py` and all affected encoding modules. Never modify downstream code without updating this document first.

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

A bipolar cognitive dimension the respondent uses to evaluate AI's role. The primary signal-bearing entity. Both poles are required — a construct with only one pole is structurally incomplete.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | unique within graph, e.g. `"n1"` |
| `type` | `"Construct"` | yes | literal |
| `label` | string | yes | positive pole label |
| `label_negative` | string \| null | yes | negative pole label; null only if genuinely unrecoverable |
| `bipolarity_complete` | bool | yes | `true` if both poles present; `false` if `label_negative` is null |
| `grounding_span` | string | yes | verbatim phrase from a Human turn supporting this node |

### Value

A terminal motivational state that constructs serve. High-abstraction anchor. Functions as a hub node in well-formed graphs.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | |
| `type` | `"Value"` | yes | literal |
| `label` | string | yes | |
| `label_negative` | — | no | omit entirely |
| `grounding_span` | string | yes | |

### Stance

A valenced attitude position on a specific topic. Domain-specific to attitude interviews. Encodes affective register, not content.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | |
| `type` | `"Stance"` | yes | literal |
| `label` | string | yes | |
| `valence` | enum | yes | one of: `"positive"`, `"negative"`, `"mixed"`, `"ambivalent"` |
| `grounding_span` | string | yes | |

### CognitiveStyleMarker

A stable processing tendency. Describes *how* the respondent reasons, not *what* they value. Maximum two per graph.

| field | type | required | notes |
|---|---|---|---|
| `id` | string | yes | |
| `type` | `"CognitiveStyleMarker"` | yes | literal |
| `label` | string | yes | |
| `grounding_span` | string | yes | |

---

## relation types

All relations are directed source → target unless noted.

| relation | source type | target type | directionality | meaning |
|---|---|---|---|---|
| `SERVES` | Construct | Value | directed | this construct is instrumental to this terminal state |
| `EXPRESSED_VIA` | Stance | Construct | directed | this valenced position is expressed through this construct |
| `MODULATED_BY` | Construct | CognitiveStyleMarker | directed | this style tendency shifts how this construct is applied |
| `CONFLICTS_WITH` | Construct | Construct | undirected | these constructs are in tension; store once in either direction |

`CONFLICTS_WITH` is rare in short interviews. Flag its presence in `extraction-log.md` — it is the highest-value relation for behavioural prediction.

---

## structural constraints

These are enforced by `extraction/validator.py`. Violations are logged and the graph is flagged; extraction continues.

| id | constraint | enforcement |
|---|---|---|
| C1 | every Construct must have `grounding_span` traceable to a Human turn | validator checks field non-empty |
| C2 | `label_negative` must be present on all Constructs, or `bipolarity_complete` set to `false` | validator flags null `label_negative` without `bipolarity_complete: false` |
| C3 | maximum 2 CognitiveStyleMarker nodes per graph | validator counts; flags if > 2 |
| C4 | no direct Stance → Value edges | validator checks all edges; rejects this source/target type combination |
| C5 | all edge `source` and `target` ids must exist in `nodes` | validator checks referential integrity |
| C6 | `valence` on Stance must be one of the four allowed values | validator checks enum membership |

---

## metadata block

Required on every graph, populated by `extractor.py`.

| field | type | notes |
|---|---|---|
| `transcript_id` | string | matches source CSV `transcript_id` column |
| `split` | string | one of: `"workforce"`, `"creatives"`, `"scientists"` |
| `extraction_model` | string | e.g. `"claude-sonnet-4-6"`, `"deepseek-v3"` |
| `prompt_version` | string | e.g. `"v1"` — matches filename in `extraction/prompts/` |
| `node_count` | int | total nodes |
| `edge_count` | int | total edges |
| `bipolarity_score` | float | mean of per-Construct bipolarity scores (1.0 complete, 0.5 incomplete); null if no Constructs |
| `speaker_turns_human` | int | number of Human turns in the source transcript |
| `speaker_turns_ai` | int | number of AI turns in the source transcript |
| `validation_violations` | list[string] | empty list if clean; violation strings from validator |

---

## complete example

```json
{
  "transcript_id": "work_0000",
  "split": "workforce",
  "extraction_model": "claude-sonnet-4-6",
  "prompt_version": "v1",
  "node_count": 4,
  "edge_count": 3,
  "bipolarity_score": 1.0,
  "speaker_turns_human": 6,
  "speaker_turns_ai": 7,
  "validation_violations": [],
  "nodes": [
    {
      "id": "n1",
      "type": "Construct",
      "label": "human value in work",
      "label_negative": "automatable work",
      "bipolarity_complete": true,
      "grounding_span": "things that humans offer to the industry that can't be automated"
    },
    {
      "id": "n2",
      "type": "Value",
      "label": "professional identity",
      "grounding_span": "what skills would be good to work on that AI can't take over"
    },
    {
      "id": "n3",
      "type": "Stance",
      "label": "anxious / searching",
      "valence": "negative",
      "grounding_span": "I'm still trying to figure out"
    },
    {
      "id": "n4",
      "type": "CognitiveStyleMarker",
      "label": "strategic / deliberate",
      "grounding_span": "really hone in on that aspect"
    }
  ],
  "edges": [
    { "source": "n1", "target": "n2", "relation": "SERVES" },
    { "source": "n3", "target": "n1", "relation": "EXPRESSED_VIA" },
    { "source": "n1", "target": "n4", "relation": "MODULATED_BY" }
  ]
}
```

---

## allowed type values (exhaustive)

```
node.type:       Construct | Value | Stance | CognitiveStyleMarker
edge.relation:   SERVES | EXPRESSED_VIA | MODULATED_BY | CONFLICTS_WITH
stance.valence:  positive | negative | mixed | ambivalent
graph.split:     workforce | creatives | scientists
```

Any value outside these sets is a validation error.

---

## change protocol

1. Update this document
2. Update `extraction/validator.py` to enforce or relax the changed constraint
3. Update the extraction prompt in `extraction/prompts/` (increment version)
4. If node fields change: update `encoding/gnn/dataset.py` (node feature construction)
5. If the change affects canonicalised graphs: re-run `canonicalisation/clusterer.py` and treat as a new experiment
6. Record the change and rationale in `.claude/context/extraction-log.md`
