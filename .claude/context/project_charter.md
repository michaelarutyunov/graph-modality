# project charter: concept graphs as a distinctive modality for consumer digital twins

**status:** active  
**author:** Michael  
**version:** 1.0 — June 2026  
**companion document:** `ENGINEERING.md`

---

## 1. motivation

Millions of AI-powered consumer digital twins (CDTs) are now being built on a common foundation: LLM agents conditioned on interview transcripts or survey responses. The premise is that if you feed a model enough of what a person said, it can simulate how that person would respond to new stimuli.

The structural problem with this premise is well-documented in the CDT literature. Interview transcripts are linear sequences of tokens. LLMs consume them as such. What gets lost is *relational structure* — which concepts a respondent connects, how those connections are weighted, which values are central to their reasoning and which are peripheral. A flat text embedding of an interview captures content; it does not capture cognitive architecture.

This project tests a specific claim: that concept graphs extracted from interview transcripts constitute a structurally distinct modality — one that carries predictive signal not recoverable from text embeddings alone. If the claim holds empirically, it has implications for CDT architecture: the path toward richer behavioural representation runs through structured cognitive representations, not just larger or better-prompted language models.

### 1.1 positioning within the CDT architecture argument

The broader CDT project identifies a structural gap between attitude simulation (where LLM-based CDTs perform reasonably well) and behavioural prediction (where they fail architecturally). Concept graphs sit on the attitude side of that divide, but they are a structurally richer representation than flat text. Specifically:

- node centrality approximates construct salience in decision-making
- bridging nodes identify concepts that mediate trade-offs
- cluster structure reveals mental model segmentation
- edge types encode the *kind* of cognitive relation, not just co-occurrence

Concept graphs are therefore an intermediate representation between raw interview text and the behavioural token streams that a true twin would require. Understanding whether they add signal is a necessary step in the architecture argument.

---

## 2. research questions

**RQ1 (primary):** Do concept graphs extracted from short AI-adoption interviews carry predictive signal for professional cohort classification beyond what flat text embeddings already capture?

**RQ2 (structural):** Do the three cohorts differ in graph topology in ways that are theoretically interpretable — in node type distributions, construct centrality, and cognitive style marker prevalence?

**RQ3 (methodological):** How much does extraction model choice affect graph quality and downstream classification performance? Can quality be evaluated systematically against a structured rubric?

**RQ4 (representational):** Does a GNN-based graph encoder outperform hand-crafted graph statistics, and what does the answer imply about the information content of structural versus semantic graph properties?

---

## 3. dataset

The Anthropic Interviewer dataset (Handa et al., 2025) provides 1,250 interview transcripts across three professionally distinct cohorts: general workforce (n=1,000), creatives (n=125), and scientists (n=125). Interviews were conducted by an AI interviewer (Claude) on claude.ai, lasting 10–15 minutes each. All participants provided informed consent for public release. Data is available under CC-BY licence.

The three-way cohort split provides ground-truth labels without requiring manual annotation. The narrow topic domain — AI adoption attitudes in professional work — means graph topology differences across cohorts are theoretically interpretable rather than arbitrary.

### 3.1 class imbalance

The 8:1:1 ratio (workforce:creatives:scientists) is manageable with standard techniques. Macro-averaged F1 is the primary evaluation metric throughout. Accuracy is reported but not used for model selection or comparison.

Imbalance is treated as a feature as well as a challenge: if concept graphs help the classifier distinguish creatives and scientists from workforce better than text alone, that is a stronger finding than overall accuracy improvement, because those are the structurally distinctive cohorts.

---

## 4. ontology

The standard CDT-optimal elicitation ontology is adapted for this dataset. Three entities from the full ontology are not applicable to attitude interviews: Episode (no recalled decision events), Trade-off (no within-session tension resolution), and Context (no situational modulation elicited). One new entity — Stance — is added as domain-specific to attitude interviews.

### 4.1 entities

**Construct** — a bipolar cognitive dimension the respondent uses to evaluate or differentiate AI's role in their work. Defined by two contrasting poles with an implicit threshold between them. Both poles must be present or inferrable from the transcript. This is the primary signal-bearing entity type.

*Examples: human nuance ↔ technical output; creative control ↔ AI-driven decisions; AI reliability ↔ verification overhead*

**Value** — a terminal motivational state that constructs serve. High-abstraction anchors that function as hub nodes in well-formed graphs. Distinct from constructs — describes what the person ultimately cares about, not how they evaluate options.

*Examples: professional identity; epistemic rigour; economic security; autonomy*

**Stance** — a valenced attitude position on a specific topic. Domain-specific to this dataset; absent from the general CDT ontology. Encodes the affective register of the respondent's engagement with a construct. Each stance carries a valence attribute: positive / negative / mixed / ambivalent.

*Examples: anxious / searching; resigned acceptance with guilt; sceptical / conditional acceptance; aspirational but blocked*

**Cognitive Style Marker** — stable processing tendencies that cross domains. Describes *how* the person reasons, not *what* they care about. Distinct from both values and stances. Typically one per transcript; two is possible.

*Examples: verification-first / maximiser; loss-averse / identity-anchored; strategic / deliberate*

### 4.2 relations

| relation | direction | meaning |
|---|---|---|
| `SERVES` | Construct → Value | this construct is instrumental to this terminal state |
| `EXPRESSED_VIA` | Stance → Construct | this valenced position is expressed through this construct |
| `MODULATED_BY` | Construct → Cognitive Style Marker | this processing tendency shifts how this construct is applied |
| `CONFLICTS_WITH` | Construct ↔ Construct | these constructs are in tension (undirected, symmetric) |

`CONFLICTS_WITH` will be rare in short interviews but is the highest-value relation for behavioural prediction when it does appear.

### 4.3 structural constraints

1. **bipolarity** — every Construct must have both poles defined, or be flagged as incomplete (bipolarity score = 0.5 rather than 1.0)
2. **value mediation** — Stances must connect to Values via at least one Construct; direct Stance→Value links indicate rationalised rather than process-grounded content and are disallowed
3. **grounding** — every node must be traceable to a span of respondent text; nodes with no grounding are hallucinations and are discarded
4. **cognitive style ceiling** — maximum two Cognitive Style Markers per transcript; extractions exceeding this are flagged for manual review

### 4.4 two representations

All extracted graphs are produced in two forms:

**Free-text** — node labels preserved as extracted. Richer; better for qualitative inspection and for the GNN route where node label embeddings carry semantic content.

**Canonicalised** — semantically similar labels mapped to a shared canonical vocabulary derived from the data itself via embedding-based clustering. Cleaner; better for cross-respondent comparison and for the graph statistics route where counting node types requires consistency.

Both representations derive from the same extraction run. Canonicalisation is a post-processing step, not a separate extraction.

---

## 5. evaluation philosophy

### 5.1 primary metric

Macro-averaged F1, computed across all three cohorts with equal weight. This treats all cohorts as equally important regardless of their frequency in the dataset. Accuracy is additionally reported but never used as a basis for comparison.

### 5.2 experimental conditions

Three conditions are compared against a text-only baseline:

| condition | description |
|---|---|
| baseline | sentence-transformer text embedding only |
| route 2 | text embedding + hand-crafted graph statistics (networkx) |
| route 3 | text embedding + GNN graph embedding (torch_geometric) |

### 5.3 handling negative results

A negative result — graph features do not improve over text-only — is treated as a substantive finding rather than a failure. It would indicate that concept graphs extracted from short, topic-constrained attitude interviews do not carry signal beyond what the text already contains. This is a specific and useful claim about the limits of graph extraction from shallow interviews, and would inform decisions about minimum elicitation depth for CDT-relevant graph recovery.

---

## 6. interpretability hypotheses

Secondary analyses test four falsifiable predictions about cohort differences in graph structure. These are tested regardless of classification outcome.

**H1 — scientist hub-and-spoke:** scientists exhibit a higher Construct:Value ratio than other cohorts, with Value nodes serving as high-degree hubs. Interpretation: scientists have more constructs serving a smaller set of terminal values (epistemic rigour, data integrity), producing a structurally different graph than the workforce cohort.

**H2 — creative negative valence:** creatives exhibit a higher proportion of negative-valence Stances than other cohorts. Interpretation: the dual satisfaction/anxiety pattern documented in the Anthropic research (97% productivity gains alongside pervasive identity anxiety) should leave a structural fingerprint in stance valence distributions.

**H3 — workforce bipolarity:** workforce respondents exhibit higher bipolarity completeness scores than creatives. Interpretation: workforce respondents articulate clearer trade-offs (human value vs automatable work), whereas creatives express more ambivalent or unresolved constructs.

**H4 — scientist cognitive style:** Cognitive Style Markers are more prevalent in scientist transcripts and more likely to be verification-oriented (maximiser, evidence-anchored) than in workforce transcripts.

---

## 7. week plan

| day | focus | gate |
|---|---|---|
| 1 | environment setup; data acquisition; speaker tagger | tagged transcripts for comparison sample |
| 2 | model comparison experiment; prompt iteration; model selection | locked extraction prompt; extraction model chosen |
| 3 | scale extraction (300 transcripts); validation; canonicalisation | graph corpus ready; `canonical_map.json` locked |
| 4 | text encoding; route 2 feature engineering; baseline + route 2 classification | route 2 results |
| 5 | route 3 GNN; structural analysis (RQ2); interpretability hypotheses | route 3 results; H1–H4 tested |
| 6–7 | synthesis and writing | — |

**critical path:** day 2 gates everything. Do not begin scale extraction until the extraction prompt is locked and the model is selected.

---

## 8. scope boundaries

**in scope:**
- graph extraction, encoding, and classification as specified
- model comparison experiment (Claude, DeepSeek, Agnes)
- structural cohort analysis (RQ2)
- interpretability hypothesis testing (H1–H4)
- reproducible codebase

**out of scope for this phase:**
- behavioural data fusion (separate project)
- cross-domain construct transfer analysis (requires multi-domain dataset)
- CDT-optimal elicitation methodology design (separate project)
- fine-tuning any encoder model
- production deployment of any component

---

## 9. references

Handa, K., Stern, M., Huang, S., Hong, J., Durmus, E., McCain, M., Yun, G., Alt, A., Millar, T., Tamkin, A., Leibrock, J., Ritchie, S., & Ganguli, D. (2025). *Introducing Anthropic Interviewer: What 1,250 professionals told us about working with AI.* Anthropic. https://anthropic.com/research/anthropic-interviewer

Kritzinger, W., Karner, M., Traar, G., Henjes, J., & Sihn, W. (2018). Digital twin in manufacturing: A categorical literature review and classification. *IFAC-PapersOnLine, 51*(11), 1016–1022.

Kelly, G. A. (1955). *The psychology of personal constructs.* Norton.

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *ICLR 2019.*

Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017.*
