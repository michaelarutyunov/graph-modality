# Visual Journey — `cdt-graph-modality`

> A production spec for a series of self-contained HTML visualisations that narrate the whole
> investigation, from "is a concept graph a distinct modality?" to the triangulated v4 verdict
> "yes — but distributionally, not relationally."
>
> **Status of this doc:** structure proposal + content brief. One HTML is already built
> (`v4-relational-last-look.html`, the finale). The rest are specified here, to be built on demand.

---

## 1. The narrative spine

One sentence per act, so every chapter knows the story it is advancing:

> **The thesis** (concept-graph structure carries signal text cannot) survives a clean build
> — extraction, canonicalisation, encoding — and *appears* to win on the first targets
> (**Act I–II**). Then the wins turn out to be **unfair** — circular labels, lexical leakage,
> deterministic edges (**Act III, the reckoning**). Rebuilt on a fair target (`stance_ambivalence`)
> and a fair edge ontology (v4), the honest answer emerges: the graph **is** a distinct modality,
> but its signal is **distributional node-attribute, not relational/topological** (**the verdict**).

The series is designed so a reader can stop after the cover and still get the arc, or read all
nine and understand *why* each methodological turn was forced.

---

## 2. Shared design system

All files follow `~/.claude/DESIGN.md` (the "Interview Engine Visual System"). To make the series
feel like one artifact, every HTML reuses the following **fixed conventions** — do not re-invent
per chapter.

### 2.1 Ground rules (from DESIGN.md)
- Warm ground `#F2F1ED`, 40px background grid at 2.5% ink, panels `#FFFFFF`, near-black ink `#14171C`.
- **Inter Tight** for prose, **JetBrains Mono** for every identifier/number/label.
- Flat surfaces; only `subtle` (`0 1px 0 rgba(20,23,28,0.04)`) and `focus`/`emphasis` shadows.
- Border-radius from the scale only (4 / 6 / 7 / 10 / 12 / 50%). Never pure black/white.
- Color encodes **meaning**, never decoration.

### 2.2 Recurring components (copy the markup/CSS from `v4-relational-last-look.html`)
| Component | Role | Where it appears |
|---|---|---|
| **Masthead** (eyebrow → h1 → subtitle → mono meta-chips) | Chapter opener | every file |
| **Chapter nav footer** (← prev · "chapter N of 9" · next →) | Series cohesion | every file |
| **Section head** (mono num + title + right-aligned status tag) | Section divider | every file |
| **Bar-with-chance-line** | macro-F1 / metric bars, chance marker | results chapters (04–09) |
| **Contrast table** (contrast · Δ · CI · PASS/FAIL badge) | paired comparisons | 05, 08, 09 |
| **Verdict callout** (terracotta = caution, sage = verdict) | takeaways | every file |
| **Concept-graph SVG** (typed nodes + coloured edges) | the data object | 02, 03, 06, 07, 09 |
| **Modality-vector glyph** (graph → coloured fixed-width vector) | encoding metaphor | 04, 05 |
| **Gate / pipeline strip** (steps + arrows, async tint for gated step) | process flow | 00, 02, 05, 09 |

### 2.3 Fixed palettes (assign once, use everywhere)

**Modalities** — each modality keeps one colour across the whole series:
| Modality | Token | Hex | Rationale |
|---|---|---|---|
| text (SBERT 768-d) | `group_intake` | `#5B6471` | the "given" input |
| graph stats (30-d) | `group_graph` | `#4F6E8C` | the structural winner |
| GIN / full_gin (128-d) | `group_state` | `#6B5C8A` | learned/computed |
| label_bag (384-d) | `group_generation` | `#5C7A5C` | concept-semantic, no edges |

**Entity types** (node fill-stroke):
Construct → `#4F6E8C` · Value → `#7A6B4F` · Stance → coloured by **valence**
(positive `#5C7A5C` / negative `#8A5C5C` / mixed `#7A6B4F` / ambivalent `#6B5C8A`) ·
CognitiveStyleMarker → `#6B5C8A`.

**Relations** (edge colour) — reuse DESIGN.md relationship tokens, mapped to this ontology:
`CONFLICTS_WITH` `#8A5C5C` · `IMPLIES`/`SUBSUMES` `#4F6E8C` · `EXPRESSED_VIA` `#16574A` ·
`SERVES` `#5C7A5C` · `MODULATED_BY` `#6B5C8A`.

**Outcome badges:** PASS = sage `#5C7A5C` on `#E1EFEA`; FAIL = terracotta `#8A5C5C` on `#F4E6E6`.

### 2.4 File naming
`docs/viz/NN-slug.html`, two-digit prefix = reading order. The existing finale should be
**renamed** `09-relational-last-look.html` for consistency (optional but recommended).

---

## 3. The series

Nine content files + one cover. Grouped into three acts. **Core** = essential to the arc;
**Optional** = can be merged into a neighbour if a shorter series is wanted (merge hints noted).

---

### `00-cover.html` — Journey map  ·  *core*
- **Eyebrow / title:** "cdt-graph-modality" / *"Is a concept graph a distinct modality? A research journey."*
- **Role:** single-screen index + the one-paragraph arc (§1 spine). Visually maps all nine chapters
  as a numbered pipeline (use the gate/pipeline strip), each linking to its file.
- **Content:** the spine paragraph; a 3-act timeline (Foundations → First answers → Reckoning & verdict);
  the headline result as a teaser chip ("graph stats 0.433 > text 0.367 — but edges add nothing");
  a legend for the fixed modality/entity/relation palettes (this is the series' Rosetta stone).
- **Visual centrepiece:** the 9-node chapter pipeline with act-coloured group stripes.
- **Source:** `CLAUDE.md` (phase summaries), this doc.

---

## Act I — Foundations (the build)

### `01-thesis-and-dataset.html` — The question & the data  ·  *core*
- **Title:** *"A graph is not a paragraph"* — the modality hypothesis.
- **Role:** establish RQ1 (does graph structure beat text for the target?) and RQ2 (are cohort
  topologies theoretically interpretable?), and introduce the dataset.
- **Content sections:**
  1. The two research questions (RQ1 predictive, RQ2 structural) — verbatim from `CHARTER.md`.
  2. The dataset: **Anthropic Interviewer (Handa et al., 2025)** — 1,250 transcripts, 10–15 min,
     AI interviewer on claude.ai, CC-BY. Cohorts: **workforce n=1,000 · creatives n=125 · scientists n=125** (8:1:1).
  3. Why macro-F1, why imbalance is "a feature as well as a challenge."
  4. The target-agnostic principle (teaser for ch. 04): encoders describe what data *is*, not what it predicts.
- **Visual centrepiece:** cohort donut/stacked bar (8:1:1) + a "text vs graph" split panel posing the question.
- **Source:** `docs/CHARTER.md` (RQ section, dataset §, evaluation philosophy).

### `02-extraction-and-ontology.html` — Transcript → concept graph  ·  *core*
- **Title:** *"Turning conversation into structure."*
- **Role:** how a raw interview becomes a typed graph; introduce the ontology (the data contract).
- **Content sections:**
  1. Speaker tagging (`[AI]` / `[Human]`; the three prefixes `Assistant:`/`AI:`/`User:` gotcha).
  2. **The ontology** — 4 entity types (`Construct`, `Value`, `Stance`, `CognitiveStyleMarker`)
     and 6 relations (`SERVES`, `EXPRESSED_VIA`, `MODULATED_BY`, `CONFLICTS_WITH`, `SUBSUMES`, `IMPLIES`).
     Show the node/edge JSON schema as a contract.
  3. **Prompt evolution** v1 → v2 → v3 (two-shot) → **v4** (per-pole grounding spans, edge rationales,
     CSM recurrence, valence definitions). Versioned files, never deleted.
  4. Multi-model extraction (DeepSeek default; Claude/Agnes comparison) + the validator (structural checks).
  5. Corpus stats: **0 failures, 0.3% violation rate, mean 14.9 nodes / 13.6 edges**.
- **Visual centrepiece:** a real ~10–14 node graph (pull a representative file from
  `s1_data/graphs/v4_think/canonical/`) rendered with the typed-node / coloured-edge SVG style,
  beside the transcript snippet it came from. Ontology legend below.
- **Source:** `.claude/context/graph-schema.md`, `.claude/context/extraction-log.md`, `s2_extraction/`,
  results-log §Phase 1.
- **Merge hint:** can absorb `03` if a 6-file series is wanted.

### `03-canonicalisation.html` — Taming the vocabulary  ·  *optional (merge into 02)*
- **Title:** *"15,753 ways to say the same thing."*
- **Role:** why free-text labels must collapse to a shared vocabulary before modelling, and the lock principle.
- **Content sections:**
  1. The problem: thousands of near-synonym free-text labels across 4 entity types.
  2. Method: SBERT embeddings → AgglomerativeClustering, cosine distance, **threshold 0.35**.
  3. v3 result: **1,271 canonical labels from 15,753 free-text; 100% coverage (18,662 nodes)**.
     v4 result: **5,649 canonical from 21,815 free-text** (same threshold).
  4. **Lock before modelling** — `canonical_map_v4.json` immutable; re-canonicalise = new experiment.
- **Visual centrepiece:** a synonym cluster visibly collapsing into one canonical node (small-multiples
  of 4–5 free-text strings → 1 canonical chip), with the cosine-distance dendrogram cut at 0.35.
- **Source:** `s3_canonicalisation/`, results-log §Phase 2, ADR-0002.

### `04-encoding-modalities.html` — One graph, four vectors  ·  *core*
- **Title:** *"What does the data IS-ness look like?"* — the four frozen modalities.
- **Role:** the heart of the architecture — every modality is a frozen vector; classifiers learn, encoders don't.
- **Content sections:**
  1. The target-agnostic doctrine (ADR-0003): same embedding serves any target without retraining.
  2. The four encoders, each as a modality-vector glyph in its fixed colour:
     - **text** — SBERT `all-mpnet-base-v2`, **768-d**, human-only turns.
     - **graph stats** — 30-d deterministic NetworkX features (density, type counts, stance-valence
       distribution, centrality) — *zero label semantics*.
     - **GIN / full_gin** — self-supervised GIN autoencoder, **128-d**, node features = type one-hot + label embedding.
     - **label_bag** — pooled MiniLM label embeddings, **384-d**, *no edges*.
  3. The two axes that will matter later: **feature axis** (structure-only ↔ +label-embeddings) and
     **edge axis** (no edges → untyped → typed). Plant these here; pay off in ch. 09.
- **Visual centrepiece:** one concept graph fanning out into four labelled vectors (768 / 30 / 128 / 384),
  each tinted with its modality colour; annotate which use edges vs labels vs neither.
- **Source:** `s4_encoding/*`, ADR-0003, `docs/ENGINEERING.md` §encoding.

---

## Act II — First answers (and a hidden flaw)

### `05-classification-and-confound.html` — Modelling, fusion & the interviewer confound  ·  *core*
- **Title:** *"The first wins — and the bug that explained them."*
- **Role:** Phase 3–5 results on the *original* targets, the confound that inflated them, and the fusion zoo.
- **Content sections:**
  1. The fixed split: stratified **875/187/188 (seed=42)**; test held out till the end.
  2. **The interviewer confound** (the chapter's drama): AI opening turns leaked cohort-specific
     vocabulary → labels leaked into text. Fix = strip AI turns. Show before/after.
  3. Phase-3 cohort results: text-only **0.8228**, text+stats **0.8390 (+0.016)**, text+GIN **0.8368**,
     and the eyebrow-raiser **GIN-only 0.8434** (graph structure alone beats text).
  4. The fusion zoo (Phase 5): single / stacked / gated / late — all on frozen embeddings, dual backend
     (torch + sklearn). Why fusion is the honest test of complementarity.
- **Visual centrepiece:** a "routes" diagram (text / stats / GIN → fusion → classifier) using the
  modality palette; the confound shown as a leak arrow that gets cut.
- **Source:** `s5_classification/*`, results-log §Phase 3 & §Phase 5, `phase1-postmortem.md`.
- **Merge hint:** can absorb `06`.

### `06-structural-analysis.html` — RQ2: do cohort topologies tell a story?  ·  *optional (merge into 05)*
- **Title:** *"Four hypotheses about how minds wire."*
- **Role:** the interpretable, hypothesis-driven structural tests (Phase 4) — the human-readable side of RQ2.
- **Content sections:**
  1. H1 (scientists hub-and-spoke) — **significant but reversed** (lower C:V ratio).
  2. H2 (creatives negative valence) — **supported**.
  3. H3 (workforce bipolarity) — not significant (ontology ceiling).
  4. H4 (scientist cognitive style) — not significant (CSM ceiling, max 2).
  5. The gap: interpretable metrics capture only a fraction of what GIN exploits → motivates "what *is*
     the graph signal?" (sets up the reckoning).
- **Visual centrepiece:** per-cohort distribution ridgelines/box-strips for C:V ratio and negative-valence
  fraction, with effect sizes (η²) annotated; PASS/FAIL per hypothesis.
- **Source:** results-log §Phase 4, `notebooks/04_structural_analysis.py`.

---

## Act III — The reckoning & the verdict

### `07-the-reckoning.html` — Why we threw the early wins out  ·  *core*
- **Title:** *"Three reasons the wins were unfair."* — the methodological turning point.
- **Role:** the most important conceptual chapter — why honest measurement forced a restart.
- **Content sections:**
  1. **Circularity** (METHOD_REVIEW #1): `ai_adoption` was labelled by DeepSeek — the same model that
     produced the graphs. "Graph predicts target" was partly the extractor agreeing with itself.
  2. **Lexical obviousness:** `cohort` leaks profession vocabulary into text — SBERT wins by keyword,
     leaving no honest headroom for structure.
  3. **Edge determinism** (ADR-0004): v3 made edge type a deterministic function of endpoint node types,
     so the relational hypothesis was *never fairly tested*.
  4. **The fix:** a new endogenous, lexically-non-obvious target **`stance_ambivalence`**,
     independently labelled (Agnes + Haiku, neither is DeepSeek; Kimi/user adjudication), **κ=0.504**,
     severe imbalance **med 843 / low 352 / high 55** (ADR-0005). Plus the **v4 edge ontology**
     (`SUBSUMES`/`IMPLIES`/`CONFLICTS_WITH` with grounding spans) to break determinism.
- **Visual centrepiece:** a three-panel "confound autopsy" (circular loop / keyword leak / deterministic
  edge), each with its fix; then the new-target spec card with the imbalance bar.
- **Source:** ADR-0004, ADR-0005, `.claude/context/ambivalence-target.md`,
  memory `edge-signal-validity-argument-adr-0004`.

### `08-phase6-verdict.html` — The definitive v4 results  ·  *core*
- **Title:** *"A fair test, three findings."* — ADR-0006.
- **Role:** the honest Phase-6 verdict on the v4 corpus, before the final last-look.
- **Content sections (the three hypotheses):**
  1. **Modality-distinctness — SUPPORTED.** graph stats **0.433** > text **0.367**, +0.066, CI excludes 0.
  2. **Complementarity (H_fusion) — NOT SUPPORTED.** No fusion beats stats-alone (text+stats − stats =
     +0.020, CI spans 0). Graph *subsumes* text rather than complementing it.
  3. **Relational/edge (H_edge) — REJECTED.** Edge axis at chance; **label_bag (no edges) 0.402 beats
     full_gin (with edges) 0.285**; structure_only at chance.
  4. The synthesis sentence: signal is **distributional node-attribute (stance valence / concept-label
     semantics), not relational/topological.**
- **Visual centrepiece:** a **modality leaderboard** (horizontal bars with chance line: text 0.367 /
  GIN 0.351 / label_bag 0.402 / stats 0.433 / text+stats 0.453), and a 3-row hypothesis scorecard
  (SUPPORTED / NOT / REJECTED).
- **Source:** results-log §"Phase 6 — v4", ADR-0006.

### `09-relational-last-look.html` — The strongest fair retest  ·  *core · BUILT*
- **Status:** already produced as `v4-relational-last-look.html` (rename to this slug).
- **Role:** the finale — B1 (endpoint-aware conflict edges), B2 (lexical control), RGCN (per-relation
  encoder), the decision gate, and the triangulated confirmation of ADR-0006 (now ADR-0007).
- **Already contains:** decision-gate strip, the `creativity_0023` opposite-valence conflict example,
  the B1 corrective contrast table, B2 free-text≈canonical bars, the RGCN edge-ladder + capacity-match
  detail, and the two-card verdict.
- **One edit for series fit:** add the shared chapter-nav footer (← 08 · chapter 9 of 9).

---

## 4. Build order & notes

1. **Build the cover (`00`) and the verdict (`08`) + finale (`09`) first** — they carry the whole story
   if the middle chapters slip. (09 is done.)
2. Then **07 (reckoning)** — the conceptual hinge; highest explanatory value per pixel.
3. Then Act I (`01`–`04`) for the build narrative, and Act II (`05`–`06`) for the first answers.
4. **Data fidelity rule:** every number/example must come from a result JSON, results-log, an ADR,
   or a real graph file — never approximated (the finale set the standard). Cite the source file in an
   HTML comment at the top of each chapter.
5. **Shorter-series option (6 files):** merge `03→02` and `06→05`, giving cover + thesis + extraction(+canon)
   + encoding + modelling(+structural) + reckoning + verdict + finale. Note this collapses to 7; drop the
   cover for 6. Decide before building Act I.
6. **Real graph examples to harvest** (run once, cache the picked IDs in a comment): a representative
   mid-size graph for ch. 02; a vivid synonym cluster for ch. 03; high vs low ambivalence graphs for ch. 08.
   `creativity_0023` (positive "AI copywriting efficiency" ⟷CONFLICTS_WITH⟷ negative "Human creative
   originality") is already used in ch. 09.

---

## 5. Open decisions for the author/user
- **Series length:** full 10 (cover + 9) vs the 6-file compression (§4.5)?
- **Standalone vs linked:** keep each file fully self-contained (current finale style, portable) — confirmed default.
- **Rename the finale** to `09-relational-last-look.html`? (recommended for ordering.)
- **Host location:** `docs/viz/` (tracked, current) vs also exporting to the Obsidian "Claude Insights" vault?
