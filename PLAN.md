# PLAN.md — Implementation Plan

> **Purpose:** Sequenced implementation plan for `cdt-graph-modality`. Each phase becomes a `bd` epic with beads covering individual implementation steps. This document is the source of truth for phase ordering, prerequisites, and pre-phase decisions.
>
> **Governance:** Agent/skill/context creation follows `.claude/context/codified-context-principles.md`. Agents are created when observed failure patterns demand them (not upfront), but the trigger table in CLAUDE.md already lists planned agents for each subsystem. Context docs are created when a specification would have prevented a mistake.

---

## Phase 0 — Environment Setup ✅

**Status:** Complete.

**What was done:**
- `pyproject.toml` with 15 deps, CPU-only torch index
- Full directory tree created (`data/`, `cache/`, `extraction/`, `canonicalisation/`, `encoding/`, `classification/`, `notebooks/`, `tests/`)
- `.gitignore` with project-specific rules (data/cache/results excluded, .gitkeep preserved)
- `data/download.py` scaffolded
- `CHARTER.md` / `ENGINEERING.md` as symlinks → `.claude/context/`
- 111 packages installed, all critical imports verified
- `.env.example` template for API keys

**Growth benchmark hit:** Tier 1 ~200 lines (on target), 0 agents (before Phase 1), 3 context docs (`graph-schema.md`, `project_charter.md`, `engineering_guide.md`).

---

## Phase 1 — Extraction

**Epic:** `bd epic "Phase 1 — Extraction"`
**Gate:** Extraction model selected, prompt locked, graphs validated on 300 transcripts.

### 1.0 Prerequisites — create before Phase 1 beads begin

| Artifact | Type | Path | Rationale |
|----------|------|------|-----------|
| **graph-schema.md** | Tier 3 (context) | `.claude/context/graph-schema.md` | Already exists. Data contract between extraction and encoding — every bead in this phase references it. |
| **extraction-log.md** | Tier 3 (context) | `.claude/context/extraction-log.md` | Create now. Tracks prompt version history, model comparison results. Required by extraction-specialist agent. |
| **schema-guardian agent** | Tier 2 (agent) | `.claude/agents/schema-guardian/AGENT.md` | Create now. The graph schema is the most critical contract in the project — any schema change must propagate to `validator.py` and all encoding modules. This agent guards that contract. |
| **extraction-specialist agent** | Tier 2 (agent) | `.claude/agents/extraction-specialist/AGENT.md` | Create now. Extraction has complex domain knowledge (ontology, grounding rules, prompt iteration protocol, API retry logic) that general-purpose prompting will get wrong. This is the highest-risk subsystem. |

**Growth benchmark target (end of Phase 1):** Tier 1 ~400 lines, 2 agents, 2–3 context docs.

### 1.1 Decisions to take before Phase 1

| Decision | Owner | Options | Default | How to resolve |
|----------|-------|---------|---------|----------------|
| Agnes context window | Michael | Include or exclude from model comparison | Exclude if < 32k tokens | Check Agnes API docs; transcripts reach 27k chars |
| Initial sample size | Michael | 300 (100/split) vs full 1,250 | 300 | Decision deferred to extraction scale step (1g); start with 300 |
| Model comparison candidates | Michael | Claude / DeepSeek / Agnes | Claude + DeepSeek minimum | Agnes depends on context window verification |
| Prompt v1 review criteria | Michael | Manual rubric scoring on 5 transcripts | Use rubric from CHARTER.md §4 | Score before model comparison |

### 1.2 Beads

#### 1a. Download dataset
- **File:** `data/download.py` ✅ (exists, needs real API keys to run)
- **Acceptance:** Three CSVs present in `data/raw/interview_transcripts/`
- **Model:** haiku
- **Blocks:** everything after it

#### 1b. Speaker tagger
- **File:** `extraction/tagger.py`
- **Acceptance:** `parse_transcript()` and `format_for_extraction()` work on one real transcript; unit tests in `tests/test_tagger.py` pass
- **Model:** sonnet
- **Blocks:** 1c, 1e, 1f, 1g

#### 1c. Extraction prompt v1
- **File:** `extraction/prompts/v1.txt`
- **Acceptance:** Prompt written per engineering guide §5.2; matches ontology in graph-schema.md
- **Model:** sonnet (with Michael review)
- **Blocks:** 1d, 1e

#### 1d. Validator
- **File:** `extraction/validator.py`
- **Acceptance:** All 6 constraints (C1–C6) from graph-schema.md enforced; unit tests in `tests/test_validator.py` pass
- **Model:** sonnet
- **Blocks:** 1e, 1g

#### 1e. Manual review on 5 transcripts
- **File:** `extraction/extractor.py` (single-transcript mode)
- **Acceptance:** 5 graphs extracted and manually reviewed; rubric scores recorded in extraction-log.md; prompt iterated once if needed
- **Model:** opus (for quality review)
- **Blocks:** 1f

#### 1f. Model comparison experiment
- **Files:** `extraction/model_comparison/sample_ids.txt`, `run_comparison.py`, `rubric_scorer.py`
- **Acceptance:** 10 fixed transcripts extracted through 2–3 models; aggregate rubric scores computed; winner selected by highest score (tiebreaker: R3 bipolarity capture); decision recorded in extraction-log.md
- **Model:** sonnet
- **Blocks:** 1g

#### 1g. Scale extraction
- **File:** `extraction/extractor.py` (batch mode)
- **Acceptance:** 300 transcripts extracted (100/split, stratified); all cached in `data/graphs/free_text/`; failed extractions logged to `extraction/failed.txt`; validator run on all graphs
- **Model:** sonnet
- **Blocks:** Phase 2

---

## Phase 2 — Canonicalisation

**Epic:** `bd epic "Phase 2 — Canonicalisation"`
**Gate:** `canonical_map.json` locked and immutable. All graphs canonicalised.

### 2.0 Prerequisites — create before Phase 2 beads begin

| Artifact | Type | Path | Rationale |
|----------|------|------|-----------|
| **canonicalisation-specialist agent** | Tier 2 (agent) | `.claude/agents/canonicalisation-specialist/AGENT.md` | Create when Phase 2 starts. Clustering has domain-specific failure modes (threshold sensitivity, vocabulary drift, label ambiguity) that general prompting mishandles. Wait until Phase 1 completes so the agent can embed real failure patterns from extraction. |
| **drift_check.py** | Script | `.claude/scripts/drift_check.py` | Validate cross-references between CLAUDE.md, agents, and context docs. Referenced in codified-context-principles.md §"Drift Detection". Should exist by this point. |

**Growth benchmark target (end of Phase 2):** Tier 1 ~500 lines, 3 agents, 4–5 context docs.

### 2.1 Decisions to take before Phase 2

| Decision | Owner | Options | Default | How to resolve |
|----------|-------|---------|---------|----------------|
| Clustering distance threshold | Michael | 0.2 (conservative) / 0.3 (default) / 0.4 (aggressive) | 0.3 | Start with default; inspect cluster quality; adjust if vocabulary is too large or too small |
| Manual review criteria | Michael | Inspect all clusters vs. only ambiguous ones | Inspect all clusters | Required before locking; expected vocab sizes: 15–25 Values, 30–50 Constructs, 20–35 Stances, 8–12 CSMs |
| Expand to full 1,250? | Michael | Stay at 300 or expand | Stay at 300 initially | Expand only if Route 2 shows signal and more power is needed; defer to Phase 3 |

### 2.2 Beads

#### 2a. Build canonical vocabulary
- **File:** `canonicalisation/clusterer.py`
- **Acceptance:** Cluster assignments produced for all entity types; vocabulary sizes in expected ranges; cluster assignments inspectable
- **Model:** sonnet
- **Blocks:** 2b

#### 2b. Manual review + lock
- **File:** `canonicalisation/canonical_map.json`
- **Acceptance:** All clusters reviewed; merges/splits applied; `canonical_map.json` written and marked as locked; never modified after this point
- **Model:** opus (Michael review)
- **Blocks:** 2c

#### 2c. Apply canonical labels
- **File:** `canonicalisation/apply_canonical.py`
- **Acceptance:** All 300 graphs canonicalised and written to `data/graphs/canonical/`; every free-text label replaced with canonical equivalent; counts match
- **Model:** sonnet
- **Blocks:** Phase 3

---

## Phase 3 — Encoding + Classification

**Epic:** `bd epic "Phase 3 — Encoding + Classification"`
**Gate:** All three routes (baseline, Route 2, Route 3) evaluated. Test set touched exactly once. Results logged.

### 3.0 Prerequisites — create before Phase 3 beads begin

| Artifact | Type | Path | Rationale |
|----------|------|------|-----------|
| **encoding-specialist agent** | Tier 2 (agent) | `.claude/agents/encoding-specialist/AGENT.md` | Create when Phase 3 starts. GNN architecture, node feature construction, and graph statistics all have domain-specific constraints (388-dim features, normalisation ranges, torch_geometric API quirks). Wait until canonicalisation completes so the agent knows the actual vocabulary. |
| **analysis-specialist agent** | Tier 2 (agent) | `.claude/agents/analysis-specialist/AGENT.md` | Create when Phase 3 starts. Classification evaluation has strict rules (test set held out, stratified splits, no hyperparameter tuning on test). Also covers notebooks. |
| **results-log.md** | Tier 3 (context) | `.claude/context/results-log.md` | Create when Phase 3 starts. Records all experiment results for cross-session reference. |

**Growth benchmark target (end of Phase 3):** Tier 1 ~550 lines, 5 agents, 6–8 context docs.

### 3.1 Decisions to take before Phase 3

| Decision | Owner | Options | Default | How to resolve |
|----------|-------|---------|---------|----------------|
| Train/val/test split | Michael | 70/15/15 vs 80/10/10 | 70/15/15 stratified | Use default; seed=42; record split IDs |
| Text encoder model | Michael | `all-mpnet-base-v2` vs `all-MiniLM-L6-v2` | `all-mpnet-base-v2` (768-dim, higher quality) | Default is stronger; MiniLM only used for label embeddings in GNN |
| Route 2 classifier | Michael | Logistic regression vs MLP | Logistic regression (interpretable) | LR first for interpretability; MLP as fallback if LR underfits |
| GNN hyperparameters | Michael | Architecture, learning rate, epochs | Per engineering guide §8.3 (GIN, lr=1e-3, 50 epochs, early stopping patience=10) | Use defaults; report train/val curves |
| Expand to 1,250 transcripts? | Michael | Based on Route 2 signal | Stay at 300 | If Route 2 shows signal above baseline, expand; otherwise note as boundary condition |

### 3.2 Beads

#### 3a. Text embeddings
- **File:** `encoding/text_encoder.py`
- **Acceptance:** 300 × 768-dim embeddings cached in `cache/text_embeddings.npy`; ID cache in `cache/text_embedding_ids.json`; idempotent (loads from cache on re-run)
- **Model:** sonnet
- **Blocks:** 3b, 3c, 3d

#### 3b. Train/test split
- **File:** `classification/baseline.py` (split logic at top)
- **Acceptance:** Fixed stratified split (seed=42); split IDs saved; test set never touched until final evaluation
- **Model:** haiku
- **Blocks:** 3c, 3d, 3e

#### 3c. Route 1 — Text-only baseline
- **File:** `classification/baseline.py`
- **Acceptance:** Logistic regression trained on text embeddings; macro-F1 on validation set recorded; confusion matrix saved
- **Model:** sonnet
- **Blocks:** 3f

#### 3d. Route 2 — Text + graph statistics
- **File:** `encoding/graph_stats.py`, `classification/route2.py`
- **Acceptance:** 36-dim feature vector per transcript; LR classifier trained on concatenated [text + graph_stats]; macro-F1 on validation set recorded; permutation feature importance computed
- **Model:** sonnet
- **Blocks:** 3f

#### 3e. Route 3 — Text + GIN graph embedding
- **Files:** `encoding/gnn/dataset.py`, `encoding/gnn/model.py`, `encoding/gnn/train.py`, `classification/route3.py`
- **Acceptance:** GIN encoder produces 128-dim graph embeddings; fused classifier trained on [text + graph_emb]; macro-F1 on validation set recorded; train/val curves plotted; early stopping applied
- **Model:** sonnet
- **Blocks:** 3f

#### 3f. Final evaluation on test set
- **Files:** `notebooks/03_classification_results.py`
- **Acceptance:** All three routes evaluated on held-out test set (touched once); results written to `results/{route}_{timestamp}.json`; macro-F1 comparison table produced; confusion matrices rendered
- **Model:** sonnet
- **Blocks:** Phase 4

---

## Phase 4 — Analysis

**Epic:** `bd epic "Phase 4 — Analysis"`
**Gate:** Results interpreted, post-mortem written, all research questions addressed.

### 4.0 Prerequisites — create before Phase 4 beads begin

| Artifact | Type | Path | Rationale |
|----------|------|------|-----------|
| **phase1-postmortem.md** | Tier 3 (context) | `.claude/context/phase1-postmortem.md` | Create during Phase 4. Captures what worked, what didn't, and what to change for future experiments. |

**Growth benchmark target (end of Phase 4):** Tier 1 ~600 lines, 5 agents, 8–10 context docs.

### 4.1 Decisions to take before Phase 4

| Decision | Owner | Options | Default | How to resolve |
|----------|-------|---------|---------|----------------|
| Significance thresholds | Michael | p < 0.05 vs p < 0.01 | p < 0.05 (standard) | Use Mann-Whitney U for pairwise, Kruskal-Wallis for three-way |
| Negative result framing | Michael | Boundary condition vs method failure | Boundary condition on elicitation depth | Pre-committed to reporting regardless of outcome |
| Permutation importance top-k | Michael | Top 10 vs top 20 features | Top 10 | Report full ranking but highlight top 10 in write-up |

### 4.2 Beads

#### 4a. Feature importance analysis (Route 2)
- **File:** `notebooks/03_classification_results.py` (importance section)
- **Acceptance:** Permutation importance for all 36 graph stat features; top features identified and interpreted; bar chart produced
- **Model:** sonnet

#### 4b. Cohort topology comparison (RQ2)
- **File:** `notebooks/02_graph_exploration.py`
- **Acceptance:** H1–H4 statistical tests computed; box plots by cohort; PCA/UMAP of graph features coloured by cohort; node type distribution stacked bars
- **Model:** sonnet

#### 4c. Results write-up
- **Files:** `.claude/context/results-log.md`, `.claude/context/phase1-postmortem.md`
- **Acceptance:** All four research questions addressed; results compared across routes; post-mortem captures lessons learned; ADR created for any architectural decisions made
- **Model:** opus (Michael review)

---

## Cross-Phase Concerns

### Context infrastructure growth plan

Per `.claude/context/codified-context-principles.md` growth benchmarks:

| Phase | Tier 1 (lines) | Tier 2 (agents) | Tier 3 (docs) | Notes |
|-------|-----------------|-----------------|----------------|-------|
| Phase 0 ✅ | ~200 | 0 | 3 | Skeleton + existing docs |
| Phase 1 | ~400 | 2 (schema-guardian, extraction-specialist) | 3–4 (+ extraction-log.md) | Core extraction knowledge |
| Phase 2 | ~500 | 3 (+ canonicalisation-specialist) | 5–6 | Add drift_check.py |
| Phase 3 | ~550 | 5 (+ encoding-specialist, analysis-specialist) | 6–8 (+ results-log.md) | Full pipeline |
| Phase 4 | ~600 | 5 | 8–10 (+ phase1-postmortem.md) | Mature |

### Agent creation order (by phase trigger)

Per codified-context-principles.md: agents are created when **all three** conditions hold — domain knowledge that general prompting gets wrong, too large for CLAUDE.md, clear trigger condition. The CLAUDE.md trigger table already lists all five agents; they get filled in at the phase boundary where their subsystem becomes active.

| Agent | Created at | Why then |
|-------|-----------|----------|
| `schema-guardian` | Phase 1 start | Graph schema is the contract; needed before any extraction |
| `extraction-specialist` | Phase 1 start | Highest-risk subsystem; prompt iteration, API retry, ontology constraints |
| `canonicalisation-specialist` | Phase 2 start | Clustering domain knowledge; can embed real failure patterns from Phase 1 extraction |
| `encoding-specialist` | Phase 3 start | GNN architecture, feature construction, torch_geometric API quirks |
| `analysis-specialist` | Phase 3 start | Classification evaluation rules (test set discipline, stratified splits) |

### Skills to create

| Skill | Created at | Why |
|-------|-----------|-----|
| `/check-bd` (already exists) | Pre-Phase 1 | Review beads for ambiguity before dispatching |
| Drift check script: `.claude/scripts/drift_check.py` | Phase 1 end | Validate cross-references per codified-context-principles.md |

### ADRs to create

| ADR | Phase | Topic |
|-----|-------|-------|
| ADR-0001 | Phase 1 | Extraction model selection (after model comparison) |
| ADR-0002 | Phase 2 | Canonical vocabulary lock decision (threshold, review criteria) |
| ADR-0003 | Phase 3 | Classification architecture decisions (if non-default choices made) |

---

## Risk Register

| Risk | Phase | Likelihood | Impact | Mitigation |
|------|-------|-----------|--------|------------|
| torch_geometric install fails | Phase 0 | Low | High | ✅ Verified working |
| Agnes context window < 32k | Phase 1 | Medium | Low | Exclude from comparison; proceed with Claude + DeepSeek |
| Extraction quality poor | Phase 1 | Medium | High | Manual review gate at 5 transcripts; iterate prompt before scale |
| Graph sparsity (< 5 nodes median) | Phase 3 | Medium | Medium | Route 2 dominates; note as boundary condition |
| GNN overfitting on 300 graphs | Phase 3 | Medium | Medium | Early stopping, dropout 0.3, weight decay 1e-4; report curves |
| Negative result | Phase 4 | Possible | Low | Pre-committed to reporting; frame as boundary condition |
