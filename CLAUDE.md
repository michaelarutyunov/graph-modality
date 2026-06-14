# CLAUDE.md — cdt-graph-modality

> Context governance: see `.claude/context/codified-context-principles.md`

---

## Project Identity

`cdt-graph-modality` is a research prototype testing whether **concept graphs extracted from interview transcripts constitute a structurally distinct modality** for consumer digital twin (CDT) representation. The architecture is built on **target-agnostic modality encoders**: each modality (text, graph stats, GIN graph embedding) produces a frozen vector representation that encodes what the data IS, not what it predicts. Downstream classifiers consume these fixed embeddings. This separation allows clean measurement of whether graph modalities add complementary signal over text alone. The stack is Python 3.11+ / PyTorch CPU-only / uv / Polars / Marimo. The pipeline is linear: download → extract → canonicalise → encode → classify → analyse.

---

## Repo Structure

```
s1_data/
  raw/                    # Downloaded CSVs (gitignored)
  tagged/                 # Speaker-tagged .jsonl (gitignored)
  graphs/
    free_text/            # Extracted graphs, free-text labels (gitignored)
    canonical/            # Canonicalised graphs (gitignored)
cache/                    # Embeddings + models (gitignored)
s2_extraction/
  prompts/                # Versioned prompts — tracked in git
  model_comparison/       # 3-model comparison
  tagger.py, extractor.py, validator.py
s3_canonicalisation/
  clusterer.py, apply_canonical.py
  canonical_map.json      # Locked, immutable
s4_encoding/              # Flat — no subfolders
  text_encoder.py         # SBERT 768-dim
  graph_stats_encoder.py  # 30-dim deterministic
  graph_gnn_encoder.py    # GIN autoencoder + frozen inference (train + --encode)
  graph_dataset.py        # PyG Dataset wrapper
  build_dataset.py        # Package as .npz
  _archived/              # Phase 3 task-supervised GIN
s5_classification/        # Flat — no subfolders
  split.py                # Train/val/test split
  classifiers.py          # Factory + re-exports (mlp_* files)
  mlp_single.py           # Single-modality MLP
  mlp_stacked.py          # Concat fusion MLP
  mlp_gated.py            # Gated attention MLP
  mlp_late.py             # Late ensemble MLP
  sklearn_classifier.py   # Sklearn wrapper (4 classifiers)
  train_loop.py           # PyTorch training loop
  train_config.py         # ExperimentConfig + sweeps
  train_run.py            # Config-driven runner (torch + sklearn)
  baseline.py             # Thin convenience wrapper
  analysis_feature_importance.py  # Permutation importance
  analysis_stats.py       # Stats-only report
  _archived/              # Phase 3 route3
s6_notebooks/             # Marimo notebooks (.py, tracked in git)
results/                  # Experiment outputs (gitignored)
tests/
docs/                     # CHARTER.md, ENGINEERING.md, PLAN.md
.claude/                  # Agents + context docs
```

---

## Key Files

| File | Purpose |
|---|---|
| `docs/CHARTER.md` | Research questions, ontology, evaluation philosophy, scope |
| `docs/ENGINEERING.md` | Full technical spec — pipeline, encoding, classification, setup |
| `docs/PLAN.md` | Sequenced implementation plan with phase gates and decisions |
| `.claude/context/graph-schema.md` | Graph JSON schema — data contract between extraction and encoding |
| `.claude/context/extraction-log.md` | Prompt version history, model comparison results |
| `.claude/context/results-log.md` | Experiment results summary — validation results recorded |
| **Extraction** | |
| `s2_extraction/prompts/v3.txt` | Active extraction prompt (two-shot: workforce + scientist examples) |
| `s2_extraction/prompts/v2.txt` | One-shot variant (workforce example only) |
| `s2_extraction/prompts/v1.txt` | Original prompt (no examples) — preserved, never deleted |
| `s2_extraction/validator.py` | Structural constraint checks — run after every extraction |
| **Canonicalisation** | |
| `s3_canonicalisation/canonical_map.json` | Locked canonical vocabulary — source of truth for all node type counting |
| **Encoding** | |
| `s4_encoding/text_encoder.py` | SBERT text embeddings (768-dim), human-only turns, frozen |
| `s4_encoding/graph_stats_encoder.py` | Route 2 feature vector — 30 dimensions, networkx-derived, deterministic |
| `s4_encoding/graph_dataset.py` | PyG Dataset — converts graphs to Data objects with node features |
| `s4_encoding/graph_gnn_encoder.py` | GIN autoencoder + frozen inference — self-supervised, target-agnostic (train + encode in one file) |
| **Classification** | |
| `s5_classification/split.py` | Fixed stratified 70/15/15 split (seed=42), cached split IDs |
| `s5_classification/baseline.py` | Route 1 — text-only logistic regression (Phase 3 reference) |
| `s5_classification/analysis_feature_importance.py` | Permutation importance analysis — which graph features matter |
| `s5_classification/analysis_stats.py` | Stats-only per-class report — graph topology discriminability |
| `s5_classification/route3.py` | DEPRECATED — conflated GIN+classifier; moved to _archived/ |
| `s5_classification/classifiers.py` | PyTorch classifier zoo + build_classifier() factory |
| `s5_classification/mlp_single.py` | Single-modality MLP baseline |
| `s5_classification/mlp_stacked.py` | Stacked (concatenation) fusion classifier |
| `s5_classification/mlp_gated.py` | Gated fusion with learned per-modality attention |
| `s5_classification/mlp_late.py` | Late fusion ensemble (average logits) |
| `s5_classification/sklearn_classifier.py` | Sklearn wrapper — any sklearn classifier behind Phase 5 interface |
| `s5_classification/train_loop.py` | PyTorch training loop — Trainer, TrainingConfig, curve plotting |
| `s5_classification/train_run.py` | Config-driven experiment runner — torch + sklearn backends |
| `s5_classification/train_config.py` | ExperimentConfig dataclass + sweep builders (torch, sklearn, all) |

---

## Architecture Principles

- **Cache everything.** Extraction is API-expensive; text encoding is compute-expensive. Both run once and write to `cache/` or `s1_data/graphs/`. Never re-extract or re-encode if cached output exists. Check cache before any API call.
- **Target-agnostic encoders, task-specific classifiers.** Modality encoders (SBERT, graph stats, GIN autoencoder) produce frozen vector representations that encode what the data IS, not what it predicts. Only classifiers learn per-task. This separation enables clean measurement of modality complementarity — the same graph embedding is used for cohort, AI adoption, or any future target without retraining the encoder.
- **Dual backend: torch + sklearn.** Classifiers can be PyTorch MLPs (epoch-based, early stopping) or sklearn estimators (one-shot fit). Both consume the same frozen .npz embeddings. Adding a new sklearn classifier is one import + one dict entry. The experiment runner dispatches by `backend` field in config.
- **Multi-backend extractor.** The extractor supports both Anthropic and OpenAI-compatible backends via `--backend`. DeepSeek uses the OpenAI-compatible endpoint (`deepseek-chat`) with JSON mode (`response_format={"type": "json_object"}`) — NOT the Anthropic-compatible endpoint which forces thinking mode and causes JSON truncation. Agnes uses OpenAI-compatible endpoint. Claude uses Anthropic SDK.
- **Prompts are versioned files.** Extraction prompts live in `s2_extraction/prompts/` as numbered text files (`v1.txt`, `v2.txt`, `v3.txt`). Never hardcode prompt text in Python. Active version: `v3.txt` (two-shot examples: workforce + scientist). Older versions preserved, never deleted.
- **Lock before modelling.** `canonical_map.json` is finalised and locked before any encoding or classification begins. No downstream code may modify it. If the vocabulary needs changing, re-run canonicalisation and re-encode from scratch.
- **Scripts for pipeline, Marimo for analysis.** Long-running or stateful stages (extraction, encoding, classification) are Python scripts. Interactive inspection, visualisation, and hypothesis testing are Marimo notebooks.
- **Graph schema is the contract.** The JSON schema in `.claude/context/graph-schema.md` is the stable interface between extraction and all downstream modules. Never change it without updating `s2_extraction/validator.py` and all encoding modules.
- **uv only.** Never use `pip` directly. All package management via `uv add` / `uv run`.
- **PYTHONPATH=.** All imports are absolute from repo root. No relative imports across module boundaries.

---

## Non-Negotiable Conventions

- `canonical_map.json` is **immutable post-lock** — if vocabulary changes are needed, re-run full canonicalisation pipeline and treat it as a new experiment, not a patch
- New route = update `docs/ENGINEERING.md` first, then implement. Never in reverse.
- Extraction prompt changes increment the version number — `v1.txt` → `v2.txt`. Old versions are never deleted.
- `s1_data/`, `cache/`, `results/` are gitignored — never commit graphs, embeddings, or result files
- `.env` is gitignored — API keys (Anthropic, DeepSeek, Agnes) live there only
- Test set is held out until final evaluation — no hyperparameter decisions on test performance
- Experiment results written to `results/{route}_{timestamp}.json` — never overwrite a prior result file
- All tests set env var `BEADS_DB` to a temp path — never pollute production Beads DB

---

## Build / Run / Test

```bash
uv run python s1_data/download.py                              # Download dataset (idempotent)
uv run python s2_extraction/extractor.py                       # Run extraction with DeepSeek (default, skips cached)
uv run python s2_extraction/extractor.py --backend anthropic  # Use Claude instead
uv run python s2_extraction/model_comparison/run_comparison.py # 3-model comparison (Claude/DeepSeek/Agnes)
uv run python s3_canonicalisation/clusterer.py                 # Build canonical vocabulary
uv run python s3_canonicalisation/apply_canonical.py           # Apply to all graphs
uv run python s4_encoding/text_encoder.py                      # Encode transcripts → 768-dim (caches)
uv run python s4_encoding/graph_stats_encoder.py               # Compute graph stats → 30-dim (caches)
uv run python s4_encoding/graph_gnn_encoder.py                 # Train GIN autoencoder (canonical labels)
uv run python s4_encoding/graph_gnn_encoder.py --label-source free_text  # Train free-text autoencoder
uv run python s4_encoding/graph_gnn_encoder.py --encode        # Frozen GIN inference (canonical)
uv run python s4_encoding/graph_gnn_encoder.py --encode --label-source free_text  # Free-text inference
uv run python s4_encoding/build_dataset.py                     # Package frozen embeddings → .npz
uv run python s5_classification/train_run.py                         # Torch sweep (42 experiments)
uv run python s5_classification/train_run.py --sweep sklearn         # Sklearn sweep (48 experiments)
uv run python s5_classification/train_run.py --sweep all --dry-run   # Full plan (90 experiments)
uv run python s5_classification/train_run.py --target ai_adoption    # Single-target sweep
uv run python s5_classification/baseline.py                          # Text-only LR baseline (convenience)
uv run python s5_classification/analysis_feature_importance.py       # Permutation importance analysis
uv run python s5_classification/analysis_stats.py                    # Stats-only per-class report
uv run marimo edit notebooks/01_extraction_review.py        # Graph inspection notebook
uv run marimo edit notebooks/02_graph_exploration.py        # Cohort topology notebook
uv run marimo edit notebooks/03_classification_results.py   # Results notebook
uv run marimo edit notebooks/04_structural_analysis.py      # H1-H4 structural analysis
uv run marimo edit notebooks/05_fusion_analysis.py          # Fusion experiment analysis
uv run pytest                                               # Run test suite
bd ready                                                    # Check available tasks (Beads)
```

---

## Shell Safety

**Always use non-interactive flags** — `cp`, `mv`, `rm` may be aliased to `-i` mode, which hangs agents.

```bash
cp -f src dst      # NOT: cp src dst
mv -f src dst      # NOT: mv src dst
rm -f file         # NOT: rm file
rm -rf dir         # NOT: rm -r dir
```

---

## Issue Tracking (Beads)

This project uses `bd` (Beads) for all task tracking.

> Issues live in a local Dolt DB (`.beads/dolt/`); sync via `bd dolt push/pull`.
> `.beads/issues.jsonl` is a passive export, not the source of truth.
> See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md) for anti-patterns.

- `bd prime` — run at session start for workflow context
- `bd ready` — check before starting any work
- `bd update <id> --claim` — claim a task before working on it
- `bd close <id>` — close when complete
- `bd remember "insight"` — persist cross-session notes
- Never use `TodoWrite`, `TaskCreate`, or markdown TODO lists — use `bd` exclusively
- Never create MEMORY.md files — use `bd remember` instead

### Dependency type constraint (beads)

`bd dep add` has two legal dependency types — using the wrong one causes `bd close` to fail:

| Relationship | Command | Type flag |
|---|---|---|
| Task B must complete after task A | `bd dep add B A` | *(default `blocks`)* |
| Task belongs to an epic | `bd dep add <task> <epic> -t parent-child` | `-t parent-child` |

**`blocks` between a task and an epic is rejected** — epics are containers, not sequenced work items. Always use `-t parent-child` to link tasks to their containing epic. Using the wrong type requires `bd close --force` as a workaround.

### Parallel sub-agent worktree merge

Parallel sub-agents (`TaskCreate` with `isolation: "worktree"`) write their output to an isolated git worktree, not to the main working directory. After a sub-agent completes, copy result files back manually using `cp -f` from the worktree path printed in the sub-agent result into the corresponding module directory in the main repo.

Every bead that dispatches parallel sub-agents must include an explicit **Merge step** in its acceptance criteria listing which files to copy back.

---

## Agent Trigger Table

| File pattern | Specialist agent |
|---|---|
| `.claude/context/graph-schema.md`, `s2_extraction/validator.py` | `.claude/agents/schema-guardian/AGENT.md` |
| `s2_extraction/**` | `.claude/agents/extraction-specialist/AGENT.md` |
| `s3_canonicalisation/**` | `.claude/agents/canonicalisation-specialist/AGENT.md` |
| `s4_encoding/**` | `.claude/agents/encoding-specialist/AGENT.md` |
| `s5_classification/**`, `s6_notebooks/**`, sklearn integration | `.claude/agents/analysis-specialist/AGENT.md` |

> All 5 planned agents are now active.

---

## Context Documents

| Topic | Path |
|---|---|
| Research charter | `docs/CHARTER.md` |
| Engineering guide | `docs/ENGINEERING.md` |
| Implementation plan | `docs/PLAN.md` |
| Graph JSON schema | `.claude/context/graph-schema.md` |
| Extraction log | `.claude/context/extraction-log.md` |
| Governance principles | `.claude/context/codified-context-principles.md` |
| Results log | `.claude/context/results-log.md` |
| Phase 1 post-mortem | `.claude/context/phase1-postmortem.md` |
| Demographic variable selection | `.claude/context/demographic-variables.md` |

---

## Current Phase

**Phases 1–5 complete. v4 edge-modality + Phase 2.6 kill-criterion ran and returned a
mostly-negative result on the old targets. Now retesting the original complementarity
hypothesis on a new, lexically-non-obvious primary target (`stance_ambivalence`).**

> The v4 / Phase 2.6 detour established two things that reframe everything below:
> (1) the old targets (`cohort`, `ai_adoption`) are unfair tests — `cohort` leaks profession
> vocabulary into text (SBERT wins by keyword), and `ai_adoption`'s graph→target signal is
> partly **circular** (same DeepSeek model produces both graph and label);
> (2) on `ai_adoption` the typed-edge encoder (GINEConv) did **not** beat a bag-of-node-types
> histogram (null-ladder FAIL, Δ=+0.003, CI spans 0). See ADR-0004 and `results-log.md`.
> The response is a new endogenous, **lexically-non-obvious** target, `stance_ambivalence`,
> independently labelled (Agnes + Haiku, neither is DeepSeek; user-adjudicated disagreements),
> on which the core hypotheses get a fair retest.

### Phase 1 ✅ — Extraction (complete)
- 1,250 transcripts extracted with DeepSeek (deepseek-chat, OpenAI-compatible endpoint, JSON mode)
- Prompt v3 active (two-shot examples: workforce + scientist)
- 0 failures, 0.3% violation rate, mean 14.9 nodes / 13.6 edges per graph
- Model comparison complete; post-mortem at `.claude/context/phase1-postmortem.md`

### Phase 2 ✅ — Canonicalisation (complete)
- 1,271 canonical labels from 15,753 free-text labels across 4 entity types
- AgglomerativeClustering, cosine distance, threshold=0.35
- 100% coverage: all 18,662 nodes mapped
- `canonical_map.json` locked (2026-06-08)

### Phase 3 ✅ — Encoding + Classification (complete)
- Text embeddings: 1,250 × 768-dim cached (`all-mpnet-base-v2`, human-only turns)
- Train/val/test split: 875/187/188 stratified (seed=42)
- **Interviewer confound found and fixed** — AI turns stripped (cohort-specific openings leaked labels)
- Route 1 (text-only): test macro-F1 = 0.8228
- Route 2 (text+stats): test macro-F1 = 0.8390 (Δ = +0.0161)
- Route 3 (text+GIN): test macro-F1 = 0.8368 (Δ = +0.0139)
- Key finding: GIN-only (0.8434) beats text-only — graph structure alone is a strong signal
- Demographic classification: AI adoption shows graph upside (+4.3pp); career stage not viable

### Phase 4 ✅ — Structural Analysis (RQ2) (complete)
- H1 (scientist hub-and-spoke): Significant but reversed — scientists have lower C:V ratio
- H2 (creative negative valence): Supported — creatives highest negative-stance fraction
- H3 (workforce bipolarity): Not significant — ceiling effect from ontology constraint
- H4 (scientist cognitive style): Not significant — CSM count ceiling (max 2)
- AI adoption exploratory: only C:V ratio differentiates tool_user vs integrated
- Full results in `.claude/context/results-log.md` and `s6_notebooks/04_structural_analysis.py`

### Phase 5 ✅ — Target-Agnostic Modality Fusion (complete)
- Frozen-embedding fusion harness built: GIN autoencoder (self-supervised), `.npz` dataset
  packaging, classifier zoo (single / stacked / gated / late), config-driven runner.
- **Key design change:** Old `s4_encoding/gnn/model.py` and `train.py` (task-supervised GIN) are DEPRECATED (moved to `s4_encoding/_archived/`).
  New: `s4_encoding/graph_gnn_encoder.py` (self-supervised GIN, train+encode in one file) + `s5_classification/` (task-specific classifiers, flat directory).

### v4 edge modality + Phase 2.6 kill-criterion ✅ — ran, mostly negative on old targets
- **v4 auditable-edge ontology** (ADR-0004): v3 made edge-type a deterministic function of
  endpoint node types, so the relational hypothesis was never fairly tested. v4 adds
  `SUBSUMES` / `IMPLIES` / `CONFLICTS_WITH` (with grounding spans) to break that determinism.
  Active prompt `s2_extraction/prompts/v4.txt`; all 1,250 transcripts re-extracted (v4_think).
- **Null-ladder (GINEConv vs bag-of-types histogram) on `ai_adoption`: FAIL** —
  Δ=+0.0028, CI=[−0.017, +0.023]. Typed edge wiring does not beat node-type counts here.
- **structure_only > chance on `ai_adoption`: PASS** (+0.20) — but flagged **circular**
  (DeepSeek labels its own graphs).
- Verdict: old targets cannot deliver an honest topology claim → introduce a new target.

### Phase 6 🔜 — Complementarity retest on `stance_ambivalence` (NEW PRIMARY TARGET)
Back to the original thesis, now on a fair target. Two hypotheses to test once the new labels
land (`cache/ambivalence.jsonl`, currently being regenerated + adjudicated):

- **H_fusion (primary):** fusion(text + graph) macro-F1 > max(text-only, graph-only) on
  `stance_ambivalence`. The complementarity claim — combined > either modality alone. Graph
  arm includes the **30-dim deterministic graph-stats** modality (METHOD_REVIEW calls it the
  cleanest topology evidence — zero label semantics), not just the learned GIN.
- **H_edge (secondary):** an edge-typed graph encoder adds value over a node-only encoder.
  This is a **2-D ablation, not a single ladder**, because two orthogonal confounds must be
  separated:
  - **Edge axis (ADR-0004):** `no edges → untyped edges (node-only GIN) → typed edges
    (GINEConv)`. The old null-ladder jumped straight from histogram to GINEConv, conflating
    "no structure" with "edge-typed structure"; the untyped middle rung isolates edge-type.
  - **Feature axis (METHOD_REVIEW decisive ablation, concerns #2/#4):** `structure-only node
    features (types/degree) ↔ +label-embedding node features`. Without this control a
    GINEConv win is still attackable as "pooled label text in disguise." Infra already exists
    (`structure-only GIN mode`, `label-bag baseline`, `graph-vs-labels disentanglement` from
    the Phase 2.x kill-criterion commits) — Phase 6 re-runs that matrix on the new target.
  - Capacity held ~equal across cells so a win reflects information, not parameter count.

Evaluation protocol unchanged: 10-seed frozen CI (seeds 42–51), PASS = CI excludes 0 AND mean
Δ ≥ +0.01 (`docs/method-review/00-evaluation-protocol.md`). Three publishable outcomes for
H_edge per the design spec (clean win / distributional-only / text-learns-it-cheaply).

- **Label production (in flight):** dual-backend Agnes + Haiku labeler → consensus + Cohen's κ
  → Kimi/user adjudication of the 277 disagreements (bead `r3p`, in progress).
- **Design + plan:** `docs/superpowers/specs/2026-06-12-ambivalence-target-design.md`,
  `docs/superpowers/plans/2026-06-12-ambivalence-target.md`.
- **Epic restructuring needed:** old replication epic `graph-modality-c21` was gated on a
  Phase 2.6 PASS that didn't happen; it must be re-gated on the H_edge result on the new
  target (or superseded). The ambivalence thread itself needs a tracking epic.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
