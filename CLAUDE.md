# CLAUDE.md — cdt-graph-modality

> Context governance: see `.claude/context/codified-context-principles.md`

---

## Project Identity

`cdt-graph-modality` is a research prototype testing whether **concept graphs extracted from interview transcripts constitute a structurally distinct modality** for consumer digital twin (CDT) representation. It runs three experimental conditions against a text-only baseline: text + hand-crafted graph statistics (route 2) and text + GIN graph embedding (route 3), all classifying professional cohort from the Anthropic Interviewer dataset (1,250 transcripts, workforce / creatives / scientists). The stack is Python 3.11+ / PyTorch CPU-only / uv / Polars / Marimo. The pipeline is linear: download → extract → canonicalise → encode → classify → analyse.

---

## Repo Structure

```
data/
  raw/                    # Downloaded CSVs from HuggingFace (gitignored)
  tagged/                 # Speaker-tagged transcripts as .jsonl (gitignored)
  graphs/
    free_text/            # Extracted graphs, free-text labels (gitignored)
    canonical/            # Canonicalised graphs (gitignored)
cache/                    # Encoded embeddings and trained weights (gitignored)
extraction/
  prompts/                # Versioned extraction prompts — tracked in git
  model_comparison/       # 3-model comparison + validation report
  tagger.py
  extractor.py
  validator.py
canonicalisation/
  clusterer.py
  apply_canonical.py
  canonical_map.json      # Locked after day 3 — never modified post-lock
encoding/
  text_encoder.py
  graph_stats.py
  gnn/
    dataset.py
    model.py
    train.py
classification/
  split.py
  baseline.py
  route2.py
  route3.py
notebooks/                # Marimo notebooks (.py files, tracked in git)
results/                  # Experiment results as JSON/CSV (gitignored)
tests/
.claude/
  agents/                 # Specialist agents
  context/                # Knowledge base
```

---

## Key Files

| File | Purpose |
|---|---|
| `CHARTER.md` | Research questions, ontology, evaluation philosophy, scope |
| `ENGINEERING.md` | Full technical spec — pipeline, encoding, classification, setup |
| `.claude/context/graph-schema.md` | Graph JSON schema — data contract between extraction and encoding |
| `.claude/context/extraction-log.md` | Prompt version history, model comparison results |
| `.claude/context/results-log.md` | Experiment results summary — validation results recorded |
| **Extraction** | |
| `extraction/prompts/v3.txt` | Active extraction prompt (two-shot: workforce + scientist examples) |
| `extraction/prompts/v2.txt` | One-shot variant (workforce example only) |
| `extraction/prompts/v1.txt` | Original prompt (no examples) — preserved, never deleted |
| `extraction/validator.py` | Structural constraint checks — run after every extraction |
| **Canonicalisation** | |
| `canonicalisation/canonical_map.json` | Locked canonical vocabulary — source of truth for all node type counting |
| **Encoding** | |
| `encoding/gnn/model.py` | GIN architecture — 388-dim node features, 128-dim graph embedding |
| `encoding/gnn/train.py` | GIN training loop — early stopping, class weights, LR scheduling |
| `encoding/graph_stats.py` | Route 2 feature vector — 30 dimensions, networkx-derived |
| **Classification** | |
| `classification/split.py` | Fixed stratified 70/15/15 split (seed=42), cached split IDs |
| `classification/baseline.py` | Route 1 — text-only logistic regression |
| `classification/route2.py` | Route 2 — text + graph stats LR with permutation importance |
| `classification/route3.py` | Route 3 — text + GIN with route comparison table |

---

## Architecture Principles

- **Cache everything.** Extraction is API-expensive; text encoding is compute-expensive. Both run once and write to `cache/` or `data/graphs/`. Never re-extract or re-encode if cached output exists. Check cache before any API call.
- **Multi-backend extractor.** The extractor supports both Anthropic and OpenAI-compatible backends via `--backend`. DeepSeek uses the OpenAI-compatible endpoint (`deepseek-chat`) with JSON mode (`response_format={"type": "json_object"}`) — NOT the Anthropic-compatible endpoint which forces thinking mode and causes JSON truncation. Agnes uses OpenAI-compatible endpoint. Claude uses Anthropic SDK.
- **Prompts are versioned files.** Extraction prompts live in `extraction/prompts/` as numbered text files (`v1.txt`, `v2.txt`, `v3.txt`). Never hardcode prompt text in Python. Active version: `v3.txt` (two-shot examples: workforce + scientist). Older versions preserved, never deleted.
- **Lock before modelling.** `canonical_map.json` is finalised and locked before any encoding or classification begins. No downstream code may modify it. If the vocabulary needs changing, re-run canonicalisation and re-encode from scratch.
- **Scripts for pipeline, Marimo for analysis.** Long-running or stateful stages (extraction, encoding, classification) are Python scripts. Interactive inspection, visualisation, and hypothesis testing are Marimo notebooks.
- **Graph schema is the contract.** The JSON schema in `.claude/context/graph-schema.md` is the stable interface between extraction and all downstream modules. Never change it without updating `extraction/validator.py` and all encoding modules.
- **uv only.** Never use `pip` directly. All package management via `uv add` / `uv run`.
- **PYTHONPATH=.** All imports are absolute from repo root. No relative imports across module boundaries.

---

## Non-Negotiable Conventions

- `canonical_map.json` is **immutable post-lock** — if vocabulary changes are needed, re-run full canonicalisation pipeline and treat it as a new experiment, not a patch
- New route = update `ENGINEERING.md` first, then implement. Never in reverse.
- Extraction prompt changes increment the version number — `v1.txt` → `v2.txt`. Old versions are never deleted.
- `data/`, `cache/`, `results/` are gitignored — never commit graphs, embeddings, or result files
- `.env` is gitignored — API keys (Anthropic, DeepSeek, Agnes) live there only
- Test set is held out until final evaluation — no hyperparameter decisions on test performance
- Experiment results written to `results/{route}_{timestamp}.json` — never overwrite a prior result file
- All tests set env var `BEADS_DB` to a temp path — never pollute production Beads DB

---

## Build / Run / Test

```bash
uv run python data/download.py                              # Download dataset (idempotent)
uv run python extraction/extractor.py                       # Run extraction with DeepSeek (default, skips cached)
uv run python extraction/extractor.py --backend anthropic  # Use Claude instead
uv run python extraction/model_comparison/run_comparison.py # 3-model comparison (Claude/DeepSeek/Agnes)
uv run python canonicalisation/clusterer.py                 # Build canonical vocabulary
uv run python canonicalisation/apply_canonical.py           # Apply to all graphs
uv run python encoding/text_encoder.py                      # Encode transcripts (caches)
uv run python classification/baseline.py                    # Text-only baseline
uv run python classification/route2.py                      # Text + graph stats
uv run python classification/route3.py                      # Text + GIN
uv run marimo edit notebooks/01_extraction_review.py        # Graph inspection notebook
uv run marimo edit notebooks/02_graph_exploration.py        # Cohort topology notebook
uv run marimo edit notebooks/03_classification_results.py   # Results notebook
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
| `.claude/context/graph-schema.md`, `extraction/validator.py` | `.claude/agents/schema-guardian/AGENT.md` |
| `extraction/**` | `.claude/agents/extraction-specialist/AGENT.md` |
| `canonicalisation/**` | `.claude/agents/canonicalisation-specialist/AGENT.md` |
| `encoding/**` | `.claude/agents/encoding-specialist/AGENT.md` |
| `classification/**`, `notebooks/**` | `.claude/agents/analysis-specialist/AGENT.md` |

> All 5 planned agents are now active.

---

## Context Documents

| Topic | Path |
|---|---|
| Research charter | `CHARTER.md` |
| Engineering guide | `ENGINEERING.md` |
| Graph JSON schema | `.claude/context/graph-schema.md` |
| Extraction log | `.claude/context/extraction-log.md` |
| Governance principles | `.claude/context/codified-context-principles.md` |
| Results log | `.claude/context/results-log.md` |
| Phase 1 post-mortem | `.claude/context/phase1-postmortem.md` *(create during Phase 4)* |

---

## Current Phase

**Phase 2 complete. Ready for Phase 3 — Encoding + Classification.**

### Phase 1 ✅ — Extraction (complete)
- 1,250 transcripts extracted with DeepSeek (deepseek-chat, OpenAI-compatible endpoint, JSON mode)
- Prompt v3 active (two-shot examples: workforce + scientist)
- 0 failures, 0.3% violation rate, mean 14.9 nodes / 13.6 edges per graph
- Model comparison: Claude won initial round; DeepSeek works after switching to OpenAI endpoint + JSON mode; Agnes is viable but leaner

### Phase 2 ✅ — Canonicalisation (complete)
- 1,271 canonical labels from 15,753 free-text labels across 4 entity types
- AgglomerativeClustering, cosine distance, threshold=0.35
- 100% coverage: all 18,662 nodes mapped
- `canonical_map.json` locked (2026-06-08)

### Phase 3 🔄 — Encoding + Classification (5/7 beads closed)
- Text embeddings: 1,250 × 768-dim cached (`all-mpnet-base-v2`)
- Train/val/test split: 875/187/188 stratified (seed=42)
- **Ceiling effect**: text-only baseline achieves val macro-F1 = 1.0 — cohorts trivially separable from language
- Route 1 (text-only): val macro-F1 = **1.0000**
- Route 2 (text + graph stats): val macro-F1 = **1.0000** (Δ = +0.0000)
- Route 3 (text + GIN): val macro-F1 = **0.9797** (Δ = -0.0203, early stopped epoch 26)
- Remaining: final test set evaluation (bead `ico`)


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
