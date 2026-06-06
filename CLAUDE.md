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
  model_comparison/       # 10-transcript comparison experiment
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
| `.claude/context/extraction-log.md` | Prompt version history, model comparison results *(created during Phase 1)* |
| `.claude/context/results-log.md` | Experiment results summary *(created during Phase 2)* |
| **Extraction** | |
| `extraction/prompts/v1.txt` | Active extraction prompt — version-controlled |
| `extraction/validator.py` | Structural constraint checks — run after every extraction |
| **Canonicalisation** | |
| `canonicalisation/canonical_map.json` | Locked canonical vocabulary — source of truth for all node type counting |
| **Encoding** | |
| `encoding/gnn/model.py` | GIN architecture — 388-dim node features, 128-dim graph embedding |
| `encoding/graph_stats.py` | Route 2 feature vector — 36 dimensions, networkx-derived |

---

## Architecture Principles

- **Cache everything.** Extraction is API-expensive; text encoding is compute-expensive. Both run once and write to `cache/` or `data/graphs/`. Never re-extract or re-encode if cached output exists. Check cache before any API call.
- **Prompts are versioned files.** Extraction prompts live in `extraction/prompts/` as numbered text files (`v1.txt`, `v2.txt`). Never hardcode prompt text in Python. The active version is noted in `.claude/context/extraction-log.md`.
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
uv run python extraction/extractor.py                       # Run extraction (skips cached)
uv run python extraction/model_comparison/run_comparison.py # 3-model comparison experiment
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
| `canonicalisation/**` | `.claude/agents/canonicalisation-specialist/AGENT.md` *(placeholder)* |
| `encoding/**` | `.claude/agents/encoding-specialist/AGENT.md` *(placeholder)* |
| `classification/**`, `notebooks/**` | `.claude/agents/analysis-specialist/AGENT.md` *(placeholder)* |

> Agents not yet created are placeholders — create on first observed failure in that domain.

---

## Context Documents

| Topic | Path |
|---|---|
| Research charter | `CHARTER.md` |
| Engineering guide | `ENGINEERING.md` |
| Graph JSON schema | `.claude/context/graph-schema.md` *(create on day 1)* |
| Extraction log | `.claude/context/extraction-log.md` *(create during Phase 1)* |
| Governance principles | `.claude/context/codified-context-principles.md` |
| Results log | `.claude/context/results-log.md` *(create during Phase 2)* |
| Phase 1 post-mortem | `.claude/context/phase1-postmortem.md` *(create after Phase 1)* |

---

## Current Phase

**Phase 1 — Extraction.** Not started. Environment setup and dataset acquisition first. Then: speaker tagger, extraction prompt v1, manual review on 5 transcripts, model comparison experiment (Claude / DeepSeek / Agnes) on 10 fixed transcripts. Gate: extraction model selected and prompt locked before any scale extraction begins. See `CHARTER.md` §7 (week plan) and `ENGINEERING.md` §12 (quick-start checklist).
