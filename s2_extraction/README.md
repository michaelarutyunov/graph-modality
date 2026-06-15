# s2_extraction — file manifest

This folder holds **three distinct concerns** that accumulated across rounds. Use this table
before adding files. **Round-3 convention:** prefix all new self-positioning files `selfpos_`
(e.g. `selfpos_extractor.py`, `prompts/selfpos_v1.txt`, `selfpos_labeler.py`) so the next batch
is self-documenting and does not add to the undifferentiated root.

## Graph extraction — core pipeline

| File | Purpose | Status |
|---|---|---|
| `tagger.py` | Speaker-tag transcript turns (`Assistant:`/`AI:`/`User:`) → `s1_data/tagged/` | **active** |
| `extractor.py` | Extract concept graphs via LLM (Anthropic + OpenAI-compatible backends); cache-first | **active** |
| `validator.py` | Structural validation against the graph schema; runs after every extraction | **active** |
| `quality_report.py` | Extraction-run quality checkpoints (continue/abort decision) | **active (ops)** |
| `_v4_run.sh` | Autonomous v4_think extraction runner with checkpoints | completed one-off (provenance; candidate for `_archived/`) |

## Graph extraction — prompts (`prompts/`)

| File | Purpose | Status |
|---|---|---|
| `v4.txt` | **Active** extraction prompt (auditable-edge ontology, v4_think corpus) | **active** |
| `v3.txt` | Two-shot prompt (v3 corpus) | superseded (kept for provenance) |
| `v2.txt`, `v1.txt` | Earlier prompts (one-shot / no-shot) | superseded (kept for provenance) |
| `ds_thinking_mode.txt` | Reference: DeepSeek thinking-mode API params (used for v4_think) | reference |
| `prompt_revision.txt` | Design note: diagnosis of the deterministic-topology problem that motivated the v4 ontology | **provenance — do not delete** |

## Demographics extraction (secondary targets)

| File | Purpose | Status |
|---|---|---|
| `demographics_extractor.py` | Extract `career_stage` + `ai_adoption` via DeepSeek | active (secondary) |
| `prompts/demographics_v1.txt` | Demographics prompt | active (secondary) |

## Ambivalence labeling (Phase 6 target — `stance_ambivalence`)

| File | Purpose | Status |
|---|---|---|
| `ambivalence_labeler.py` | Dual-backend labeling (Agnes + Haiku; neither is the DeepSeek extractor → breaks circularity) | active |
| `ambivalence_consensus.py` | Merge labels → consensus + disagreement worklist | active |
| `ambivalence_adjudicator.py` | Kimi adjudication of disagreements | active |
| `prompts/ambivalence_v1.txt` | Anchored ordinal ambivalence rubric | active |

## Model comparison (`model_comparison/`)

| Path | Purpose |
|---|---|
| `model_comparison/` | 3-model graph-extraction comparison + demographics comparison (reports, sample IDs, runners) |
