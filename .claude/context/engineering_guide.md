# engineering guide: concept graphs as a distinctive modality for CDTs

**status:** active  
**author:** Michael  
**version:** 1.0 — June 2026  
**companion document:** `CHARTER.md`

---

## 1. tooling decisions

| tool | choice | rationale |
|---|---|---|
| package manager | `uv` | fast, reproducible, pyproject.toml-native |
| dataframes | `polars` | no pandas dependency; `.to_list()` and `.to_numpy()` for library interop |
| torch variant | CPU-only | graphs are 6–10 nodes; GPU provides no meaningful speedup at this scale |
| notebooks | Marimo | reactive, saves as `.py` (version-control friendly), good for graph visualisation |
| pipeline stages | scripts | extraction and encoding run once and cache; not suitable for notebook kernels |

### 1.1 script vs notebook allocation

| component | type | rationale |
|---|---|---|
| `s1_data/download.py` | script | one-shot, stateful download with existence check |
| `s2_extraction/tagger.py` | script | pure text processing, no interactivity needed |
| `s2_extraction/extractor.py` | script | long-running API batch with retry logic; kernel death is unacceptable |
| `s2_extraction/validator.py` | script | called programmatically by extractor |
| `s3_canonicalisation/clusterer.py` | script | one-shot clustering; output locked before any modelling |
| `s4_encoding/text_encoder.py` | script | long-running; caches to disk |
| `s4_encoding/graph_stats_encoder.py` | script | called by classification notebooks |
| `s4_encoding/graph_gnn_encoder.py` | script | self-supervised GIN training + frozen inference (train once, encode with --encode) |
| `s4_encoding/build_dataset.py` | script | package frozen embeddings as .npz per split/target |
| `s5_classification/run.py` | script | config-driven experiment runner (any arch × any target) |
| `s6_notebooks/01_extraction_review.py` | Marimo | interactive graph inspection, model comparison scoring |
| `s6_notebooks/02_graph_exploration.py` | Marimo | cohort topology visualisation, hypothesis testing |
| `s6_notebooks/03_classification_results.py` | Marimo | confusion matrices, feature importance, route comparison |
| `s6_notebooks/04_structural_analysis.py` | Marimo | H1-H4 confirmatory analysis, AI adoption exploratory |
| `s6_notebooks/05_fusion_analysis.py` | Marimo | fusion experiment: complementarity matrices, arch comparison |

---

## 2. project structure

```
cdt-graph-modality/
│
├── docs/
│   ├── CHARTER.md → ../.claude/context/   # symlinks to context docs
│   ├── ENGINEERING.md → ../.claude/context/
│   ├── PLAN.md
│   └── adr/
│
├── s1_data/
│   ├── raw/                              # downloaded CSVs (gitignored)
│   ├── tagged/                           # speaker-tagged .jsonl (gitignored)
│   └── graphs/
│       ├── free_text/                    # extracted graphs (gitignored)
│       └── canonical/                    # canonicalised graphs (gitignored)
│
├── cache/                                # embeddings + models (gitignored)
│   ├── text_embeddings_human_only.npy
│   ├── graph_stats.npy
│   ├── gin_embeddings_canonical.npy / _free_text.npy
│   ├── gin_encoder_canonical.pt / _free_text.pt
│   └── modality_dataset/                 # .npz per split/target
│
├── s2_extraction/
│   ├── prompts/v1.txt–v3.txt            # versioned, never deleted
│   ├── tagger.py, extractor.py, validator.py
│   └── model_comparison/
│
├── s3_canonicalisation/
│   ├── clusterer.py, apply_canonical.py
│   └── canonical_map.json               # locked, immutable
│
├── s4_encoding/                              # flat — no subfolders
│   ├── text_encoder.py                   # SBERT 768-dim
│   ├── graph_stats_encoder.py            # 30-dim deterministic
│   ├── graph_gnn_encoder.py              # GIN autoencoder + frozen inference
│   ├── graph_dataset.py                  # PyG Dataset wrapper
│   ├── build_dataset.py                  # package as .npz
│   └── _archived/                        # Phase 3 task-supervised GIN
│
├── s5_classification/                        # flat — no subfolders
│   ├── split.py                          # train/val/test split
│   ├── train_config.py                   # ExperimentConfig + sweep builders
│   ├── train_run.py                      # config-driven runner (torch + sklearn)
│   ├── train_loop.py                     # PyTorch training loop
│   ├── classifiers.py                    # PyTorch classifier zoo + factory
│   ├── mlp_single.py                     # Single-modality MLP baseline
│   ├── mlp_stacked.py                    # Stacked (concat) fusion MLP
│   ├── mlp_gated.py                      # Gated (attention) fusion MLP
│   ├── mlp_late.py                       # Late (ensemble) fusion MLP
│   ├── sklearn_classifier.py             # sklearn wrapper (4 classifiers)
│   ├── baseline.py                       # thin convenience wrapper
│   ├── analysis_feature_importance.py    # permutation importance
│   ├── analysis_stats.py                 # stats-only per-class report
│   └── _archived/                        # Phase 3 route3
│
├── s6_notebooks/
│   ├── 01_extraction_review.py           # Marimo
│   ├── 02_graph_exploration.py           # Marimo
│   ├── 03_classification_results.py      # Marimo
│   ├── 04_structural_analysis.py         # Marimo
│   └── 05_fusion_analysis.py             # Marimo
│
├── results/
│   └── fusion/{target}/{arch}_{mods}/     # experiment outputs (gitignored)
│
├── tests/
├── .env, .gitignore, pyproject.toml
```

---

## 3. environment setup

### 3.1 uv initialisation

```bash
# install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# initialise project
uv init cdt-graph-modality
cd cdt-graph-modality

# add core dependencies
uv add polars sentence-transformers networkx scikit-learn \
       tqdm jsonschema python-dotenv anthropic \
       huggingface-hub marimo matplotlib pyvis

# CPU-only torch — specify index via pyproject.toml (see 3.2), then:
uv add torch torchvision
uv add torch-geometric
```

### 3.2 pyproject.toml

```toml
[project]
name = "cdt-graph-modality"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "polars>=1.0",
    "sentence-transformers>=3.0",
    "networkx>=3.3",
    "scikit-learn>=1.5",
    "tqdm>=4.66",
    "jsonschema>=4.22",
    "python-dotenv>=1.0",
    "anthropic>=0.30",
    "huggingface-hub>=0.24",
    "marimo>=0.7",
    "matplotlib>=3.9",
    "pyvis>=0.3",
    "torch>=2.3",
    "torch-geometric>=2.5",
]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

> The `[tool.uv.sources]` block tells uv to pull torch and torchvision exclusively from the CPU wheel index. This gives you a ~250MB install rather than ~2GB. torch_geometric resolves against whichever torch version uv installs.

### 3.3 API keys

```bash
# .env (never committed)
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=...       # for model comparison only
AGNES_API_KEY=...          # for model comparison only; verify context window limit first
```

Load in all scripts:
```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.environ["ANTHROPIC_API_KEY"]
```

---

## 4–7. extraction, validation, canonicalisation

These stages are Phase 1–2 of the pipeline. See the source files for implementation:

| Stage | Key files | Design doc |
|---|---|---|
| Speaker tagging | `s2_extraction/tagger.py` | Regex-based, `^Assistant:`, `^AI:`, `^User:` prefixes |
| Extraction | `s2_extraction/extractor.py` | Batch 50, 3 retries, cache-first, validator after each |
| Prompt | `s2_extraction/prompts/v1.txt`–`v3.txt` | Versioned, never deleted; v3 = two-shot (active) |
| Validation | `s2_extraction/validator.py` | 6 constraints (C1–C6), failures logged not raised |
| Model comparison | `s2_extraction/model_comparison/` | 10 fixed transcripts, 5-criterion rubric, R3 tiebreaker |
| Canonicalisation | `s3_canonicalisation/clusterer.py`, `apply_canonical.py` | AgglomerativeClustering, cosine 0.35, locked post-review |

The graph JSON schema (`.claude/context/graph-schema.md`) is the data contract. The ontology (entity types, relation types, constraints) is defined in `docs/CHARTER.md` §4. The `schema-guardian` agent enforces consistency across prompt, validator, and downstream consumers.

---

## 8. encoding pipeline

Three frozen modality encoders. All are cache-first — never recompute if cache exists.

### 8.1 Text (`s4_encoding/text_encoder.py`)

| Property | Value |
|---|---|
| Model | `all-mpnet-base-v2` (frozen pretrained) |
| Output | 768-dim per transcript |
| Input | Human-only turns (removes interviewer confound) |
| Cache | `cache/text_embeddings_human_only.npy` |

### 8.2 Graph stats (`s4_encoding/graph_stats_encoder.py`)

| Property | Value |
|---|---|
| Method | Deterministic networkx features |
| Output | 30-dim per graph |
| Features | Structural (7), type distribution (6), construct quality (3), stance valence (8), centrality (3), cognitive style (2) |
| Cache | `cache/graph_stats.npy` |
| Label dependence | **None** — reads `type`, `valence`, `bipolarity_complete` only |

### 8.3 GIN autoencoder (`s4_encoding/graph_gnn_encoder.py`)

Two independently trained encoders, one per label source:

| Property | Canonical | Free-text |
|---|---|---|
| Graph data | `s1_data/graphs/canonical/` | `s1_data/graphs/free_text/` |
| Encoder weights | `cache/gin_encoder_canonical.pt` | `cache/gin_encoder_free_text.pt` |
| Embedding cache | `cache/gin_embeddings_canonical.npy` | `cache/gin_embeddings_free_text.npy` |
| Training curves | `cache/gin_autoencoder_curves_canonical.png` | `cache/gin_autoencoder_curves_free_text.png` |

**Architecture:** 2-layer GINConv (388→256→128) + global mean pool. Decoder: Linear(128, 4) for node type reconstruction. Loss: cross-entropy. Trained on ALL 1,250 graphs (no split). Adam, lr=1e-3.

**Node features (both):** 4 type one-hot + 384 MiniLM label embedding = 388 dims. The label embedding is the only difference between canonical and free-text graphs.

#### 8.3.2a Label source: free-text vs canonical

See §8.3.2a above for the full trade-off analysis. The canonical-vs-free-text choice is parameterized as `ExperimentConfig.graph_label_source`.

#### 8.3.2b Dual-encoder design

Both autoencoders were trained (2026-06-09), both achieving 100% node type accuracy. `graph_label_source` now controls both the graph data AND which encoder is used — no out-of-distribution inference. Each label source gets a matched encoder.

### 8.4 Dataset packaging (`s4_encoding/build_dataset.py`)

Packages all three frozen embeddings into .npz per split/target:
- `cache/modality_dataset/{target}_{split}.npz` — 6 files (2 targets × 3 splits)
- Keys: `text_emb` (768), `stats_emb` (30), `graph_emb` (128), `labels`, `transcript_ids`
- The canonical .npz is the default. Free-text GNN embeddings are swapped on-the-fly by `train_run.py`.

---

## 9. classification

**Architecture principle:** All classifiers consume frozen modality embeddings. Encoders are never fine-tuned. Only classifier weights learn per-task.

### 9.1 Backends

| Backend | Files | Training | Classifiers |
|---|---|---|---|
| `torch` | `mlp_*.py`, `classifiers.py`, `train_loop.py` | Epoch-based, early stopping, Adam | single, stacked, gated, late |
| `sklearn` | `sklearn_classifier.py` | One-shot `.fit()` | logistic, random_forest, gradient_boost, svm |

Both backends share `train_config.py` (config) and `train_run.py` (orchestration). Adding a new sklearn classifier is one import + one dict entry.

### 9.2 Experiment runner

```bash
uv run python s5_classification/train_run.py                    # torch sweep (42)
uv run python s5_classification/train_run.py --sweep sklearn    # sklearn sweep (48)
uv run python s5_classification/train_run.py --sweep all        # combined (90)
uv run python s5_classification/train_run.py --target ai_adoption --backend sklearn
```

Each experiment produces: `model.pt`/`.joblib`, `curves.png`, `test_preds.npy`, `metrics.json`.

### 9.3 Experiment matrix

| Dimension | Values |
|---|---|
| Backend | torch, sklearn |
| Target | AI adoption (binary, n=1,224), Cohort (3-class, n=1,250) |
| Modalities | text, stats, graph, text+stats, text+graph, text+stats+graph |
| Architecture (torch) | single, stacked, gated, late |
| Architecture (sklearn) | logistic, random_forest, gradient_boost, svm |
| Graph label source | canonical (default), free_text |

### 9.4 Analysis tools

| File | Purpose |
|---|---|
| `s5_classification/analysis_feature_importance.py` | Permutation importance — which 30 graph features matter |
| `s5_classification/analysis_stats.py` | Stats-only per-class report |
| `s5_classification/baseline.py` | Thin wrapper — text-only LR convenience alias |

### 9.5 Evaluation

- **Primary metric:** macro-averaged F1. Accuracy reported but not used for selection.
- **Split:** 70/15/15 stratified, seed=42, cached at `cache/split_ids.json`
- **Test set discipline:** held out until final evaluation. No hyperparameter tuning on test.
- **Per-example predictions** saved for complementarity analysis (`notebooks/05_fusion_analysis.py`).

---

## 10. marimo notebooks

All in `s6_notebooks/`:

| Notebook | Purpose |
|---|---|
| `01_extraction_review.py` | Graph inspection, pyvis rendering, rubric scoring |
| `02_graph_exploration.py` | Cohort topology, H1–H4 tests, PCA/UMAP |
| `03_classification_results.py` | F1 comparison, confusion matrices, feature importance |
| `04_structural_analysis.py` | H1–H4 confirmatory analysis, AI adoption exploratory |
| `05_fusion_analysis.py` | Complementarity matrices, architecture comparison |
