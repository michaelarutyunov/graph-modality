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
| `data/download.py` | script | one-shot, stateful download with existence check |
| `extraction/tagger.py` | script | pure text processing, no interactivity needed |
| `extraction/extractor.py` | script | long-running API batch with retry logic; kernel death is unacceptable |
| `extraction/validator.py` | script | called programmatically by extractor |
| `canonicalisation/clusterer.py` | script | one-shot clustering; output locked before any modelling |
| `encoding/text_encoder.py` | script | long-running; caches to disk |
| `encoding/graph_stats.py` | script | called by classification notebooks |
| `encoding/gnn_encoder.py` | script | training loop; called by classification notebooks |
| `notebooks/01_extraction_review.py` | Marimo | interactive graph inspection, model comparison scoring |
| `notebooks/02_graph_exploration.py` | Marimo | cohort topology visualisation, hypothesis testing |
| `notebooks/03_classification_results.py` | Marimo | confusion matrices, feature importance, route comparison |

---

## 2. project structure

```
cdt-graph-modality/
│
├── data/
│   ├── raw/                              # downloaded CSVs (gitignored)
│   │   └── interview_transcripts/
│   │       ├── workforce_transcripts.csv
│   │       ├── creatives_transcripts.csv
│   │       └── scientists_transcripts.csv
│   ├── tagged/                           # speaker-tagged transcripts (.jsonl, gitignored)
│   └── graphs/
│       ├── free_text/                    # one JSON per transcript (gitignored)
│       └── canonical/                    # canonicalised graphs (gitignored)
│
├── cache/                                # encoded embeddings (gitignored)
│   ├── text_embeddings.npy
│   └── graph_stats.npy
│
├── extraction/
│   ├── prompts/
│   │   └── v1.txt                        # extraction prompt (versioned in git)
│   ├── tagger.py
│   ├── extractor.py
│   ├── validator.py
│   └── model_comparison/
│       ├── sample_ids.txt                # fixed 10-transcript comparison set
│       ├── run_comparison.py
│       └── rubric_scorer.py
│
├── canonicalisation/
│   ├── clusterer.py
│   ├── canonical_map.json                # locked after day 3; never modified post-lock
│   └── apply_canonical.py
│
├── encoding/
│   ├── text_encoder.py
│   ├── graph_stats.py
│   └── gnn/
│       ├── dataset.py                    # torch_geometric Dataset wrapper
│       ├── model.py                      # GIN architecture
│       └── train.py                      # training loop
│
├── classification/
│   ├── baseline.py
│   ├── route2.py
│   └── route3.py
│
├── notebooks/
│   ├── 01_extraction_review.py           # Marimo
│   ├── 02_graph_exploration.py           # Marimo
│   └── 03_classification_results.py      # Marimo
│
├── tests/
│   ├── test_tagger.py
│   ├── test_validator.py
│   └── test_graph_stats.py
│
├── .env                                  # API keys (gitignored)
├── .gitignore
├── pyproject.toml
├── CHARTER.md
└── ENGINEERING.md
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

## 4. data acquisition

### 4.1 download with existence check

`data/download.py` — run once before anything else.

```python
from pathlib import Path
from huggingface_hub import snapshot_download

RAW_DIR = Path("data/raw")
EXPECTED = [
    "interview_transcripts/workforce_transcripts.csv",
    "interview_transcripts/creatives_transcripts.csv",
    "interview_transcripts/scientists_transcripts.csv",
]

def ensure_dataset() -> None:
    missing = [f for f in EXPECTED if not (RAW_DIR / f).exists()]
    if not missing:
        print("dataset already present — skipping download")
        return
    print(f"downloading {len(missing)} missing file(s)...")
    snapshot_download(
        repo_id="Anthropic/AnthropicInterviewer",
        repo_type="dataset",
        local_dir=RAW_DIR,
        ignore_patterns=["*.md", "*.json"],   # skip metadata files
    )
    print("download complete")

if __name__ == "__main__":
    ensure_dataset()
```

### 4.2 loading with polars

```python
import polars as pl
from pathlib import Path

RAW_DIR = Path("data/raw/interview_transcripts")

def load_split(split: str) -> pl.DataFrame:
    """Load one split and attach the split label."""
    path = RAW_DIR / f"{split}_transcripts.csv"
    return pl.read_csv(path).with_columns(pl.lit(split).alias("split"))

def load_all() -> pl.DataFrame:
    splits = ["workforce", "creatives", "scientists"]
    return pl.concat([load_split(s) for s in splits])
```

### 4.3 class label encoding

```python
SPLIT_TO_LABEL = {"workforce": 0, "creatives": 1, "scientists": 2}

df = load_all().with_columns(
    pl.col("split").replace(SPLIT_TO_LABEL).alias("label")
)
```

---

## 5. graph extraction pipeline

### 5.1 speaker tagging

`extraction/tagger.py`

Transcripts use three speaker prefixes (confirmed from dataset inspection, 2026-06-07):
- ``Assistant:`` — first AI turn (opening)
- ``AI:`` — subsequent AI turns
- ``User:`` — all human turns

**NOTE:** The earlier documentation stated ``Human:`` but no transcript in the dataset uses that prefix.

```python
import re
from dataclasses import dataclass

_SPEAKER_PREFIX = re.compile(r"^(Assistant|AI|User):\s*", re.MULTILINE)

@dataclass
class Turn:
    speaker: str          # "AI" | "Human"
    text: str
    turn_index: int

def parse_transcript(raw_text: str) -> list[Turn]:
    parts = _SPEAKER_PREFIX.split(raw_text.strip())
    # parts alternates: [pre-match, speaker, text, speaker, text, ...]
    turns = []
    i = 1
    while i < len(parts) - 1:
        speaker_raw = parts[i].strip()
        text = parts[i + 1].strip()
        speaker = "Human" if speaker_raw == "User" else "AI"
        turns.append(Turn(speaker=speaker, text=text, turn_index=len(turns)))
        i += 2
    return turns

def format_for_extraction(turns: list[Turn]) -> str:
    """Format tagged transcript for insertion into extraction prompt."""
    lines = []
    for t in turns:
        tag = "[AI]" if t.speaker == "AI" else "[Human]"
        lines.append(f"{tag}: {t.text}")
    return "\n\n".join(lines)
```

### 5.2 extraction prompt

`extraction/prompts/v1.txt`

```
You are an expert cognitive mapping assistant. Your task is to extract a structured
concept graph from an interview transcript according to a strict ontology.
Return a valid JSON object only — no preamble, no explanation, no markdown fences.

ONTOLOGY
========

Entity types:
- Construct: a bipolar cognitive dimension used to evaluate AI's role. Must have
  both a positive pole and a negative pole. If only one pole is inferrable, set
  bipolarity_complete to false.
- Value: a terminal motivational state that constructs serve. High-abstraction.
- Stance: a valenced attitude position. Required attribute: valence (one of:
  positive, negative, mixed, ambivalent).
- CognitiveStyleMarker: stable processing tendency (HOW the person reasons,
  not WHAT they care about). Maximum 2 per transcript.

Relation types:
- SERVES: Construct → Value
- EXPRESSED_VIA: Stance → Construct
- MODULATED_BY: Construct → CognitiveStyleMarker
- CONFLICTS_WITH: Construct ↔ Construct (undirected, use sparingly)

CONSTRAINTS
===========
1. Extract nodes from [Human] turns only. Use [AI] turns as context only.
2. Every node must include a grounding_span: a short verbatim phrase from a
   [Human] turn that supports the node's existence.
3. No direct Stance → Value edges. Stances must connect to Values via Constructs.
4. Maximum 2 CognitiveStyleMarker nodes per transcript.
5. Discard any node you cannot ground in a [Human] turn.

OUTPUT SCHEMA
=============
{
  "transcript_id": "string",
  "nodes": [
    {
      "id": "n1",
      "type": "Construct | Value | Stance | CognitiveStyleMarker",
      "label": "string",
      "label_negative": "string or null (Construct only)",
      "bipolarity_complete": "bool (Construct only)",
      "valence": "positive | negative | mixed | ambivalent | null (Stance only)",
      "grounding_span": "string"
    }
  ],
  "edges": [
    {
      "source": "n1",
      "target": "n2",
      "relation": "SERVES | EXPRESSED_VIA | MODULATED_BY | CONFLICTS_WITH"
    }
  ]
}

TRANSCRIPT
==========
{transcript}
```

**Prompt iteration protocol:** run on 5 transcripts before the model comparison. Score manually on rubric. Iterate once. Lock before comparison experiment.

### 5.3 model comparison experiment

`extraction/model_comparison/run_comparison.py`

**Sample design:** 10 transcripts, fixed in `sample_ids.txt`:
- 4 workforce (2 short <8k chars, 2 long >15k chars)
- 3 creatives
- 3 scientists

Same 10 transcripts through all three models identically.

**Rubric** (score 1–3 per criterion; see CHARTER.md §4 for full definitions):

| id | criterion |
|---|---|
| R1 | graph size consistency |
| R2 | ontology adherence |
| R3 | bipolarity capture |
| R4 | hallucination rate |
| R5 | relation typing accuracy |

**Decision rule:** highest aggregate score wins. On ties, R3 (bipolarity capture) is the tiebreaker — it is the theoretically most critical property.

**Agnes API note:** verify context window limit before running. Transcripts reach 27k characters. Any model with a context window under 32k will silently truncate — check documentation before including in comparison.

```python
# run_comparison.py skeleton
def run_for_model(model_name: str, sample_ids: list[str], transcripts: dict) -> dict:
    results = {}
    for tid in sample_ids:
        prompt = build_prompt(transcripts[tid])
        raw = call_api(model_name, prompt)
        graph = parse_and_validate(raw, tid)
        results[tid] = graph
    return results
```

### 5.4 scale extraction

`extraction/extractor.py`

- Process in batches of 50
- Check cache before every API call: if `data/graphs/free_text/{tid}.json` exists, skip
- Retry on API failure: 3 attempts with exponential backoff (2s, 8s, 32s)
- Log failures to `extraction/failed.txt` for manual re-run
- Run validator after each extraction; invalid graphs logged separately

```python
from pathlib import Path
import json, time

GRAPH_DIR = Path("data/graphs/free_text")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

def extract_with_cache(tid: str, transcript: str, client) -> dict | None:
    cache_path = GRAPH_DIR / f"{tid}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    
    for attempt in range(3):
        try:
            result = call_extraction_api(client, transcript)
            graph = json.loads(result)
            validate_graph(graph)                  # raises on violation
            cache_path.write_text(json.dumps(graph, indent=2))
            return graph
        except Exception as e:
            if attempt == 2:
                log_failure(tid, str(e))
                return None
            time.sleep(2 ** (attempt + 1))
```

**Scale decision:** start with 300 transcripts (100 per split, stratified sample). Expand to full 1,250 only if route 2 shows signal and you want to confirm with more power.

---

## 6. graph validation

`extraction/validator.py`

Checks run after every extraction. Failures are logged, not raised — extraction continues.

```python
def validate_graph(graph: dict) -> list[str]:
    """Returns list of violation strings. Empty = valid."""
    violations = []
    
    node_ids = {n["id"] for n in graph["nodes"]}
    
    # bipolarity check
    for n in graph["nodes"]:
        if n["type"] == "Construct":
            if not n.get("label_negative"):
                n["bipolarity_complete"] = False
                violations.append(f"{n['id']}: missing negative pole")
    
    # cognitive style ceiling
    csm_count = sum(1 for n in graph["nodes"] 
                    if n["type"] == "CognitiveStyleMarker")
    if csm_count > 2:
        violations.append(f"CognitiveStyleMarker count {csm_count} exceeds ceiling of 2")
    
    # edge validity
    for e in graph["edges"]:
        if e["source"] not in node_ids or e["target"] not in node_ids:
            violations.append(f"edge references unknown node: {e}")
    
    # no direct Stance → Value
    node_type = {n["id"]: n["type"] for n in graph["nodes"]}
    for e in graph["edges"]:
        if (node_type.get(e["source"]) == "Stance" and 
            node_type.get(e["target"]) == "Value"):
            violations.append(f"direct Stance→Value edge disallowed: {e}")
    
    return violations
```

---

## 7. canonicalisation

`canonicalisation/clusterer.py`

One-time process run after scale extraction. Output locked before any modelling.

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np, json, polars as pl
from pathlib import Path

def build_canonical_map(graphs: list[dict], entity_type: str,
                         distance_threshold: float = 0.3) -> dict[str, str]:
    """
    For one entity type, cluster all label strings by embedding similarity.
    Canonical label = the label closest to the cluster centroid.
    Returns {free_text_label: canonical_label}.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    labels = list({
        n["label"] for g in graphs 
        for n in g["nodes"] if n["type"] == entity_type
    })
    
    embeddings = model.encode(labels, normalize_embeddings=True)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    cluster_ids = clustering.fit_predict(embeddings)
    
    canonical_map = {}
    for cluster_id in set(cluster_ids):
        mask = cluster_ids == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_labels = [l for l, m in zip(labels, mask) if m]
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        canonical = cluster_labels[distances.argmin()]
        for label in cluster_labels:
            canonical_map[label] = canonical
    
    return canonical_map
```

**Manual review step:** after clustering, inspect cluster assignments for each entity type. Expected vocabulary sizes: 15–25 Values, 30–50 Constructs, 20–35 Stances, 8–12 Cognitive Style Markers. Merge or split clusters as needed. Document decisions in a review log. Lock `canonical_map.json` before any downstream modelling — it does not change between experiments.

---

## 8. encoding pipeline

### 8.1 text encoder (shared across all routes)

`encoding/text_encoder.py`

```python
from sentence_transformers import SentenceTransformer
import numpy as np, polars as pl
from pathlib import Path

CACHE_PATH = Path("cache/text_embeddings.npy")
ID_CACHE = Path("cache/text_embedding_ids.json")

def encode_transcripts(df: pl.DataFrame, 
                        model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    if CACHE_PATH.exists():
        print("loading cached text embeddings")
        return np.load(CACHE_PATH)
    
    model = SentenceTransformer(model_name)
    texts = df["text"].to_list()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.save(CACHE_PATH, embeddings)
    import json
    ID_CACHE.write_text(json.dumps(df["transcript_id"].to_list()))
    print(f"cached {len(embeddings)} embeddings → {CACHE_PATH}")
    return embeddings
```

Output: 768-dimensional embedding per transcript.  
Runtime estimate: ~25 minutes on CPU for 1,250 transcripts; ~6 minutes for 300.  
**Always load from cache after first run.**

### 8.2 route 2: graph statistics (numpy + networkx)

`encoding/graph_stats.py`

Uses canonicalised graphs. Produces a ~35-dimensional feature vector per transcript.

```python
import networkx as nx
import numpy as np
from pathlib import Path
import json

ENTITY_TYPES = ["Construct", "Value", "Stance", "CognitiveStyleMarker"]
VALENCES = ["positive", "negative", "mixed", "ambivalent"]

def graph_to_features(graph_path: Path) -> np.ndarray:
    g_data = json.loads(graph_path.read_text())
    
    G = nx.DiGraph()
    for n in g_data["nodes"]:
        G.add_node(n["id"], **n)
    for e in g_data["edges"]:
        G.add_edge(e["source"], e["target"], relation=e["relation"])
    
    nodes = g_data["nodes"]
    n_total = max(len(nodes), 1)
    
    # --- structural ---
    n_edges = G.number_of_edges()
    density = nx.density(G)
    n_components = nx.number_weakly_connected_components(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0.0
    try:
        diameter = nx.diameter(G.to_undirected())
    except (nx.NetworkXError, nx.exception.NetworkXException):
        diameter = -1.0
    
    # --- node type distribution ---
    type_counts = {t: 0 for t in ENTITY_TYPES}
    for n in nodes:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
    
    n_construct = type_counts["Construct"]
    n_value = type_counts["Value"]
    n_stance = type_counts["Stance"]
    n_csm = type_counts["CognitiveStyleMarker"]
    
    construct_value_ratio = n_construct / max(n_value, 1)
    stance_construct_ratio = n_stance / max(n_construct, 1)
    
    # --- construct quality ---
    constructs = [n for n in nodes if n["type"] == "Construct"]
    bipolarity_score = (
        np.mean([1.0 if n.get("bipolarity_complete") else 0.5 
                 for n in constructs])
        if constructs else 0.0
    )
    construct_degrees = [G.degree(n["id"]) for n in constructs]
    mean_construct_degree = np.mean(construct_degrees) if construct_degrees else 0.0
    max_construct_degree = max(construct_degrees) if construct_degrees else 0.0
    
    # --- stance valence ---
    stances = [n for n in nodes if n["type"] == "Stance"]
    valence_counts = {v: 0 for v in VALENCES}
    for s in stances:
        v = s.get("valence", "ambivalent")
        valence_counts[v] = valence_counts.get(v, 0) + 1
    
    dominant_valence = max(valence_counts, key=valence_counts.get) if stances else "absent"
    valence_onehot = [1.0 if dominant_valence == v else 0.0 for v in VALENCES]
    valence_onehot.append(1.0 if dominant_valence == "absent" else 0.0)
    
    # --- centrality ---
    try:
        betweenness = nx.betweenness_centrality(G)
        bc_values = list(betweenness.values())
        max_bc = max(bc_values) if bc_values else 0.0
        mean_bc = np.mean(bc_values) if bc_values else 0.0
        value_nodes = [n["id"] for n in nodes if n["type"] == "Value"]
        max_value_bc = max(betweenness.get(v, 0.0) for v in value_nodes) if value_nodes else 0.0
    except Exception:
        max_bc = mean_bc = max_value_bc = 0.0
    
    # --- cognitive style ---
    csm_present = float(n_csm > 0)
    csm_count_clipped = min(n_csm, 2) / 2.0  # normalised to [0, 1]
    
    features = np.array([
        # structural (7)
        n_total / 15.0,           # normalised by expected max
        n_edges / 20.0,
        density,
        n_components / n_total,
        avg_degree / 5.0,
        max_degree / 10.0,
        (diameter + 1) / 10.0,   # +1 so -1 (disconnected) maps to 0
        # node type distribution (6)
        n_construct / n_total,
        n_value / n_total,
        n_stance / n_total,
        n_csm / n_total,
        construct_value_ratio / 5.0,
        stance_construct_ratio / 3.0,
        # construct quality (3)
        bipolarity_score,
        mean_construct_degree / 5.0,
        max_construct_degree / 10.0,
        # stance valence (8 = 4 counts + 5 onehot)
        valence_counts["positive"] / max(n_stance, 1),
        valence_counts["negative"] / max(n_stance, 1),
        valence_counts["mixed"] / max(n_stance, 1),
        valence_counts["ambivalent"] / max(n_stance, 1),
        *valence_onehot,          # 5 values
        # centrality (3)
        max_bc,
        mean_bc,
        max_value_bc,
        # cognitive style (2)
        csm_present,
        csm_count_clipped,
    ], dtype=np.float32)
    
    return features  # shape: (36,)
```

**Total feature vector dimension:** 36  
**Interpretation layer:** use sklearn permutation importance on the trained logistic regression model to identify which features drive classification. This is where the interpretable story lives.

### 8.3 route 3: GIN graph encoder

#### 8.3.1 node feature construction

`encoding/gnn/dataset.py`

Each node gets a 388-dimensional feature vector:
- entity type one-hot: 4 dimensions
- label embedding: 384 dimensions (`all-MiniLM-L6-v2` on the free-text label)

```python
import torch
from torch_geometric.data import Data, Dataset
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

ENTITY_TYPES = ["Construct", "Value", "Stance", "CognitiveStyleMarker"]
RELATIONS = ["SERVES", "EXPRESSED_VIA", "MODULATED_BY", "CONFLICTS_WITH"]
TYPE_TO_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}
REL_TO_IDX = {r: i for i, r in enumerate(RELATIONS)}

class GraphDataset(Dataset):
    def __init__(self, graph_paths: list[Path], labels: list[int]):
        super().__init__()
        self.graph_paths = graph_paths
        self.labels = labels
        self.label_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def len(self):
        return len(self.graph_paths)
    
    def get(self, idx: int) -> Data:
        g_data = json.loads(self.graph_paths[idx].read_text())
        nodes = g_data["nodes"]
        edges = g_data["edges"]
        
        node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        
        # node features: type onehot (4) + label embedding (384) = 388
        labels_text = [n["label"] for n in nodes]
        label_embeddings = self.label_encoder.encode(
            labels_text, normalize_embeddings=True, show_progress_bar=False
        )
        
        type_onehots = torch.zeros(len(nodes), 4)
        for i, n in enumerate(nodes):
            type_onehots[i, TYPE_TO_IDX.get(n["type"], 0)] = 1.0
        
        x = torch.cat([
            type_onehots,
            torch.tensor(label_embeddings, dtype=torch.float32)
        ], dim=1)  # shape: (n_nodes, 388)
        
        # edge indices and features
        if edges:
            edge_index = torch.tensor(
                [[node_id_to_idx[e["source"]], node_id_to_idx[e["target"]]]
                 for e in edges],
                dtype=torch.long
            ).t().contiguous()  # shape: (2, n_edges)
            
            edge_attr = torch.zeros(len(edges), 4)
            for i, e in enumerate(edges):
                rel_idx = REL_TO_IDX.get(e["relation"], 0)
                edge_attr[i, rel_idx] = 1.0
        else:
            # isolated graph (no edges) — valid edge_index for torch_geometric
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4))
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(self.labels[idx], dtype=torch.long),
            transcript_id=g_data["transcript_id"],
        )
```

#### 8.3.2 GIN model

`encoding/gnn/model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int = 388, hidden: int = 256, 
                 out_channels: int = 128, n_classes: int = 3):
        super().__init__()
        
        # GIN layer 1
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.conv1 = GINConv(mlp1)
        self.bn1 = nn.BatchNorm1d(hidden)
        
        # GIN layer 2
        mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channels),
        )
        self.conv2 = GINConv(mlp2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # classifier head (receives fused text + graph embedding)
        self.classifier = nn.Sequential(
            nn.Linear(768 + out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
    
    def encode_graph(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return global_mean_pool(x, batch)  # shape: (batch_size, 128)
    
    def forward(self, data, text_embeddings: torch.Tensor):
        graph_emb = self.encode_graph(data.x, data.edge_index, data.batch)
        fused = torch.cat([text_embeddings, graph_emb], dim=1)  # (B, 896)
        return self.classifier(fused)
```

#### 8.3.3 training loop

`encoding/gnn/train.py`

```python
import torch
from torch_geometric.loader import DataLoader
import numpy as np

def compute_class_weights(labels: list[int], n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    return torch.tensor(weights / weights.sum() * n_classes, dtype=torch.float32)

def train(model, train_loader, val_loader, 
          text_emb_train, text_emb_val,
          class_weights, epochs: int = 50):
    
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=5, factor=0.5
    )
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            text_batch = text_emb_train[batch_idx * train_loader.batch_size :
                                        (batch_idx + 1) * train_loader.batch_size]
            optimiser.zero_grad()
            logits = model(batch, torch.tensor(text_batch, dtype=torch.float32))
            loss = criterion(logits, batch.y)
            loss.backward()
            optimiser.step()
        
        val_f1 = evaluate(model, val_loader, text_emb_val)
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "cache/best_gin.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"early stopping at epoch {epoch}")
                break
        
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d} | val macro-F1: {val_f1:.4f}")
```

---

## 9. classification

### 9.1 train/test split

Fixed before any modelling. Use stratified split to preserve class ratios.

```python
from sklearn.model_selection import train_test_split

# 70 / 15 / 15 stratified split
ids = df["transcript_id"].to_list()
labels = df["label"].to_list()

ids_train, ids_temp, y_train, y_temp = train_test_split(
    ids, labels, test_size=0.30, stratify=labels, random_state=42
)
ids_val, ids_test, y_val, y_test = train_test_split(
    ids_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
```

**The test set is held out until final evaluation. No hyperparameter decisions are made on test set performance.**

### 9.2 baseline and route 2

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# baseline: text embeddings only
X_train_base = text_emb[train_idx]
X_val_base = text_emb[val_idx]

lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
lr.fit(X_train_base, y_train)
preds = lr.predict(X_val_base)
print(f"baseline macro-F1: {f1_score(y_val, preds, average='macro'):.4f}")

# route 2: text + graph stats
X_train_r2 = np.hstack([text_emb[train_idx], graph_stats[train_idx]])
lr_r2 = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
lr_r2.fit(X_train_r2, y_train)
```

### 9.3 evaluation

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_full(model, X_test, y_test, route_name: str):
    preds = model.predict(X_test)
    print(f"\n=== {route_name} ===")
    print(classification_report(y_test, preds,
                                 target_names=["workforce", "creatives", "scientists"],
                                 digits=4))
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"macro F1: {macro_f1:.4f}")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds,
        display_labels=["workforce", "creatives", "scientists"],
        ax=ax, colorbar=False
    )
    ax.set_title(f"{route_name} — confusion matrix")
    plt.tight_layout()
    plt.savefig(f"cache/{route_name.lower().replace(' ', '_')}_cm.png", dpi=150)
```

---

## 10. marimo notebooks

Marimo notebooks are `.py` files run with `marimo edit notebooks/01_extraction_review.py`.

### 10.1 01_extraction_review.py

**Purpose:** interactive inspection of extracted graphs; model comparison rubric scoring.

Key cells:
- transcript selector (dropdown by split and ID)
- side-by-side graph rendering using `pyvis` (NetworkX → interactive HTML via `mo.Html`)
- rubric scoring UI per model per transcript
- aggregate score table export

```python
import marimo as mo
import pyvis.network as pvnet

def render_graph(graph_data: dict) -> mo.Html:
    net = pvnet.Network(height="400px", width="100%", directed=True)
    colour_map = {
        "Construct": "#7F77DD",
        "Value": "#1D9E75",
        "Stance": "#BA7517",
        "CognitiveStyleMarker": "#D85A30",
    }
    for n in graph_data["nodes"]:
        label = n["label"]
        if n["type"] == "Construct" and n.get("label_negative"):
            label = f"{n['label']} ↔ {n['label_negative']}"
        net.add_node(n["id"], label=label,
                     color=colour_map.get(n["type"], "#888"))
    for e in graph_data["edges"]:
        net.add_edge(e["source"], e["target"], label=e["relation"])
    return mo.Html(net.generate_html())
```

### 10.2 02_graph_exploration.py

**Purpose:** cohort topology visualisation; interpretability hypothesis testing (H1–H4).

Key cells:
- box plots of graph statistics by cohort
- H1–H4 statistical tests (Mann-Whitney U, Kruskal-Wallis)
- PCA/UMAP of graph stat feature vectors coloured by cohort
- node type distribution stacked bar charts

### 10.3 03_classification_results.py

**Purpose:** final results presentation; route comparison; feature importance.

Key cells:
- macro F1 comparison table (baseline / route 2 / route 3)
- confusion matrix grid
- route 2 permutation feature importance bar chart
- per-class F1 comparison across routes

---

## 11. open questions and risks

| question | status | resolution |
|---|---|---|
| Agnes context window limit | unverified | check before model comparison; exclude if < 32k |
| torch_geometric installation | unverified | test `uv add torch torch-geometric` in fresh venv on day 1 |
| snapshot_download behind auth | possible | if 403, use `huggingface-hub` login: `huggingface-cli login` |
| extraction quality at scale | unknown | manually validate 20 random graphs after scale extraction |
| canonical vocabulary drift | possible | run cluster stability check: add 100 more transcripts, measure reassignment rate |
| GNN overfitting on 300 graphs | possible | early stopping on val F1; dropout 0.3; weight decay 1e-4; report train/val/test curves |
| graph sparsity limiting GIN | possible | if median graph < 5 nodes, route 2 will likely dominate; note as boundary condition |
| negative result | possible | pre-commit to reporting; frame as boundary condition on elicitation depth |

---

## 12. quick-start checklist

```
[ ] uv installed
[ ] uv init; pyproject.toml configured with pytorch-cpu index
[ ] uv sync — verify torch is CPU-only (check torch.__version__ ends in +cpu)
[ ] torch_geometric import succeeds
[ ] .env with API keys
[ ] python data/download.py — dataset present in data/raw/
[ ] python extraction/tagger.py — test on one transcript
[ ] extraction prompt v1 manually reviewed on 5 transcripts
[ ] model comparison sample IDs fixed in sample_ids.txt
[ ] marimo edit notebooks/01_extraction_review.py — opens without error
```
