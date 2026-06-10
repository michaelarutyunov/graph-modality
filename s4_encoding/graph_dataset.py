"""PyTorch Geometric Dataset wrapping extracted concept graphs.

Each graph is converted to a ``torch_geometric.data.Data`` object with:
- ``x``: node features — type one-hot (4) + label embedding (384) = 388 dims
- ``edge_index``: adjacency list (2 x n_edges)
- ``edge_attr``: relation type one-hot (4)
- ``y``: cohort label (0=workforce, 1=creatives, 2=scientists)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import torch
from torch_geometric.data import Data, Dataset

if TYPE_CHECKING:
    from pathlib import Path

ENTITY_TYPES = ["Construct", "Value", "Stance", "CognitiveStyleMarker"]
RELATIONS = ["SERVES", "EXPRESSED_VIA", "MODULATED_BY", "CONFLICTS_WITH"]
TYPE_TO_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}
REL_TO_IDX = {r: i for i, r in enumerate(RELATIONS)}

SPLIT_TO_LABEL = {"workforce": 0, "creatives": 1, "scientists": 2}

FeatureMode = Literal["full", "structure_only"]


class GraphDataset(Dataset):
    """PyG Dataset over extracted concept graphs."""

    def __init__(
        self,
        graph_paths: list[Path],
        labels: list[int] | None = None,
        label_encoder_name: str = "all-MiniLM-L6-v2",
        feature_mode: Literal["full", "structure_only"] = "full",
    ):
        super().__init__()
        self.graph_paths = graph_paths
        self._labels = labels
        self.feature_mode = feature_mode
        self._label_encoder = None
        if feature_mode != "structure_only":
            from sentence_transformers import SentenceTransformer

            self._label_encoder = SentenceTransformer(label_encoder_name)

    def len(self) -> int:
        return len(self.graph_paths)

    def get(self, idx: int) -> Data:
        g_data = json.loads(self.graph_paths[idx].read_text(encoding="utf-8"))
        nodes = g_data.get("nodes", [])
        edges = g_data.get("edges", [])

        node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

        type_onehots = torch.zeros(len(nodes), 4)
        for i, n in enumerate(nodes):
            ntype = n.get("type", "")
            if ntype in TYPE_TO_IDX:
                type_onehots[i, TYPE_TO_IDX[ntype]] = 1.0

        if self.feature_mode == "structure_only":
            # ── node features: type one-hot (4) + degree (1) = 5 dims ──
            degree = torch.zeros(len(nodes), 1)
            for e in edges:
                if e["source"] in node_id_to_idx:
                    degree[node_id_to_idx[e["source"]], 0] += 1.0
                if e["target"] in node_id_to_idx:
                    degree[node_id_to_idx[e["target"]], 0] += 1.0
            x = torch.cat([type_onehots, degree], dim=1)  # (n_nodes, 5)
        else:
            # ── node features: type one-hot (4) + label embedding (384) ──
            assert self._label_encoder is not None
            labels_text = [n.get("label", "") for n in nodes]
            label_embeddings = self._label_encoder.encode(
                labels_text, normalize_embeddings=True, show_progress_bar=False
            )
            x = torch.cat(
                [
                    type_onehots,
                    torch.tensor(label_embeddings, dtype=torch.float32),
                ],
                dim=1,
            )  # (n_nodes, 388)

        # ── edge index and attributes ──────────────────────────────
        if edges:
            edge_index = (
                torch.tensor(
                    [[node_id_to_idx[e["source"]], node_id_to_idx[e["target"]]] for e in edges],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )  # (2, n_edges)

            edge_attr = torch.zeros(len(edges), 4)
            for i, e in enumerate(edges):
                rel = e.get("relation", "?")
                if rel in REL_TO_IDX:
                    edge_attr[i, REL_TO_IDX[rel]] = 1.0
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4))

        y = (
            torch.tensor(self._labels[idx], dtype=torch.long)
            if self._labels is not None
            else torch.tensor(-1, dtype=torch.long)
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            transcript_id=g_data.get("transcript_id", ""),
        )
