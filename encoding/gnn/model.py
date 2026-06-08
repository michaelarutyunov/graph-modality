"""GIN-based graph encoder for route 3.

Architecture: two GINConv layers → global mean pool → 128-dim graph embedding.
The graph embedding is fused with a 768-dim text embedding before classification.

Reference: Xu et al. (2019) "How Powerful are Graph Neural Networks?" ICLR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GraphEncoder(nn.Module):
    """GIN encoder producing a 128-dim graph-level embedding.

    Args:
        in_channels: Dimension of input node features (default 388).
        hidden: Hidden dimension for GIN MLPs.
        out_channels: Output graph embedding dimension.
        n_classes: Number of classification targets.
    """

    def __init__(
        self,
        in_channels: int = 388,
        hidden: int = 256,
        out_channels: int = 128,
        n_classes: int = 3,
    ):
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

        # Classifier head (fused text + graph embedding)
        self.classifier = nn.Sequential(
            nn.Linear(768 + out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Produce a 128-dim graph embedding from node features."""
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return global_mean_pool(x, batch)  # (batch_size, 128)

    def forward(
        self,
        data,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass: encode graph, fuse with text, classify.

        Args:
            data: A batch of ``torch_geometric.data.Data`` objects.
            text_embeddings: Text embeddings aligned with the batch,
                             shape ``(batch_size, 768)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        graph_emb = self.encode_graph(data.x, data.edge_index, data.batch)
        fused = torch.cat([text_embeddings, graph_emb], dim=1)  # (B, 896)
        return self.classifier(fused)
