"""Gated fusion classifier — learn per-modality softmax attention weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionClassifier(nn.Module):
    """Learn per-example modality attention weights via softmax.

    A small gate network predicts per-modality weights from the concatenated
    embeddings. Each modality's embedding is multiplied by its weight before
    classification. Supports N modalities.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_classes: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.modality_names = sorted(modality_dims.keys())
        self.modality_sizes = [modality_dims[m] for m in self.modality_names]
        total = sum(self.modality_sizes)
        n_modalities = len(self.modality_names)
        self.n_classes = n_classes

        # Gate: predict per-modality attention weights from all features
        self.gate = nn.Sequential(
            nn.Linear(total, 128),
            nn.ReLU(),
            nn.Linear(128, n_modalities),
        )

        self.classifier = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply gated fusion and classify.

        Args:
            embeddings: Dict mapping modality name → tensor of shape
                ``(batch_size, dim)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        ordered = [embeddings[m] for m in self.modality_names]
        concat = torch.cat(ordered, dim=1)

        # Per-example softmax weights over modalities
        attn = F.softmax(self.gate(concat), dim=1)  # (B, n_modalities)

        # Expand each modality's weight across its dimensions and multiply
        weights = torch.cat([
            attn[:, i:i + 1].expand(-1, sz)
            for i, sz in enumerate(self.modality_sizes)
        ], dim=1)

        return self.classifier(concat * weights)
