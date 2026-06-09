"""Stacked classifier — concatenate all selected frozen modalities → MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


class StackedClassifier(nn.Module):
    """Concatenate all selected frozen modalities → MLP.

    The simplest fusion approach.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_classes: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.modality_names = sorted(modality_dims.keys())
        total_dim = sum(modality_dims[m] for m in self.modality_names)
        self.total_dim = total_dim
        self.n_classes = n_classes

        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate embeddings and classify.

        Args:
            embeddings: Dict mapping modality name → tensor of shape
                ``(batch_size, dim)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        ordered = [embeddings[m] for m in self.modality_names]
        concat = torch.cat(ordered, dim=1)
        return self.net(concat)
