"""Single-modality classifier — MLP baseline on one frozen embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


class SingleModalityClassifier(nn.Module):
    """MLP on a single frozen modality embedding.

    Used for text-only, stats-only, and GIN-only baselines.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden: int = 256,
        modality_name: str = "",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.modality_name = modality_name
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Dict mapping modality name → tensor of shape
                ``(batch_size, dim)``. Uses only ``self.modality_name``
                if set, otherwise the first available modality.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        if self.modality_name and self.modality_name in embeddings:
            x = embeddings[self.modality_name]
        else:
            x = next(iter(embeddings.values()))
        return self.net(x)
