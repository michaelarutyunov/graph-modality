"""Late fusion classifier — separate MLP per modality → average logits (ensemble)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LateFusionClassifier(nn.Module):
    """Separate classifier per modality → average logits (ensemble).

    Each modality gets its own MLP classifier. Final prediction is the
    average of all modality-specific logits.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_classes: int,
        hidden: int = 128,
    ):
        super().__init__()
        self.modality_names = sorted(modality_dims.keys())
        self.n_classes = n_classes

        self.classifiers = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )
            for m, d in modality_dims.items()
        })

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run each modality's classifier and average the logits.

        Args:
            embeddings: Dict mapping modality name → tensor of shape
                ``(batch_size, dim)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        logits = [
            self.classifiers[m](embeddings[m])
            for m in self.modality_names
        ]
        return torch.stack(logits, dim=0).mean(dim=0)
