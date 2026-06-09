"""Classifier zoo for target-agnostic modality fusion.

Four architectures, all consuming frozen modality embeddings with a shared
interface. Encoders are never fine-tuned — only classifier weights are learned.

Architectures:
- SingleModalityClassifier: MLP on one embedding (baselines)
- StackedClassifier: Concat selected modalities → MLP
- GatedFusionClassifier: Learn per-modality softmax attention → MLP
- LateFusionClassifier: Separate MLP per modality → average logits (ensemble)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class StackedClassifier(nn.Module):
    """Concatenate all selected frozen modalities → MLP.

    The simplest fusion approach — equivalent to the old Route 2/Route 3 pattern.
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


def build_classifier(
    architecture: str,
    modality_dims: dict[str, int],
    n_classes: int,
    hidden: int = 256,
) -> nn.Module:
    """Factory function for classifier instantiation.

    Args:
        architecture: One of ``"single"``, ``"stacked"``, ``"gated"``, ``"late"``.
        modality_dims: Mapping from modality name to embedding dimension.
            For ``"single"``, must contain exactly one modality.
        n_classes: Number of output classes.
        hidden: Hidden layer dimension.

    Returns:
        Instantiated classifier module.
    """
    if architecture == "single":
        if len(modality_dims) != 1:
            raise ValueError(
                f"SingleModalityClassifier expects exactly one modality, "
                f"got {len(modality_dims)}: {list(modality_dims.keys())}"
            )
        modality_name = next(iter(modality_dims.keys()))
        input_dim = modality_dims[modality_name]
        return SingleModalityClassifier(input_dim, n_classes, hidden, modality_name)
    elif architecture == "stacked":
        return StackedClassifier(modality_dims, n_classes, hidden)
    elif architecture == "gated":
        return GatedFusionClassifier(modality_dims, n_classes, hidden)
    elif architecture == "late":
        return LateFusionClassifier(modality_dims, n_classes, hidden)
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: single, stacked, gated, late."
        )
