"""Classifier zoo for target-agnostic modality fusion.

Four architectures, all consuming frozen modality embeddings. Encoders are
never fine-tuned — only classifier weights are learned.

Each classifier lives in its own file for readability. This module re-exports
them all and provides the ``build_classifier`` factory function.
"""

from s5_classification.mlp_single import SingleModalityClassifier
from s5_classification.mlp_stacked import StackedClassifier
from s5_classification.mlp_gated import GatedFusionClassifier
from s5_classification.mlp_late import LateFusionClassifier

import torch.nn as nn


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


__all__ = [
    "SingleModalityClassifier",
    "StackedClassifier",
    "GatedFusionClassifier",
    "LateFusionClassifier",
    "build_classifier",
]
