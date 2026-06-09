"""Experiment configuration for modality fusion experiments.

Each ``ExperimentConfig`` fully specifies one experiment: target, modalities,
architecture, and hyperparameters. The config is hashed for reproducibility
and saved alongside results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Known modality embedding dimensions
MODALITY_DIMS = {
    "text": 768,
    "stats": 30,
    "graph": 128,
}

TARGET_CLASSES = {
    "cohort": 3,       # workforce, creatives, scientists
    "ai_adoption": 2,  # tool_user, integrated
}

Architecture = Literal["single", "stacked", "gated", "late"]
Target = Literal["cohort", "ai_adoption"]


@dataclass
class ExperimentConfig:
    """Fully-specified experiment configuration.

    Attributes:
        tag: Human-readable label, e.g. ``"text-only_single"``.
        target: Classification target.
        modalities: Which frozen embeddings to use, e.g. ``["text", "graph"]``.
        architecture: Classifier architecture.
        hidden_dim: Hidden layer dimension for classifier MLPs.
        lr: Learning rate for Adam optimizer.
        weight_decay: Weight decay for Adam optimizer.
        max_epochs: Maximum training epochs.
        early_stopping_patience: Stop if val F1 doesn't improve for this many epochs.
        seed: Random seed for reproducibility.
    """

    tag: str
    target: Target
    modalities: list[str]
    architecture: Architecture
    hidden_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    early_stopping_patience: int = 10
    seed: int = 42

    @property
    def n_classes(self) -> int:
        """Number of output classes for this target."""
        return TARGET_CLASSES[self.target]

    @property
    def modality_dims(self) -> dict[str, int]:
        """Mapping from modality name → embedding dimension."""
        return {m: MODALITY_DIMS[m] for m in self.modalities}

    @property
    def output_dir(self) -> str:
        """Relative output directory for this experiment's results."""
        modality_str = "-".join(self.modalities)
        return f"results/fusion/{self.target}/{self.architecture}_{modality_str}"


def build_sweep() -> list[ExperimentConfig]:
    """Build the full experiment sweep.

    Minimum sweep (16 experiments): AI adoption with 4 modality combos × 4 architectures.
    Extended sweep: adds cohort target with the same combinations.

    Single-modality baselines: text-only, stats-only, GIN-only
    Fusion combos: text+stats, text+GIN, text+stats+GIN
    Architectures: single, stacked, gated, late

    Returns:
        List of ExperimentConfig objects, one per experiment.
    """
    configs: list[ExperimentConfig] = []

    targets: list[Target] = ["ai_adoption", "cohort"]

    modality_combos = [
        ["text"],
        ["stats"],
        ["graph"],
        ["text", "stats"],
        ["text", "graph"],
        ["text", "stats", "graph"],
    ]

    architectures: list[Architecture] = ["single", "stacked", "gated", "late"]

    for target in targets:
        for mods in modality_combos:
            for arch in architectures:
                # SingleModalityClassifier only makes sense with one modality
                if arch == "single" and len(mods) != 1:
                    continue
                # Don't use single architecture for multi-modality combos
                if len(mods) > 1 and arch == "single":
                    continue

                mod_str = "-".join(mods)
                tag = f"{target}_{arch}_{mod_str}"
                configs.append(
                    ExperimentConfig(
                        tag=tag,
                        target=target,
                        modalities=mods,
                        architecture=arch,
                    )
                )

    return configs


def print_sweep_summary(configs: list[ExperimentConfig]) -> None:
    """Print a summary of the experiment sweep."""
    by_target: dict[str, int] = {}
    for cfg in configs:
        by_target[cfg.target] = by_target.get(cfg.target, 0) + 1

    print(f"Total experiments: {len(configs)}")
    for target, count in sorted(by_target.items()):
        print(f"  {target}: {count}")
    print()

    # Print each config as a compact table row
    print(f"{'Tag':<50} {'Target':<15} {'Arch':<8} {'Modalities'}")
    print("-" * 100)
    for cfg in configs:
        mod_str = "+".join(cfg.modalities)
        print(f"{cfg.tag:<50} {cfg.target:<15} {cfg.architecture:<8} {mod_str}")
