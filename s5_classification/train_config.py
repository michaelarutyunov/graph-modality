"""Experiment configuration for modality fusion experiments.

Each ``ExperimentConfig`` fully specifies one experiment: target, modalities,
architecture, backend, and hyperparameters. The config is saved alongside results
for reproducibility.

Supports two backends:
- ``torch``: PyTorch MLP classifiers (single, stacked, gated, late)
- ``sklearn``: scikit-learn classifiers (logistic, random_forest, gradient_boost, svm)
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
    "cohort": 3,  # workforce, creatives, scientists
    "ai_adoption": 2,  # tool_user, integrated
    "stance_ambivalence": 3,  # low, med, high (ordinal)
}

TorchArchitecture = Literal["single", "stacked", "gated", "late"]
SklearnArchitecture = Literal["logistic", "random_forest", "gradient_boost", "svm"]
Target = Literal["cohort", "ai_adoption", "stance_ambivalence"]
Backend = Literal["torch", "sklearn"]
LabelSource = Literal["canonical", "free_text"]


@dataclass
class ExperimentConfig:
    """Fully-specified experiment configuration.

    Attributes:
        tag: Human-readable label, e.g. ``"ai_adoption_single_text"``.
        target: Classification target.
        modalities: Which frozen embeddings to use, e.g. ``["text", "graph"]``.
        architecture: Classifier architecture.
        backend: ``"torch"`` (MLP) or ``"sklearn"`` (traditional).
        graph_label_source: ``"canonical"`` (default) or ``"free_text"``.
            Which label set the GNN reads. Only affects ``graph`` modality.
            Graph stats (30-dim) use structural fields identical in both.
        hidden_dim: Hidden layer dimension (torch only).
        lr: Learning rate for Adam optimizer (torch only).
        weight_decay: Weight decay for Adam optimizer (torch only).
        max_epochs: Maximum training epochs (torch only).
        early_stopping_patience: Stop if val F1 doesn't improve (torch only).
        seed: Random seed for reproducibility.
    """

    tag: str
    target: Target
    modalities: list[str]
    architecture: str
    backend: Backend = "torch"
    graph_label_source: LabelSource = "canonical"
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
        prefix = f"{self.backend}_" if self.backend != "torch" else ""
        label_suffix = (
            f"_{self.graph_label_source}"
            if ("graph" in self.modalities and self.graph_label_source != "canonical")
            else ""
        )
        return (
            f"results/fusion/{self.target}/{prefix}{self.architecture}_{modality_str}{label_suffix}"
        )


def build_sweep() -> list[ExperimentConfig]:
    """Build the full PyTorch experiment sweep (42 experiments).

    Returns:
        List of ExperimentConfig objects.
    """
    configs: list[ExperimentConfig] = []

    targets: list[Target] = ["ai_adoption", "cohort", "stance_ambivalence"]

    modality_combos = [
        ["text"],
        ["stats"],
        ["graph"],
        ["text", "stats"],
        ["text", "graph"],
        ["text", "stats", "graph"],
    ]

    architectures: list[TorchArchitecture] = ["single", "stacked", "gated", "late"]

    for target in targets:
        for mods in modality_combos:
            for arch in architectures:
                if arch == "single" and len(mods) != 1:
                    continue
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
                        backend="torch",
                    )
                )

    return configs


def build_sklearn_sweep() -> list[ExperimentConfig]:
    """Build the sklearn experiment sweep.

    Covers the same modality combos x targets but with sklearn classifiers.
    All sklearn models work on concatenated features (equivalent to stacked fusion).
    Single-modality baselines included to compare against PyTorch single/mlp.

    Returns:
        List of ExperimentConfig objects.
    """
    configs: list[ExperimentConfig] = []

    targets: list[Target] = ["ai_adoption", "cohort", "stance_ambivalence"]

    modality_combos = [
        ["text"],
        ["stats"],
        ["graph"],
        ["text", "stats"],
        ["text", "graph"],
        ["text", "stats", "graph"],
    ]

    sklearn_archs: list[SklearnArchitecture] = [
        "logistic",
        "random_forest",
        "gradient_boost",
        "svm",
    ]

    for target in targets:
        for mods in modality_combos:
            for arch in sklearn_archs:
                mod_str = "-".join(mods)
                tag = f"{target}_sklearn_{arch}_{mod_str}"
                configs.append(
                    ExperimentConfig(
                        tag=tag,
                        target=target,
                        modalities=mods,
                        architecture=arch,
                        backend="sklearn",
                    )
                )

    return configs


def build_all_sweeps() -> list[ExperimentConfig]:
    """Build combined torch + sklearn sweeps."""
    return build_sweep() + build_sklearn_sweep()


def print_sweep_summary(configs: list[ExperimentConfig]) -> None:
    """Print a summary of the experiment sweep."""
    by_target: dict[str, int] = {}
    by_backend: dict[str, int] = {}
    for cfg in configs:
        by_target[cfg.target] = by_target.get(cfg.target, 0) + 1
        by_backend[cfg.backend] = by_backend.get(cfg.backend, 0) + 1

    print(f"Total experiments: {len(configs)}")
    for target, count in sorted(by_target.items()):
        print(f"  {target}: {count}")
    print(f"  backend: {by_backend}")
    print()

    print(f"{'Tag':<55} {'Backend':<10} {'Target':<14} {'Arch':<16} {'Modalities'}")
    print("-" * 120)
    for cfg in configs:
        mod_str = "+".join(cfg.modalities)
        print(f"{cfg.tag:<55} {cfg.backend:<10} {cfg.target:<14} {cfg.architecture:<16} {mod_str}")
