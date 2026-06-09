"""Route 1 baseline — text-only logistic regression (convenience alias).

Delegates to the unified experiment runner. Equivalent to:
    uv run python classification/train_run.py --sweep sklearn --target cohort

Usage:
    uv run python classification/baseline.py
"""

from __future__ import annotations

from s5_classification.train_config import ExperimentConfig
from s5_classification.train_run import run_experiment

BASELINE_CONFIGS = [
    ExperimentConfig(
        tag="baseline_text_cohort",
        target="cohort",
        modalities=["text"],
        architecture="logistic",
        backend="sklearn",
    ),
    ExperimentConfig(
        tag="baseline_text_ai_adoption",
        target="ai_adoption",
        modalities=["text"],
        architecture="logistic",
        backend="sklearn",
    ),
]

if __name__ == "__main__":
    for cfg in BASELINE_CONFIGS:
        result = run_experiment(cfg)
        print(f"  {cfg.tag}: test F1 = {result['metrics']['test_macro_f1']:.4f}")
    print("\nDone.")
