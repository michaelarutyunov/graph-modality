"""Config-driven experiment runner for modality fusion.

Runs any ExperimentConfig: loads frozen .npz, builds classifier, trains,
evaluates on test set, and saves results.

Usage:
    uv run python classification/fusion/run.py           # full sweep
    uv run python classification/fusion/run.py --dry-run # print sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from classification.fusion.config import (
    ExperimentConfig,
    build_sweep,
    print_sweep_summary,
)
from classification.fusion.models import build_classifier
from classification.fusion.train import Trainer, TrainingConfig, plot_curves
from encoding.build_dataset import load_dataset


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run a single experiment end-to-end.

    Args:
        cfg: Fully-specified experiment configuration.

    Returns:
        Dictionary with training results and test metrics.
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {cfg.tag}")
    print(f"  Target: {cfg.target} ({cfg.n_classes} classes)")
    print(f"  Modalities: {cfg.modalities}")
    print(f"  Architecture: {cfg.architecture}")
    print(f"  Hidden dim: {cfg.hidden_dim}")
    print(f"{'='*70}")

    # ── Load data ──────────────────────────────────────────────────────────
    train_data = load_dataset(cfg.target, "train")
    val_data = load_dataset(cfg.target, "val")
    test_data = load_dataset(cfg.target, "test")

    print(f"  Train: {len(train_data['labels'])}, "
          f"Val: {len(val_data['labels'])}, "
          f"Test: {len(test_data['labels'])}")

    # ── Build classifier ───────────────────────────────────────────────────
    modality_dims = cfg.modality_dims
    model = build_classifier(
        architecture=cfg.architecture,
        modality_dims=modality_dims,
        n_classes=cfg.n_classes,
        hidden=cfg.hidden_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {cfg.architecture}, {n_params:,} parameters")

    # ── Train ──────────────────────────────────────────────────────────────
    train_cfg = TrainingConfig(
        n_classes=cfg.n_classes,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        seed=cfg.seed,
    )

    trainer = Trainer(model, train_cfg)
    train_results = trainer.fit(train_data, val_data)

    # ── Test evaluation ────────────────────────────────────────────────────
    test_metrics = trainer.evaluate(test_data)
    print(f"  Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Per-class F1: {test_metrics['per_class_f1']}")

    # ── Save results ───────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    import torch
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"  Model saved to {output_dir / 'model.pt'}")

    # Save training curves
    plot_curves(
        train_results["train_losses"],
        train_results["val_losses"],
        train_results["val_f1s"],
        output_dir / "curves.png",
    )

    # Save test predictions (per-example, for complementarity analysis)
    np.save(output_dir / "test_preds.npy", test_metrics["predictions"])
    np.save(output_dir / "test_labels.npy", test_metrics["labels"])
    print(f"  Predictions saved to {output_dir / 'test_preds.npy'}")

    # Save metrics
    metrics = {
        "tag": cfg.tag,
        "target": cfg.target,
        "modalities": cfg.modalities,
        "architecture": cfg.architecture,
        "n_classes": cfg.n_classes,
        "hidden_dim": cfg.hidden_dim,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "best_val_f1": train_results["best_val_f1"],
        "best_epoch": train_results["best_epoch"],
        "epochs_run": train_results["epochs_run"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Metrics saved to {output_dir / 'metrics.json'}")

    return {**train_results, "test": test_metrics, "metrics": metrics}


def run_sweep(
    configs: list[ExperimentConfig] | None = None,
    skip_existing: bool = True,
) -> list[dict]:
    """Run a full sweep of experiments.

    Args:
        configs: List of experiment configs. Uses ``build_sweep()`` if None.
        skip_existing: If True, skip experiments whose output directory
            already contains metrics.json.

    Returns:
        List of result dictionaries, one per experiment.
    """
    if configs is None:
        configs = build_sweep()

    results: list[dict] = []

    for i, cfg in enumerate(configs):
        output_dir = Path(cfg.output_dir)
        if skip_existing and (output_dir / "metrics.json").exists():
            print(f"\n[{i+1}/{len(configs)}] SKIP {cfg.tag} — already exists")
            # Load existing metrics for summary
            existing = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            results.append({"metrics": existing})
            continue

        print(f"\n[{i+1}/{len(configs)}] Running {cfg.tag}...")
        try:
            result = run_experiment(cfg)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"error": str(e), "tag": cfg.tag})

    return results


def print_comparison_table(results: list[dict]) -> None:
    """Print a summary comparison table of all experiments."""
    print(f"\n{'='*100}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    print(f"{'Tag':<48} {'Target':<14} {'Arch':<8} {'Val F1':<8} {'Test F1':<8} {'Epochs':<8}")
    print("-" * 100)

    for r in results:
        if "error" in r:
            print(f"{r['tag']:<48} ERROR: {r['error'][:30]}")
            continue

        m = r.get("metrics", {})
        tag = m.get("tag", "?")
        target = m.get("target", "?")
        arch = m.get("architecture", "?")
        val_f1 = m.get("best_val_f1", float("nan"))
        test_f1 = m.get("test_macro_f1", float("nan"))
        epochs = m.get("epochs_run", "?")

        print(f"{tag:<48} {target:<14} {arch:<8} {val_f1:<8.4f} {test_f1:<8.4f} {str(epochs):<8}")

    # Best by target
    print(f"\n{'='*100}")
    print("BEST BY TARGET")
    for target in ["ai_adoption", "cohort"]:
        target_results = [
            r["metrics"] for r in results
            if "metrics" in r and r["metrics"].get("target") == target
        ]
        if not target_results:
            continue
        best = max(target_results, key=lambda m: m.get("test_macro_f1", 0.0))
        print(f"  {target}: {best['tag']} — test F1 = {best['test_macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run modality fusion experiments."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep without running experiments.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["ai_adoption", "cohort"],
        default=None,
        help="Run only experiments for a specific target.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even if results already exist.",
    )
    args = parser.parse_args()

    configs = build_sweep()

    if args.target:
        configs = [c for c in configs if c.target == args.target]
        print(f"Filtered to {args.target}: {len(configs)} experiments")

    print_sweep_summary(configs)

    if args.dry_run:
        print("Dry run — no experiments executed.")
    else:
        results = run_sweep(configs, skip_existing=not args.force)
        print_comparison_table(results)
