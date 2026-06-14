"""Config-driven experiment runner for modality fusion.

Runs any ExperimentConfig: loads frozen .npz, builds classifier, trains,
evaluates on test set, and saves results. Supports two backends:
- ``torch``: PyTorch MLP classifiers (epoch-based training, curves)
- ``sklearn``: scikit-learn classifiers (one-shot fit, no curves)

Usage:
    uv run python classification/train_run.py                          # torch sweep
    uv run python classification/train_run.py --sweep sklearn          # sklearn sweep
    uv run python classification/train_run.py --sweep all --dry-run    # full plan
    uv run python classification/train_run.py --target ai_adoption     # single target
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from s4_encoding.build_dataset import load_dataset
from s5_classification.classifiers import build_classifier
from s5_classification.sklearn_classifier import SklearnClassifier
from s5_classification.train_config import (
    ExperimentConfig,
    build_all_sweeps,
    build_sklearn_sweep,
    build_sweep,
    print_sweep_summary,
)
from s5_classification.train_loop import Trainer, TrainingConfig, plot_curves


def _load_data(cfg: ExperimentConfig) -> tuple[dict, dict, dict]:
    """Load train/val/test data for an experiment.

    If ``cfg.graph_label_source`` is ``"free_text"``, the GNN embeddings
    are swapped for free-text-label embeddings on-the-fly. Text and stats
    are unaffected (the .npz contains canonical-graph defaults for those,
    but stats are identical and text is independent).
    """
    train_data = load_dataset(cfg.target, "train")
    val_data = load_dataset(cfg.target, "val")
    test_data = load_dataset(cfg.target, "test")

    if cfg.graph_label_source == "free_text" and "graph" in cfg.modalities:
        from s4_encoding.graph_gnn_encoder import encode_graphs

        ft_embs, ft_ids = encode_graphs(label_source="free_text")
        ft_idx = {tid: i for i, tid in enumerate(ft_ids)}

        for split_data, split_name in [
            (train_data, "train"),
            (val_data, "val"),
            (test_data, "test"),
        ]:
            tids = split_data["transcript_ids"]
            idx = [ft_idx[t] for t in tids if t in ft_idx]
            if len(idx) != len(tids):
                raise ValueError(
                    f"Missing free-text graph embeddings for "
                    f"{len(tids) - len(idx)} transcripts in {split_name}"
                )
            split_data["graph_emb"] = ft_embs[idx]
        print("  (swapped graph_emb: canonical → free_text labels + free_text encoder)")

    return train_data, val_data, test_data


def _build_embeddings(
    data: dict[str, np.ndarray],
    modality_keys: list[str],
) -> dict[str, np.ndarray]:
    """Extract selected modality arrays from .npz data.

    .npz keys are ``text_emb``, ``stats_emb``, ``graph_emb``.
    Returns dict with modality name → array (keyed by bare name: text, stats, graph).
    """
    return {m: data[f"{m}_emb"] for m in modality_keys}


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch backend
# ═══════════════════════════════════════════════════════════════════════════════


def _run_torch_on_data(
    cfg: ExperimentConfig,
    train_data: dict,
    val_data: dict,
    test_data: dict,
) -> dict:
    """Build, train, and evaluate a PyTorch model on pre-sliced data.

    No file I/O — returns the fitted model alongside training/test results.
    """
    model = build_classifier(
        architecture=cfg.architecture,
        modality_dims=cfg.modality_dims,
        n_classes=cfg.n_classes,
        hidden=cfg.hidden_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {cfg.architecture}, {n_params:,} parameters")

    train_cfg = TrainingConfig(
        n_classes=cfg.n_classes,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        seed=cfg.seed,
        class_weight=cfg.class_weight,
    )

    trainer = Trainer(model, train_cfg)
    train_results = trainer.fit(train_data, val_data)
    test_metrics = trainer.evaluate(test_data)

    return {"model": model, "train_results": train_results, "test_metrics": test_metrics}


def _run_torch(cfg: ExperimentConfig) -> dict:
    """Run a PyTorch experiment, loading data and saving outputs to disk."""
    train_data, val_data, test_data = _load_data(cfg)

    print(
        f"  Train: {len(train_data['labels'])}, "
        f"Val: {len(val_data['labels'])}, "
        f"Test: {len(test_data['labels'])}"
    )

    fit_result = _run_torch_on_data(cfg, train_data, val_data, test_data)
    model = fit_result["model"]
    train_results = fit_result["train_results"]
    test_metrics = fit_result["test_metrics"]

    import torch

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")

    plot_curves(
        train_results["train_losses"],
        train_results["val_losses"],
        train_results["val_f1s"],
        output_dir / "curves.png",
    )

    return {
        "train_results": train_results,
        "test_metrics": test_metrics,
        "output_dir": output_dir,
        "extra_metrics": {
            "hidden_dim": cfg.hidden_dim,
            "lr": cfg.lr,
            "best_epoch": train_results["best_epoch"],
            "epochs_run": train_results["epochs_run"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sklearn backend
# ═══════════════════════════════════════════════════════════════════════════════


def _run_sklearn_on_data(
    cfg: ExperimentConfig,
    train_data: dict,
    val_data: dict,
    test_data: dict,
) -> dict:
    """Build, fit, and evaluate a sklearn model on pre-sliced data.

    No file I/O — returns the fitted model alongside training/test results.
    """
    model = SklearnClassifier(
        architecture=cfg.architecture,
        modality_names=sorted(cfg.modalities),
        n_classes=cfg.n_classes,
        seed=cfg.seed,
    )
    print(f"  Model: sklearn/{cfg.architecture} on {cfg.modalities}")

    # Fit
    train_emb = _build_embeddings(train_data, cfg.modalities)
    model.fit(train_emb, train_data["labels"])

    # Evaluate on val
    val_emb = _build_embeddings(val_data, cfg.modalities)
    val_preds = model.predict(val_emb)
    val_f1 = float(f1_score(val_data["labels"], val_preds, average="macro", zero_division=0))

    # Evaluate on test
    test_emb = _build_embeddings(test_data, cfg.modalities)
    test_preds = model.predict(test_emb)
    test_f1 = float(f1_score(test_data["labels"], test_preds, average="macro", zero_division=0))
    per_class = f1_score(test_data["labels"], test_preds, average=None, zero_division=0)
    cm = confusion_matrix(test_data["labels"], test_preds).tolist()

    print(f"  Val macro-F1: {val_f1:.4f}")
    print(f"  Test macro-F1: {test_f1:.4f}")

    test_metrics = {
        "macro_f1": test_f1,
        "per_class_f1": {str(i): float(f) for i, f in enumerate(per_class)},
        "confusion_matrix": cm,
        "predictions": test_preds,
        "labels": test_data["labels"],
    }

    train_results = {
        "best_val_f1": val_f1,
        "best_epoch": 0,  # sklearn doesn't epoch
        "epochs_run": 0,  # sklearn doesn't epoch
        "train_losses": [],
        "val_losses": [],
        "val_f1s": [],
    }

    return {"model": model, "train_results": train_results, "test_metrics": test_metrics}


def _run_sklearn(cfg: ExperimentConfig) -> dict:
    """Run a sklearn experiment, loading data and saving outputs to disk."""
    train_data, val_data, test_data = _load_data(cfg)

    print(
        f"  Train: {len(train_data['labels'])}, "
        f"Val: {len(val_data['labels'])}, "
        f"Test: {len(test_data['labels'])}"
    )

    fit_result = _run_sklearn_on_data(cfg, train_data, val_data, test_data)
    model = fit_result["model"]
    train_results = fit_result["train_results"]
    test_metrics = fit_result["test_metrics"]

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")

    # Placeholder curve for sklearn (val F1 as single bar)
    _save_sklearn_curve(train_results["best_val_f1"], output_dir / "curves.png")

    return {
        "train_results": train_results,
        "test_metrics": test_metrics,
        "output_dir": output_dir,
        "extra_metrics": {},
    }


def _save_sklearn_curve(val_f1: float, save_path: Path) -> None:
    """Save a simple bar chart showing the sklearn result (no training curve)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Val macro-F1"], [val_f1], color="steelblue", width=0.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Sklearn — Validation F1")
    ax.axhline(y=val_f1, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(ax.patches, [val_f1], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            fontsize=11,
        )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment_on_data(
    cfg: ExperimentConfig,
    train_data: dict,
    val_data: dict,
    test_data: dict,
) -> dict:
    """Build, fit, and evaluate ``cfg`` on pre-sliced data dicts.

    Dispatches to the torch or sklearn backend based on ``cfg.backend``. Performs
    no file I/O — used by both ``train_run`` (which adds file I/O) and
    ``repeated_run`` (which runs many seeds without persisting models).

    Returns:
        Dict with keys ``model``, ``train_results``, ``test_metrics``.
    """
    if cfg.backend == "sklearn":
        return _run_sklearn_on_data(cfg, train_data, val_data, test_data)
    return _run_torch_on_data(cfg, train_data, val_data, test_data)


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run a single experiment end-to-end.

    Dispatches to the correct backend based on ``cfg.backend``.
    """
    print(f"\n{'=' * 70}")
    print(f"Experiment: {cfg.tag}")
    print(f"  Backend: {cfg.backend}")
    print(f"  Target: {cfg.target} ({cfg.n_classes} classes)")
    print(f"  Modalities: {cfg.modalities}")
    print(f"  Architecture: {cfg.architecture}")
    if "graph" in cfg.modalities:
        print(f"  Graph labels: {cfg.graph_label_source}")
    print(f"{'=' * 70}")

    result = _run_sklearn(cfg) if cfg.backend == "sklearn" else _run_torch(cfg)

    train_results = result["train_results"]
    test_metrics = result["test_metrics"]
    output_dir = result["output_dir"]
    extra = result["extra_metrics"]

    # ── Save common outputs ─────────────────────────────────────────────────
    np.save(output_dir / "test_preds.npy", test_metrics["predictions"])
    np.save(output_dir / "test_labels.npy", test_metrics["labels"])
    print(f"  Predictions saved to {output_dir / 'test_preds.npy'}")

    metrics = {
        "tag": cfg.tag,
        "backend": cfg.backend,
        "target": cfg.target,
        "modalities": cfg.modalities,
        "architecture": cfg.architecture,
        "n_classes": cfg.n_classes,
        "seed": cfg.seed,
        "graph_label_source": cfg.graph_label_source,
        "best_val_f1": train_results["best_val_f1"],
        "best_epoch": train_results["best_epoch"],
        "epochs_run": train_results["epochs_run"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        **extra,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Metrics saved to {output_dir / 'metrics.json'}")

    return {**train_results, "test": test_metrics, "metrics": metrics}


def run_sweep(
    configs: list[ExperimentConfig],
    skip_existing: bool = True,
) -> list[dict]:
    """Run a sweep of experiments."""
    results: list[dict] = []

    for i, cfg in enumerate(configs):
        output_dir = Path(cfg.output_dir)
        if skip_existing and (output_dir / "metrics.json").exists():
            print(f"\n[{i + 1}/{len(configs)}] SKIP {cfg.tag} — already exists")
            existing = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            results.append({"metrics": existing})
            continue

        print(f"\n[{i + 1}/{len(configs)}] Running {cfg.tag}...")
        try:
            result = run_experiment(cfg)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"error": str(e), "tag": cfg.tag})

    return results


def print_comparison_table(results: list[dict]) -> None:
    """Print a summary comparison table of all experiments."""
    print(f"\n{'=' * 110}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 110}")
    header = f"{'Tag':<52} {'Backend':<8} {'Target':<14} {'Arch':<12} {'Val F1':<8} {'Test F1':<8}"
    print(header)
    print("-" * 110)

    for r in results:
        if "error" in r:
            print(f"{r['tag']:<52} ERROR: {r['error'][:30]}")
            continue

        m = r.get("metrics", {})
        tag = m.get("tag", "?")
        backend = m.get("backend", "torch")
        target = m.get("target", "?")
        arch = m.get("architecture", "?")
        val_f1 = m.get("best_val_f1", float("nan"))
        test_f1 = m.get("test_macro_f1", float("nan"))

        print(f"{tag:<52} {backend:<8} {target:<14} {arch:<12} {val_f1:<8.4f} {test_f1:<8.4f}")

    # Best by target
    print(f"\n{'=' * 110}")
    print("BEST BY TARGET")
    for target in ["ai_adoption", "cohort", "stance_ambivalence"]:
        target_results = [
            r["metrics"] for r in results if "metrics" in r and r["metrics"].get("target") == target
        ]
        if not target_results:
            continue
        best = max(target_results, key=lambda m: m.get("test_macro_f1", 0.0))
        print(
            f"  {target}: {best['tag']} (backend={best.get('backend', 'torch')}) — "
            f"test F1 = {best['test_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run modality fusion experiments.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep without running experiments.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["ai_adoption", "cohort", "stance_ambivalence"],
        default=None,
        help="Run only experiments for a specific target.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "sklearn"],
        default=None,
        help="Run only experiments for a specific backend.",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["torch", "sklearn", "all"],
        default="torch",
        help="Which sweep to run (default: torch).",
    )
    parser.add_argument(
        "--graph-label-source",
        type=str,
        choices=["canonical", "free_text"],
        default=None,
        help="Override graph label source for all experiments "
        "(default: canonical). Only affects graph modality.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even if results already exist.",
    )
    args = parser.parse_args()

    # Select configs
    if args.sweep == "torch":
        configs = build_sweep()
    elif args.sweep == "sklearn":
        configs = build_sklearn_sweep()
    else:
        configs = build_all_sweeps()

    if args.target:
        configs = [c for c in configs if c.target == args.target]
        print(f"Filtered to {args.target}: {len(configs)} experiments")
    if args.backend:
        configs = [c for c in configs if c.backend == args.backend]
        print(f"Filtered to backend={args.backend}: {len(configs)} experiments")
    if args.graph_label_source:
        for c in configs:
            c.graph_label_source = args.graph_label_source
        print(f"Graph label source override: {args.graph_label_source}")

    print_sweep_summary(configs)

    if args.dry_run:
        print("Dry run — no experiments executed.")
    else:
        results = run_sweep(configs, skip_existing=not args.force)
        print_comparison_table(results)
