"""Repeated-evaluation runner with paired statistics.

Implements the frozen evaluation protocol (`docs/method-review/00-evaluation-protocol.md`):
runs the full torch + sklearn experiment matrix across 10 seeds, aggregates per
``(target, modality_combo, architecture, backend)``, applies the validation-based
selection rule, and computes paired text-vs-fusion deltas with 95% t-CIs and McNemar's
exact test.

Usage:
    uv run python s5_classification/repeated_run.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

from s5_classification.repeated_eval import get_split_data
from s5_classification.train_config import (
    Target,
    build_sklearn_sweep,
    build_sweep,
)
from s5_classification.train_run import run_experiment_on_data

SEEDS = list(range(10))
TARGETS: list[Target] = ["ai_adoption", "cohort"]
TEXT_ONLY_COMBO = ("text",)
FUSION_COMBOS = (
    ("text", "stats"),
    ("text", "graph"),
    ("text", "stats", "graph"),
)

OUT_DIR = Path("results/method_review/phase1")
RUNS_PATH = OUT_DIR / "runs.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

T_975_DF9 = float(t_dist.ppf(0.975, 9))


def _ci95(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_lo, ci_hi) for a 95% t-CI with df=9."""
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    margin = T_975_DF9 * std / np.sqrt(len(values))
    return mean, std, mean - margin, mean + margin


def _majority_class_macro_f1(train_labels: np.ndarray, test_labels: np.ndarray) -> float:
    """Macro-F1 of always predicting the train-majority class on the test split."""
    majority = int(np.bincount(train_labels).argmax())
    preds = np.full_like(test_labels, majority)
    return float(f1_score(test_labels, preds, average="macro", zero_division=0))


def run_all() -> dict:
    """Run the full sweep x10 seeds, write runs.jsonl and summary.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_configs = build_sweep() + build_sklearn_sweep()

    rows: list[dict] = []
    chance_scores: dict[str, list[float]] = defaultdict(list)

    # Cache split data per (target, seed) to avoid reloading modalities repeatedly.
    split_cache: dict[tuple[str, int], tuple[dict, dict, dict]] = {}

    with open(RUNS_PATH, "w") as f:
        for cfg in base_configs:
            for seed in SEEDS:
                key = (cfg.target, seed)
                if key not in split_cache:
                    split_cache[key] = get_split_data(cfg.target, seed)
                train_data, val_data, test_data = split_cache[key]

                # Chance baseline: only need to compute once per (target, seed).
                chance_key = f"{cfg.target}_seed{seed}"
                if chance_key not in chance_scores:
                    chance_f1 = _majority_class_macro_f1(train_data["labels"], test_data["labels"])
                    chance_scores[cfg.target].append(chance_f1)

                seed_cfg = replace(cfg, seed=seed)
                print(
                    f"[{cfg.target}|{cfg.architecture}|{cfg.backend}|{cfg.modalities}] seed={seed}"
                )
                result = run_experiment_on_data(seed_cfg, train_data, val_data, test_data)

                train_results = result["train_results"]
                test_metrics = result["test_metrics"]

                row = {
                    "tag": cfg.tag,
                    "target": cfg.target,
                    "modality_combo": cfg.modalities,
                    "architecture": cfg.architecture,
                    "backend": cfg.backend,
                    "seed": seed,
                    "val_macro_f1": train_results["best_val_f1"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "per_class_f1": test_metrics["per_class_f1"],
                    "predictions": np.asarray(test_metrics["predictions"]).tolist(),
                    "labels": np.asarray(test_metrics["labels"]).tolist(),
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")

    summary = _build_summary(rows, chance_scores)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _config_key(target: str, modalities: list[str], architecture: str, backend: str) -> str:
    return f"{target}|{'-'.join(modalities)}|{architecture}|{backend}"


def _build_summary(rows: list[dict], chance_scores: dict[str, list[float]]) -> dict:
    # Group rows by (target, modality_combo, architecture, backend)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = _config_key(row["target"], row["modality_combo"], row["architecture"], row["backend"])
        groups[key].append(row)

    configs_summary: dict[str, dict] = {}
    for key, group_rows in groups.items():
        group_rows = sorted(group_rows, key=lambda r: r["seed"])
        test_f1s = np.array([r["test_macro_f1"] for r in group_rows])
        val_f1s = np.array([r["val_macro_f1"] for r in group_rows])
        mean, std, ci_lo, ci_hi = _ci95(test_f1s)
        configs_summary[key] = {
            "target": group_rows[0]["target"],
            "modality_combo": group_rows[0]["modality_combo"],
            "architecture": group_rows[0]["architecture"],
            "backend": group_rows[0]["backend"],
            "n_seeds": len(group_rows),
            "mean_val_macro_f1": float(np.mean(val_f1s)),
            "mean_test_macro_f1": mean,
            "std_test_macro_f1": std,
            "ci95_test_macro_f1": [ci_lo, ci_hi],
        }

    # Selection rule: per (target, modality_combo), pick config with highest mean val F1.
    by_target_combo: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for key, info in configs_summary.items():
        combo = tuple(info["modality_combo"])
        by_target_combo[(info["target"], combo)].append(key)

    selected: dict[str, dict] = {}
    for (target, combo), keys in by_target_combo.items():
        best_key = max(keys, key=lambda k: configs_summary[k]["mean_val_macro_f1"])
        sel_key = f"{target}|{'-'.join(combo)}"
        selected[sel_key] = {**configs_summary[best_key], "config_key": best_key}

    # Chance baseline per target (mean across seeds).
    chance_summary: dict[str, dict] = {}
    for target, scores in chance_scores.items():
        scores_arr = np.array(scores)
        mean, std, ci_lo, ci_hi = _ci95(scores_arr)
        chance_summary[target] = {
            "mean_macro_f1": mean,
            "std_macro_f1": std,
            "ci95_macro_f1": [ci_lo, ci_hi],
        }

    # Paired text-vs-fusion deltas.
    rows_by_key: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        key = _config_key(row["target"], row["modality_combo"], row["architecture"], row["backend"])
        rows_by_key[key][row["seed"]] = row

    deltas: dict[str, dict] = {}
    for target in TARGETS:
        text_sel_key = f"{target}|{'-'.join(TEXT_ONLY_COMBO)}"
        if text_sel_key not in selected:
            continue
        text_config_key = selected[text_sel_key]["config_key"]
        text_rows = rows_by_key[text_config_key]

        for combo in FUSION_COMBOS:
            fusion_sel_key = f"{target}|{'-'.join(combo)}"
            if fusion_sel_key not in selected:
                continue
            fusion_config_key = selected[fusion_sel_key]["config_key"]
            fusion_rows = rows_by_key[fusion_config_key]

            per_seed_delta = np.array(
                [
                    fusion_rows[seed]["test_macro_f1"] - text_rows[seed]["test_macro_f1"]
                    for seed in SEEDS
                ]
            )
            mean, std, ci_lo, ci_hi = _ci95(per_seed_delta)
            real_effect = bool((ci_lo > 0 or ci_hi < 0) and mean >= 0.01)

            # McNemar exact test at seed=0.
            text_preds0 = np.array(text_rows[0]["predictions"])
            fusion_preds0 = np.array(fusion_rows[0]["predictions"])
            labels0 = np.array(text_rows[0]["labels"])
            text_correct = text_preds0 == labels0
            fusion_correct = fusion_preds0 == labels0

            table = np.zeros((2, 2), dtype=int)
            for tc, fc in zip(text_correct, fusion_correct, strict=False):
                table[int(tc), int(fc)] += 1

            mcnemar_result = mcnemar(table, exact=True)
            mcnemar_p = float(mcnemar_result.pvalue)  # type: ignore[attr-defined]

            deltas[fusion_sel_key] = {
                "target": target,
                "fusion_combo": list(combo),
                "text_config_key": text_config_key,
                "fusion_config_key": fusion_config_key,
                "mean_delta": mean,
                "std_delta": std,
                "ci95_delta": [ci_lo, ci_hi],
                "real_effect": real_effect,
                "mcnemar_p_seed0": mcnemar_p,
            }

    return {
        "seeds": SEEDS,
        "configs": configs_summary,
        "selected": selected,
        "deltas": deltas,
        "chance_baseline": chance_summary,
    }


if __name__ == "__main__":
    summary = run_all()
    print(json.dumps(summary["selected"], indent=2))
    print(json.dumps(summary["deltas"], indent=2))
    print(json.dumps(summary["chance_baseline"], indent=2))
