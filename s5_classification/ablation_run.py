"""Graph-vs-labels disentanglement runner (P2.4 — Method-Review Phase 2).

Compares five single-modality embedding variants through the Phase-1
repeated-evaluation protocol (10 seeds, t-CIs, paired deltas):

  - text             : SBERT 768-d (raw transcript text)
  - label_bag (c)    : pooled MiniLM label embeddings, no edges (P2.2)
  - structure_only(b): GIN trained on type+degree only, no label semantics (P2.1)
  - full_gin (a)     : GIN trained on type+label embeddings (existing canonical)
  - masked_gin (a')  : GIN with masked node-type objective (P2.3)

Each variant is probed with a single-modality logistic regression
(``ablation_probe.probe_variant``) — fixed capacity isolates the embedding's
information content. Renders the epic kill-criterion verdict.

Usage:
    uv run python s5_classification/ablation_run.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from s5_classification.ablation_probe import probe_variant
from s5_classification.repeated_eval import get_split_data
from s5_classification.repeated_run import _ci95, _majority_class_macro_f1

SEEDS = list(range(10))
TARGETS = ["ai_adoption", "cohort"]

VARIANTS: dict[str, tuple[str, str]] = {
    "text": ("cache/text_embeddings_human_only.npy", "cache/text_embedding_ids_human_only.json"),
    "label_bag": ("cache/label_bag_embeddings.npy", "cache/label_bag_embedding_ids.json"),
    "structure_only": (
        "cache/gin_embeddings_structure_only.npy",
        "cache/gin_embedding_ids_structure_only.json",
    ),
    "full_gin": ("cache/gin_embeddings_canonical.npy", "cache/gin_embedding_ids_canonical.json"),
    "masked_gin": ("cache/gin_embeddings_masked.npy", "cache/gin_embedding_ids_masked.json"),
}

OUT_DIR = Path("results/method_review/phase2")
RUNS_PATH = OUT_DIR / "runs.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"


def run_all() -> dict:
    """Run all 5 variants x 10 seeds x 2 targets, write runs.jsonl + summary.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    chance_scores: dict[str, list[float]] = defaultdict(list)

    with open(RUNS_PATH, "w") as f:
        for target in TARGETS:
            for seed in SEEDS:
                # Chance baseline (majority-class on the protocol split, same as P1.3).
                train_data, _val_data, test_data = get_split_data(target, seed)
                chance_scores[target].append(
                    _majority_class_macro_f1(train_data["labels"], test_data["labels"])
                )

                for variant, (emb_path, ids_path) in VARIANTS.items():
                    print(f"[{target}|{variant}] seed={seed}")
                    result = probe_variant(emb_path, ids_path, target, seed)
                    row = {
                        "target": target,
                        "variant": variant,
                        "seed": seed,
                        "val_macro_f1": result["val_macro_f1"],
                        "test_macro_f1": result["test_macro_f1"],
                        "predictions": result["predictions"],
                        "labels": result["labels"],
                    }
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")

    summary = _build_summary(rows, chance_scores)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _build_summary(rows: list[dict], chance_scores: dict[str, list[float]]) -> dict:
    rows_by_key: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        rows_by_key[(row["target"], row["variant"])][row["seed"]] = row

    variants_summary: dict[str, dict] = {}
    for (target, variant), seed_rows in rows_by_key.items():
        test_f1s = np.array([seed_rows[s]["test_macro_f1"] for s in SEEDS])
        mean, std, ci_lo, ci_hi = _ci95(test_f1s)
        variants_summary[f"{target}|{variant}"] = {
            "target": target,
            "variant": variant,
            "mean_test_macro_f1": mean,
            "std_test_macro_f1": std,
            "ci95_test_macro_f1": [ci_lo, ci_hi],
        }

    chance_summary: dict[str, dict] = {}
    for target, scores in chance_scores.items():
        mean, std, ci_lo, ci_hi = _ci95(np.array(scores))
        chance_summary[target] = {
            "mean_macro_f1": mean,
            "std_macro_f1": std,
            "ci95_macro_f1": [ci_lo, ci_hi],
        }

    def _paired_delta(target: str, variant_a: str, variant_b: str) -> dict:
        """mean/CI of per-seed (variant_a - variant_b) test macro-F1, plus real_effect."""
        a_rows = rows_by_key[(target, variant_a)]
        b_rows = rows_by_key[(target, variant_b)]
        per_seed = np.array(
            [a_rows[s]["test_macro_f1"] - b_rows[s]["test_macro_f1"] for s in SEEDS]
        )
        mean, std, ci_lo, ci_hi = _ci95(per_seed)
        real_effect = bool((ci_lo > 0 or ci_hi < 0) and mean >= 0.01)
        return {
            "mean_delta": mean,
            "std_delta": std,
            "ci95_delta": [ci_lo, ci_hi],
            "real_effect": real_effect,
        }

    def _paired_delta_vs_chance(target: str, variant: str) -> dict:
        """mean/CI of per-seed (variant - chance) test macro-F1, plus real_effect."""
        v_rows = rows_by_key[(target, variant)]
        per_seed = np.array([v_rows[s]["test_macro_f1"] - chance_scores[target][s] for s in SEEDS])
        mean, std, ci_lo, ci_hi = _ci95(per_seed)
        real_effect = bool((ci_lo > 0 or ci_hi < 0) and mean >= 0.01)
        return {
            "mean_delta": mean,
            "std_delta": std,
            "ci95_delta": [ci_lo, ci_hi],
            "real_effect": real_effect,
        }

    deltas: dict[str, dict] = {}
    for target in TARGETS:
        deltas[f"{target}|full_gin-label_bag"] = {
            "comparison": "(a) full_gin - (c) label_bag",
            **_paired_delta(target, "full_gin", "label_bag"),
        }
        deltas[f"{target}|structure_only-chance"] = {
            "comparison": "(b) structure_only - chance",
            **_paired_delta_vs_chance(target, "structure_only"),
        }
        deltas[f"{target}|masked_gin-full_gin"] = {
            "comparison": "(a') masked_gin - (a) full_gin",
            **_paired_delta(target, "masked_gin", "full_gin"),
        }

    return {
        "seeds": SEEDS,
        "variants": variants_summary,
        "chance_baseline": chance_summary,
        "deltas": deltas,
    }


if __name__ == "__main__":
    summary = run_all()
    print(json.dumps(summary["variants"], indent=2))
    print(json.dumps(summary["deltas"], indent=2))
    print(json.dumps(summary["chance_baseline"], indent=2))
