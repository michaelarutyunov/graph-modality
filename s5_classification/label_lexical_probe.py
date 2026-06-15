"""Lexical-vs-conceptual probe: free-text vs canonical label_bag (B2).

Compares the label_bag baseline (mean-pooled MiniLM embeddings, no edges) when
node labels come from the canonical vocabulary (synonym-collapsed) vs. the raw
free-text labels produced by extraction. If FREETEXT >> CANON, part of the
distributional signal identified in Phase 6 / ADR-0006 is surface-lexical
wording rather than concept identity; if FREETEXT ~ CANON, the signal is
conceptual / synonym-invariant.

This is a characterisation, not a PASS/FAIL test.

Usage:
    uv run python s5_classification/label_lexical_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from s4_encoding.build_dataset import _load_ambivalence_labels
from s5_classification.repeated_eval import make_split
from s5_classification.repeated_run import _ci95

SEEDS = list(range(42, 52))
TARGET = "stance_ambivalence"

CANON_EMB = "cache/label_bag_embeddings.npy"
CANON_IDS = "cache/label_bag_embedding_ids.json"
FREETEXT_EMB = "cache/label_bag_embeddings_free_text.npy"
FREETEXT_IDS = "cache/label_bag_embedding_ids_free_text.json"

OUT_DIR = Path("results/method-review/label_lexical_probe_v4")
CLASS_NAMES = ["low", "med", "high"]


def _load_embeddings(emb_path: str, ids_path: str) -> tuple[np.ndarray, list[str]]:
    embeddings = np.load(emb_path)
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    return embeddings, ids


def _aligned_matrices() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load CANON and FREETEXT embeddings, aligned on the intersection of ids."""
    canon_emb, canon_ids = _load_embeddings(CANON_EMB, CANON_IDS)
    free_emb, free_ids = _load_embeddings(FREETEXT_EMB, FREETEXT_IDS)

    canon_lookup = {tid: i for i, tid in enumerate(canon_ids)}
    free_lookup = {tid: i for i, tid in enumerate(free_ids)}

    shared_ids = sorted(set(canon_ids) & set(free_ids))
    if not shared_ids:
        raise ValueError("No overlapping transcript_ids between CANON and FREETEXT embeddings")

    canon_idx = [canon_lookup[tid] for tid in shared_ids]
    free_idx = [free_lookup[tid] for tid in shared_ids]

    return canon_emb[canon_idx], free_emb[free_idx], shared_ids


def _fit_probe(
    embeddings: np.ndarray,
    lookup: dict[str, int],
    labels_dict: dict[str, int],
    seed: int,
) -> dict:
    train_ids, val_ids, test_ids = make_split(TARGET, seed)

    def _slice(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        idx = [lookup[tid] for tid in ids]
        x = embeddings[idx]
        y = np.array([labels_dict[tid] for tid in ids], dtype=np.int64)
        return x, y

    x_train, y_train = _slice(train_ids)
    x_val, y_val = _slice(val_ids)
    x_test, y_test = _slice(test_ids)

    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(x_train, y_train)

    val_preds = model.predict(x_val)
    val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))  # type: ignore[arg-type]

    test_preds = model.predict(x_test)
    test_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))  # type: ignore[arg-type]
    per_class_scores = cast(
        "np.ndarray",
        f1_score(y_test, test_preds, average=None, labels=[0, 1, 2], zero_division=0),  # type: ignore[arg-type]
    )
    per_class_f1 = [float(v) for v in per_class_scores]

    return {
        "val_macro_f1": val_f1,
        "test_macro_f1": test_f1,
        "per_class_f1": per_class_f1,
        "predictions": np.asarray(test_preds).tolist(),
        "labels": y_test.tolist(),
    }


def run(out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    canon_emb, free_emb, shared_ids = _aligned_matrices()
    shared_lookup = {tid: i for i, tid in enumerate(shared_ids)}
    labels_dict = _load_ambivalence_labels()

    rows: dict[str, list[dict]] = {"canon": [], "freetext": []}

    runs_path = out_dir / "runs.jsonl"
    with open(runs_path, "w") as f:
        for seed in SEEDS:
            for variant, emb in (("canon", canon_emb), ("freetext", free_emb)):
                result = _fit_probe(emb, shared_lookup, labels_dict, seed)
                row = {"variant": variant, "seed": seed, **result}
                rows[variant].append(row)
                f.write(json.dumps(row) + "\n")
                print(f"[{variant}] seed={seed} test_macro_f1={result['test_macro_f1']:.4f}")

    canon_f1s = np.array([r["test_macro_f1"] for r in rows["canon"]])
    free_f1s = np.array([r["test_macro_f1"] for r in rows["freetext"]])
    delta = free_f1s - canon_f1s

    canon_mean, canon_std, canon_lo, canon_hi = _ci95(canon_f1s)
    free_mean, free_std, free_lo, free_hi = _ci95(free_f1s)
    delta_mean, delta_std, delta_lo, delta_hi = _ci95(delta)

    canon_per_class = np.array([r["per_class_f1"] for r in rows["canon"]]).mean(axis=0)
    free_per_class = np.array([r["per_class_f1"] for r in rows["freetext"]]).mean(axis=0)

    delta_excludes_zero = bool(delta_lo > 0 or delta_hi < 0)
    if delta_excludes_zero and delta_mean > 0:
        interpretation = (
            "FREETEXT > CANON (CI excludes 0, positive) -> signal is partly surface-lexical"
        )
    elif delta_excludes_zero and delta_mean < 0:
        interpretation = (
            "FREETEXT < CANON (CI excludes 0, negative) -> canonicalisation adds signal "
            "(synonym-collapse helps); raw lexical form is noisier"
        )
    else:
        interpretation = (
            "FREETEXT ~ CANON (CI includes 0) -> signal is conceptual / synonym-invariant"
        )

    summary = {
        "target": TARGET,
        "seeds": SEEDS,
        "n_shared_ids": len(shared_ids),
        "variants": {
            "canon": {
                "mean_test_macro_f1": canon_mean,
                "std_test_macro_f1": canon_std,
                "ci95_test_macro_f1": [canon_lo, canon_hi],
                "per_class_f1_mean": {
                    name: float(v) for name, v in zip(CLASS_NAMES, canon_per_class, strict=True)
                },
            },
            "freetext": {
                "mean_test_macro_f1": free_mean,
                "std_test_macro_f1": free_std,
                "ci95_test_macro_f1": [free_lo, free_hi],
                "per_class_f1_mean": {
                    name: float(v) for name, v in zip(CLASS_NAMES, free_per_class, strict=True)
                },
            },
        },
        "delta_freetext_minus_canon": {
            "mean_delta": delta_mean,
            "std_delta": delta_std,
            "ci95_delta": [delta_lo, delta_hi],
            "ci_excludes_zero": delta_excludes_zero,
        },
        "interpretation": interpretation,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
