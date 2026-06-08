"""Quick evaluation of all 5 routes on test set — graph-only routes included.

Usage:
    uv run python classification/eval_all_routes.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch_geometric.loader import DataLoader

from classification.route3b import GraphOnlyClassifier
from classification.split import load_split
from encoding.gnn.model import GraphEncoder
from encoding.gnn.train import (
    BATCH_SIZE,
    CANONICAL_DIR,
    COHORT_TO_LABEL,
    GRAPH_EMB_DIM,
    IN_CHANNELS,
    N_CLASSES,
    PREFIX_TO_COHORT,
    build_prefix_to_transcript_id,
    gather_text_embeddings,
    load_text_embedding_dict,
    precompute_graph_data,
)
from encoding.graph_stats import compute_all_stats
from encoding.text_encoder import encode_transcripts

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

LABEL_MAP = {0: "workforce", 1: "creatives", 2: "scientists"}
LABEL_NAMES = [LABEL_MAP[i] for i in range(3)]


def _metrics(y_true, y_pred, route_name, **extra):
    """Compute standard metrics dict."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class = f1_score(y_true, y_pred, average=None)
    return {
        "route": route_name,
        "macro_f1": macro_f1,
        "per_class_f1": {LABEL_MAP[i]: float(per_class[i]) for i in range(3)},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=LABEL_NAMES
        ),
        "n_samples": len(y_true),
        **extra,
    }


def _rt1(test_ids, labels_dict, text_emb, text_ids):
    """Route 1: text-only."""
    model = joblib.load(CACHE_DIR / "baseline_model.joblib")
    id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    idx = [id_to_idx[t] for t in test_ids if t in id_to_idx]
    X = text_emb[idx]
    y = np.array([labels_dict[t] for t in test_ids if t in id_to_idx], dtype=np.int32)
    return _metrics(y, model.predict(X), "route1_text_only")


def _rt2(test_ids, labels_dict):
    """Route 2: text + stats."""
    model = joblib.load(CACHE_DIR / "route2_model.joblib")
    text_emb, text_ids = encode_transcripts()
    stats_arr, stat_ids = compute_all_stats(graph_dir=Path("data/graphs/canonical"))
    t_map = {tid: i for i, tid in enumerate(text_ids)}
    s_map = {tid: i for i, tid in enumerate(stat_ids)}
    common = sorted(set(text_ids) & set(stat_ids))
    X = np.concatenate(
        [
            np.array([text_emb[t_map[t]] for t in common], dtype=np.float32),
            np.array([stats_arr[s_map[t]] for t in common], dtype=np.float32),
        ],
        axis=1,
    )
    test_set = set(test_ids)
    idx = [i for i, t in enumerate(common) if t in test_set]
    y = np.array([labels_dict[common[i]] for i in idx], dtype=np.int32)
    return _metrics(y, model.predict(X[idx]), "route2_text_stats")


def _rt2b(test_ids, labels_dict):
    """Route 2b: stats only."""
    model = joblib.load(CACHE_DIR / "route2b_model.joblib")
    stats, stat_ids = compute_all_stats(graph_dir=Path("data/graphs/canonical"))
    s_map = {tid: i for i, tid in enumerate(stat_ids)}
    idx = [s_map[t] for t in test_ids if t in s_map]
    X = stats[idx]
    y = np.array([labels_dict[t] for t in test_ids if t in s_map], dtype=np.int32)
    return _metrics(y, model.predict(X), "route2b_stats_only")


def _rt3():
    """Route 3: text + GIN."""
    _train_ids, _val_ids, test_ids, _labels = load_split()
    prefix_to_tid = build_prefix_to_transcript_id(CANONICAL_DIR)
    tid_to_path = {}
    for stem, tid in prefix_to_tid.items():
        tid_to_path[tid] = CANONICAL_DIR / f"{stem}.json"

    paths = [tid_to_path[t] for t in test_ids if t in tid_to_path]
    labels_list = [
        COHORT_TO_LABEL[PREFIX_TO_COHORT[t.rsplit("_", 1)[0]]]
        for t in test_ids
        if t in tid_to_path
    ]

    text_emb_dict = load_text_embedding_dict()
    data = precompute_graph_data(paths, labels_list)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    model = GraphEncoder(
        in_channels=IN_CHANNELS, out_channels=GRAPH_EMB_DIM, n_classes=N_CLASSES
    ).to(device)
    model.load_state_dict(torch.load(CACHE_DIR / "best_gin.pt", weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            tids = batch.transcript_id
            tembs = gather_text_embeddings(tids, text_emb_dict).to(device)
            preds = model(batch, tembs).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    return _metrics(np.array(all_labels), np.array(all_preds), "route3_text_gin")


def _rt3b():
    """Route 3b: GIN only."""
    _train_ids, _val_ids, test_ids, _labels = load_split()
    prefix_to_tid = build_prefix_to_transcript_id(CANONICAL_DIR)
    tid_to_path = {}
    for stem, tid in prefix_to_tid.items():
        tid_to_path[tid] = CANONICAL_DIR / f"{stem}.json"

    paths = [tid_to_path[t] for t in test_ids if t in tid_to_path]
    labels_list = [
        COHORT_TO_LABEL[PREFIX_TO_COHORT[t.rsplit("_", 1)[0]]]
        for t in test_ids
        if t in tid_to_path
    ]

    data = precompute_graph_data(paths, labels_list)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    model = GraphOnlyClassifier().to(device)
    model.load_state_dict(
        torch.load(CACHE_DIR / "best_gin_graph_only.pt", weights_only=True)
    )
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    return _metrics(np.array(all_labels), np.array(all_preds), "route3b_gin_only")


def main():
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    print(f"Test set evaluation — {ts}")

    _train_ids, _val_ids, test_ids, labels_dict = load_split()
    print(f"Test set: {len(test_ids)} transcripts")

    text_emb, text_ids = encode_transcripts()

    # Evaluate all 5 routes
    r1 = _rt1(test_ids, labels_dict, text_emb, text_ids)
    print(f"  R1  (text):         {r1['macro_f1']:.4f}")

    r2 = _rt2(test_ids, labels_dict)
    print(f"  R2  (text+stats):   {r2['macro_f1']:.4f}")

    r3 = _rt3()
    print(f"  R3  (text+GIN):     {r3['macro_f1']:.4f}")

    r2b = _rt2b(test_ids, labels_dict)
    print(f"  R2b (stats only):   {r2b['macro_f1']:.4f}")

    r3b = _rt3b()
    print(f"  R3b (GIN only):     {r3b['macro_f1']:.4f}")

    # ── Table ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("FINAL TEST SET EVALUATION — ALL 5 ROUTES")
    print("=" * 75)
    print(f"{'Route':<28} {'Modalities':<15} {'macro-F1':>10} {'Δ vs R1':>10}")
    print("-" * 75)
    for name, mods, f1, delta in [
        ("Route 1 (Text-only)", "text", r1["macro_f1"], 0.0),
        ("Route 2 (Text + Stats)", "text+stats", r2["macro_f1"], r2["macro_f1"] - r1["macro_f1"]),
        ("Route 3 (Text + GIN)", "text+gin", r3["macro_f1"], r3["macro_f1"] - r1["macro_f1"]),
        ("Route 2b (Stats only)", "stats", r2b["macro_f1"], r2b["macro_f1"] - r1["macro_f1"]),
        ("Route 3b (GIN only)", "gin", r3b["macro_f1"], r3b["macro_f1"] - r1["macro_f1"]),
    ]:
        print(f"{name:<28} {mods:<15} {f1:>10.4f} {delta:>+10.4f}")
    print("-" * 75)

    # Per-class
    print("\nPer-class F1:")
    print(f"{'Class':<14} {'R1':>8} {'R2':>8} {'R3':>8} {'R2b':>8} {'R3b':>8}")
    print("-" * 56)
    for cls_name in LABEL_NAMES:
        print(
            f"{cls_name:<14} "
            f"{r1['per_class_f1'][cls_name]:>8.4f} "
            f"{r2['per_class_f1'][cls_name]:>8.4f} "
            f"{r3['per_class_f1'][cls_name]:>8.4f} "
            f"{r2b['per_class_f1'][cls_name]:>8.4f} "
            f"{r3b['per_class_f1'][cls_name]:>8.4f}"
        )
    print("=" * 75)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for tag, r in [
        ("route1", r1), ("route2", r2), ("route3", r3),
        ("route2b", r2b), ("route3b", r3b),
    ]:
        p = RESULTS_DIR / f"{tag}_{ts}.json"
        with open(p, "w") as f:
            json.dump(r, f, indent=2)
        print(f"Saved {p}")

    comparison = {
        "timestamp": ts,
        "routes": {
            "R1_text": r1["macro_f1"],
            "R2_text_stats": r2["macro_f1"],
            "R3_text_gin": r3["macro_f1"],
            "R2b_stats_only": r2b["macro_f1"],
            "R3b_gin_only": r3b["macro_f1"],
        },
    }
    comp_path = RESULTS_DIR / f"comparison_5routes_{ts}.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved {comp_path}")


if __name__ == "__main__":
    main()
