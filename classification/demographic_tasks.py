"""Classify demographic targets — AI adoption (binary) and career stage (3-class).

Evaluates all 5 routes on each target: text-only (R1), text+stats (R2),
text+GIN (R3), stats-only (R2b), GIN-only (R3b).

Usage:
    uv run python classification/demographic_tasks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.loader import DataLoader

from classification.route3b import GraphOnlyClassifier
from encoding.gnn.model import GraphEncoder
from encoding.gnn.train import (
    BATCH_SIZE,
    CANONICAL_DIR,
    COHORT_TO_LABEL,
    GRAPH_EMB_DIM,
    IN_CHANNELS,
    N_CLASSES as GNN_N_CLASSES,
    PREFIX_TO_COHORT,
    build_prefix_to_transcript_id,
    gather_text_embeddings,
    load_text_embedding_dict,
    precompute_graph_data,
)
from encoding.graph_stats import compute_all_stats
from encoding.text_encoder import encode_transcripts

# ── Paths ────────────────────────────────────────────────────────────

CACHE_DIR = Path("cache")
DEMOGRAPHICS_PATH = CACHE_DIR / "demographics.jsonl"

# ── Target configs ───────────────────────────────────────────────────

AI_TARGET = "ai_adoption"
AI_CLASSES = ["tool_user", "integrated"]  # binary, drop novice/power_user
AI_DROPS = {"novice", "power_user"}

CS_TARGET = "career_stage"
CS_CLASSES = ["early", "mid", "late"]  # 3-class, drop uncertain
CS_DROPS = {"uncertain"}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

TEXT_DIM = 768
STATS_DIM = 30


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════


def _load_demographics() -> dict[str, dict]:
    """Load demographic extractions. Returns tid → extraction dict."""
    entries: dict[str, dict] = {}
    for line in DEMOGRAPHICS_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        e = json.loads(line)
        tid = e.get("transcript_id", "")
        if tid:
            entries[tid] = e
    return entries


def _build_target(
    attr: str,
    classes: list[str],
    drops: set[str],
) -> tuple[list[str], list[int], dict[int, str]]:
    """Build a classification target from demographics data.

    Returns:
        (transcript_ids, labels, label_map) where label_map is int → class_name.
    """
    demos = _load_demographics()
    label_map = {i: name for i, name in enumerate(classes)}

    ids: list[str] = []
    labels: list[int] = []

    for tid, entry in sorted(demos.items()):
        label = entry.get(attr, {}).get("label", "?")
        if label in drops:
            continue
        if label not in classes:
            continue
        ids.append(tid)
        labels.append(classes.index(label))

    return ids, labels, label_map


def _stratified_split(
    ids: list[str],
    labels: list[int],
) -> tuple[list[str], list[str], list[str], list[int], list[int], list[int]]:
    """Create stratified 70/15/15 split.

    Returns:
        (train_ids, val_ids, test_ids, train_labels, val_labels, test_labels)
    """
    train_ids, temp_ids, train_l, temp_l = train_test_split(
        ids, labels, train_size=TRAIN_RATIO, stratify=labels, random_state=SEED,
    )
    val_ids, test_ids, val_l, test_l = train_test_split(
        temp_ids, temp_l, train_size=0.5, stratify=temp_l, random_state=SEED,
    )
    return train_ids, val_ids, test_ids, train_l, val_l, test_l


def _print_split_info(
    target_name: str,
    train_l: list[int],
    val_l: list[int],
    test_l: list[int],
    label_map: dict[int, str],
) -> None:
    """Print split summary."""
    print(f"\n{target_name} split:")
    for name, labels in [("train", train_l), ("val", val_l), ("test", test_l)]:
        dist = {label_map[i]: sum(1 for l in labels if l == i) for i in label_map}
        print(f"  {name}: {len(labels)} — {dist}")


# ═══════════════════════════════════════════════════════════════════════
# Logistic regression routes (R1, R2, R2b)
# ═══════════════════════════════════════════════════════════════════════


def _train_lr(X_train, y_train, n_classes):
    """Train balanced logistic regression."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        random_state=SEED,
    ).fit(X_train, y_train)


def _eval_lr(model, X_test, y_test, label_map, route_name):
    """Evaluate LR and return metrics dict."""
    y_pred = model.predict(X_test)
    return _metrics(y_test, y_pred, label_map, route_name)


def _route1_text(
    train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes,
):
    """Text-only LR."""
    text_emb, text_ids = encode_transcripts()
    id_to_idx = {tid: i for i, tid in enumerate(text_ids)}

    def extract(ids):
        idx = [id_to_idx[t] for t in ids if t in id_to_idx]
        return text_emb[idx]
    return _train_and_eval_lr(
        extract, train_ids, val_ids, test_ids, train_l, val_l, test_l,
        label_map, n_classes, f"R1_text_{n_classes}class",
    )


def _route2_text_stats(
    train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes,
):
    """Text + stats LR."""
    text_emb, text_ids = encode_transcripts()
    stats_arr, stat_ids = compute_all_stats(graph_dir=Path("data/graphs/canonical"))
    t_map = {tid: i for i, tid in enumerate(text_ids)}
    s_map = {tid: i for i, tid in enumerate(stat_ids)}
    common = sorted(set(text_ids) & set(stat_ids))

    X_all = np.concatenate(
        [
            np.array([text_emb[t_map[t]] for t in common], dtype=np.float32),
            np.array([stats_arr[s_map[t]] for t in common], dtype=np.float32),
        ],
        axis=1,
    )
    cid_to_idx = {tid: i for i, tid in enumerate(common)}

    def extract(ids):
        idx = [cid_to_idx[t] for t in ids if t in cid_to_idx]
        return X_all[idx]
    return _train_and_eval_lr(
        extract, train_ids, val_ids, test_ids, train_l, val_l, test_l,
        label_map, n_classes, f"R2_text_stats_{n_classes}class",
    )


def _route2b_stats(
    train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes,
):
    """Stats-only LR."""
    stats_arr, stat_ids = compute_all_stats(graph_dir=Path("data/graphs/canonical"))
    s_map = {tid: i for i, tid in enumerate(stat_ids)}

    def extract(ids):
        idx = [s_map[t] for t in ids if t in s_map]
        return stats_arr[idx]
    return _train_and_eval_lr(
        extract, train_ids, val_ids, test_ids, train_l, val_l, test_l,
        label_map, n_classes, f"R2b_stats_{n_classes}class",
    )


def _train_and_eval_lr(
    extract_fn, train_ids, val_ids, test_ids, train_l, val_l, test_l,
    label_map, n_classes, route_name,
):
    """Train LR on train, evaluate on val (for model selection) and test."""
    X_train = extract_fn(train_ids)
    y_train = np.array(train_l, dtype=np.int32)
    X_val = extract_fn(val_ids)
    y_val = np.array(val_l, dtype=np.int32)
    X_test = extract_fn(test_ids)
    y_test = np.array(test_l, dtype=np.int32)

    model = _train_lr(X_train, y_train, n_classes)
    val_result = _eval_lr(model, X_val, y_val, label_map, f"{route_name}_val")
    test_result = _eval_lr(model, X_test, y_test, label_map, route_name)
    return val_result, test_result


# ═══════════════════════════════════════════════════════════════════════
# GIN routes (R3, R3b)
# ═══════════════════════════════════════════════════════════════════════


def _prepare_gin_data(
    ids_list: list[str],
    labels_list: list[int],
    n_classes: int,
    use_text: bool,
) -> tuple[DataLoader, dict | None]:
    """Prepare a DataLoader for GIN evaluation.

    Args:
        ids_list: Transcript IDs for this split.
        labels_list: Integer labels.
        use_text: If True, also return text_emb_dict.

    Returns:
        (loader, text_emb_dict_or_None)
    """
    prefix_to_tid = build_prefix_to_transcript_id(CANONICAL_DIR)
    tid_to_path = {}
    for stem, tid in prefix_to_tid.items():
        tid_to_path[tid] = CANONICAL_DIR / f"{stem}.json"

    paths = [tid_to_path[t] for t in ids_list if t in tid_to_path]
    aligned_labels = []
    for t in ids_list:
        if t in tid_to_path:
            prefix = t.rsplit("_", 1)[0]
            aligned_labels.append(
                COHORT_TO_LABEL.get(
                    PREFIX_TO_COHORT.get(prefix, ""),
                    labels_list[ids_list.index(t)] if t in ids_list else 0,
                )
            )

    # Rebuild labels properly — use the provided demographic labels, not cohort
    demos = _load_demographics()
    aligned_labels = []
    for t in ids_list:
        if t in tid_to_path:
            lbl = demos.get(t, {})
            # Use the demographic label if available, fall back to provided list
            aligned_labels.append(labels_list[ids_list.index(t)])

    data = precompute_graph_data(paths, aligned_labels)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    if use_text:
        text_emb_dict = load_text_embedding_dict()
        return loader, text_emb_dict
    return loader, None


def _train_gin_model(
    train_ids, val_ids, train_l, val_l, n_classes, use_text,
):
    """Train a GIN model from scratch for a demographic target."""
    device = torch.device("cpu")

    # Prepare data
    train_loader, train_text = _prepare_gin_data(
        train_ids, train_l, n_classes, use_text,
    )
    val_loader, val_text = _prepare_gin_data(
        val_ids, val_l, n_classes, use_text,
    )

    # Model
    if use_text:
        model = GraphEncoder(
            in_channels=IN_CHANNELS, out_channels=GRAPH_EMB_DIM, n_classes=n_classes,
        ).to(device)
    else:
        model = GraphOnlyClassifier(
            in_channels=IN_CHANNELS, out_channels=GRAPH_EMB_DIM, n_classes=n_classes,
        ).to(device)

    # Count classes for weighting
    unique, counts = np.unique(train_l, return_counts=True)
    class_weights = torch.tensor(
        len(train_l) / (n_classes * counts.astype(float)), dtype=torch.float32,
    )
    print(f"  Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5,
    )

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, 51):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if use_text:
                tids = batch.transcript_id
                tembs = gather_text_embeddings(tids, train_text).to(device)
                logits = model(batch, tembs)
            else:
                logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if use_text:
                    tids = batch.transcript_id
                    tembs = gather_text_embeddings(tids, val_text).to(device)
                    logits = model(batch, tembs)
                else:
                    logits = model(batch)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())

        val_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, val_loader, val_text, best_val_f1


def _eval_gin(
    model, loader, text_emb_dict, use_text, label_map, route_name,
):
    """Evaluate a trained GIN model and return metrics."""
    device = torch.device("cpu")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_text:
                tids = batch.transcript_id
                tembs = gather_text_embeddings(tids, text_emb_dict).to(device)
                logits = model(batch, tembs)
            else:
                logits = model(batch)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    return _metrics(all_labels, all_preds, label_map, route_name)


def _route3_gin(train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes):
    """Text + GIN."""
    print(f"\n  Training GIN (text+graph, {n_classes}-class)...")
    model, _, _, best_f1 = _train_gin_model(
        train_ids, val_ids, train_l, val_l, n_classes, use_text=True,
    )
    print(f"  Best val F1: {best_f1:.4f}")

    test_loader, test_text = _prepare_gin_data(test_ids, test_l, n_classes, use_text=True)
    result = _eval_gin(
        model, test_loader, test_text, use_text=True,
        label_map=label_map, route_name=f"R3_text_gin_{n_classes}class",
    )
    result["best_val_f1"] = best_f1
    return result


def _route3b_gin(train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes):
    """GIN only."""
    print(f"\n  Training GIN (graph-only, {n_classes}-class)...")
    model, _, _, best_f1 = _train_gin_model(
        train_ids, val_ids, train_l, val_l, n_classes, use_text=False,
    )
    print(f"  Best val F1: {best_f1:.4f}")

    test_loader, _ = _prepare_gin_data(test_ids, test_l, n_classes, use_text=False)
    result = _eval_gin(
        model, test_loader, None, use_text=False,
        label_map=label_map, route_name=f"R3b_gin_{n_classes}class",
    )
    result["best_val_f1"] = best_f1
    return result


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _metrics(y_true, y_pred, label_map, route_name):
    """Standard metrics dict."""
    n_classes = len(label_map)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "route": route_name,
        "macro_f1": macro_f1,
        "per_class_f1": {
            label_map[i]: float(per_class[i]) for i in range(n_classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred,
            target_names=[label_map[i] for i in range(n_classes)],
            zero_division=0,
        ),
    }


def _print_table(target_name, results, label_map):
    """Print comparison table for one target."""
    print(f"\n{'='*70}")
    print(f"{target_name} — TEST SET COMPARISON")
    print(f"{'='*70}")
    print(f"{'Route':<25} {'macro-F1':>10} {'Δ vs text':>10}")
    print("-" * 47)
    text_f1 = results.get("R1_text", {}).get("macro_f1", 0)
    for key, name in [
        ("R1_text", "R1 (text)"),
        ("R2_text_stats", "R2 (text+stats)"),
        ("R3_text_gin", "R3 (text+GIN)"),
        ("R2b_stats", "R2b (stats only)"),
        ("R3b_gin", "R3b (GIN only)"),
    ]:
        r = results.get(key)
        if r is None:
            continue
        delta = r["macro_f1"] - text_f1 if text_f1 else 0
        print(f"{name:<25} {r['macro_f1']:>10.4f} {delta:>+10.4f}")

    # Per-class
    print(f"\n{'Class':<14} ", end="")
    for key in ["R1_text", "R2_text_stats", "R3_text_gin", "R2b_stats", "R3b_gin"]:
        if key in results:
            print(f"{key[:3]:>8}", end=" ")
    print()
    print("-" * (14 + 9 * len([k for k in results if k])))

    for cls_name in label_map.values():
        print(f"{cls_name:<14} ", end="")
        for key in ["R1_text", "R2_text_stats", "R3_text_gin", "R2b_stats", "R3b_gin"]:
            if key in results:
                f1 = results[key].get("per_class_f1", {}).get(cls_name, 0)
                print(f"{f1:>8.4f}", end=" ")
        print()

    # Confusion matrices
    print("\nConfusion matrices:")
    for key, name in [
        ("R1_text", "R1 (text)"),
        ("R2_text_stats", "R2 (text+stats)"),
        ("R3_text_gin", "R3 (text+GIN)"),
        ("R2b_stats", "R2b (stats only)"),
        ("R3b_gin", "R3b (GIN only)"),
    ]:
        r = results.get(key)
        if r is None:
            continue
        cm = np.array(r["confusion_matrix"])
        class_names = [label_map[i] for i in range(len(label_map))]
        print(f"\n  {name}:")
        header = " " * 14 + "".join(f"{n:>10}" for n in class_names)
        print(header)
        for i, name_i in enumerate(class_names):
            row = f"  {name_i:<12}" + "".join(f"{cm[i,j]:>10}" for j in range(len(class_names)))
            print(row)
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def run_target(target_name: str, attr: str, classes: list[str], drops: set[str]):
    """Run all 5 routes on a single classification target.

    Returns:
        Dict of route_key → test_result.
    """
    ids, labels, label_map = _build_target(attr, classes, drops)
    n_classes = len(classes)
    print(f"\n{'#'*60}")
    print(f"# {target_name}")
    print(f"# {n_classes} classes: {classes}")
    print(f"# N={len(ids)} (dropped {drops})")
    print(f"{'#'*60}")

    train_ids, val_ids, test_ids, train_l, val_l, test_l = _stratified_split(
        ids, labels,
    )
    _print_split_info(target_name, train_l, val_l, test_l, label_map)

    results: dict[str, dict] = {}

    # Route 1: text-only LR
    print("\n--- R1: text-only ---")
    _, r1 = _route1_text(
        train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes,
    )
    results["R1_text"] = r1
    print(f"  Test macro-F1: {r1['macro_f1']:.4f}")

    # Route 2: text + stats LR
    print("\n--- R2: text + stats ---")
    _, r2 = _route2_text_stats(
        train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes,
    )
    results["R2_text_stats"] = r2
    print(f"  Test macro-F1: {r2['macro_f1']:.4f}")

    # Route 2b: stats only LR
    print("\n--- R2b: stats only ---")
    _, r2b = _route2b_stats(
        train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes,
    )
    results["R2b_stats"] = r2b
    print(f"  Test macro-F1: {r2b['macro_f1']:.4f}")

    # Route 3: text + GIN
    print("\n--- R3: text + GIN ---")
    r3 = _route3_gin(
        train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes,
    )
    results["R3_text_gin"] = r3
    print(f"  Test macro-F1: {r3['macro_f1']:.4f}")

    # Route 3b: GIN only
    print("\n--- R3b: GIN only ---")
    r3b = _route3b_gin(
        train_ids, val_ids, test_ids, train_l, val_l, test_l, label_map, n_classes,
    )
    results["R3b_gin"] = r3b
    print(f"  Test macro-F1: {r3b['macro_f1']:.4f}")

    _print_table(target_name, results, label_map)
    return results


def main():
    all_results = {}

    # ── AI Adoption (binary) ─────────────────────────────────────────
    all_results["ai_adoption"] = run_target(
        "AI Adoption (tool_user vs integrated)",
        AI_TARGET, AI_CLASSES, AI_DROPS,
    )

    # ── Career Stage (3-class) ───────────────────────────────────────
    all_results["career_stage"] = run_target(
        "Career Stage (early/mid/late)",
        CS_TARGET, CS_CLASSES, CS_DROPS,
    )

    # Save
    import datetime as _dt
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H%M%SZ")
    out = Path("results") / f"demographic_classification_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
