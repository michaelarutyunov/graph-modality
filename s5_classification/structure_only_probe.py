"""v4 structure_only probe: topology > chance on v4 graphs (Phase 2.6, bead n62).

Trains GIN with structure_only features (type one-hot + degree, 5-d) on v4_think
graphs, per-seed frozen-CI protocol, both targets. Tests whether topology alone
clears the chance baseline under the v4 ontology.

Usage:
    PYTHONPATH=. uv run python s5_classification/structure_only_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

from s4_encoding.build_dataset import _load_ai_adoption_labels, _load_ambivalence_labels
from s4_encoding.graph_dataset import GraphDataset
from s4_encoding.graph_gnn_encoder import (
    IN_CHANNELS_STRUCTURE_ONLY,
    N_NODE_TYPES,
    V4_FREE_TEXT_DIR,
    GINAutoencoder,
    GINEncoder,
)
from s5_classification.repeated_eval import make_split
from s5_classification.split import load_transcript_ids_with_labels

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
PROBE_DIR = CACHE_DIR / "structure_only_probe_v4"

# ── Training config (mirrors graph_gnn_encoder.py) ───────────────────────────
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5

SEEDS = list(range(42, 52))
TARGETS = ["ai_adoption", "cohort", "stance_ambivalence"]

CHANCE = {"cohort": 0.2959, "ai_adoption": 0.3367}


# ── Helpers ──────────────────────────────────────────────────────────────────


def majority_class_macro_f1(y: np.ndarray) -> float:
    """Macro-F1 of a majority-class-only classifier on ``y`` (chance baseline)."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(y.reshape(-1, 1), y)
    return float(f1_score(y, clf.predict(y.reshape(-1, 1)), average="macro"))


def _load_labels(target: str) -> dict[str, int]:
    if target == "cohort":
        return load_transcript_ids_with_labels()
    elif target == "ai_adoption":
        return _load_ai_adoption_labels()
    elif target == "stance_ambivalence":
        return _load_ambivalence_labels()
    raise ValueError(f"Unknown target: {target!r}")


def _chance(target: str, ids_to_labels: dict[str, int]) -> float:
    if target in CHANCE:
        return CHANCE[target]
    y = np.array(list(ids_to_labels.values()), dtype=np.int64)
    return majority_class_macro_f1(y)


def _load_graphs() -> list:
    """Pre-load all v4_think graphs as PyG Data objects (structure_only)."""
    paths = sorted(V4_FREE_TEXT_DIR.glob("*.json"))
    dataset = GraphDataset(paths, [-1] * len(paths), feature_mode="structure_only")
    return [dataset[i] for i in range(len(dataset))]


def _train_gin(data_list: list, seed: int) -> GINEncoder:
    """Train GIN autoencoder on v4 graphs (structure_only). Returns frozen encoder."""
    torch.manual_seed(seed)
    device = torch.device("cpu")

    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=True)

    model = GINAutoencoder(in_channels=IN_CHANNELS_STRUCTURE_ONLY).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
    )

    best_loss = float("inf")
    epochs_no_improve = 0

    for _epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            node_labels = batch.x[:, :N_NODE_TYPES].argmax(dim=1).long()
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, node_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break

    model.encoder.eval()
    return model.encoder


def _encode(encoder: GINEncoder, data_list: list) -> np.ndarray:
    """Produce 128-dim frozen embeddings for all graphs."""
    device = torch.device("cpu")
    encoder = encoder.to(device)
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            graph_emb, _ = encoder(
                batch.x.to(device),
                batch.edge_index.to(device),
                batch.batch.to(device),
            )
            embeddings.append(graph_emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> float:
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    return float(f1_score(y_test, clf.predict(X_test), average="macro"))


def run_target(target: str, data_list: list, graph_paths: list[Path]) -> dict:
    """Run 10-seed structure_only probe for one target."""
    ids_list = [p.stem for p in graph_paths]
    ids_to_labels = _load_labels(target)
    chance = _chance(target, ids_to_labels)

    results: list[dict] = []
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        train_ids, _val_ids, test_ids = make_split(target, seed)

        encoder = _train_gin(data_list, seed)
        embeddings = _encode(encoder, data_list)

        emb_lookup = {tid: i for i, tid in enumerate(ids_list)}
        X_train = np.array([embeddings[emb_lookup[tid]] for tid in train_ids])
        X_test = np.array([embeddings[emb_lookup[tid]] for tid in test_ids])
        y_train = np.array([ids_to_labels[tid] for tid in train_ids], dtype=np.int64)
        y_test = np.array([ids_to_labels[tid] for tid in test_ids], dtype=np.int64)

        f1 = _eval(X_train, y_train, X_test, y_test, seed)
        delta = f1 - chance
        print(f"F1={f1:.4f} delta={delta:+.4f}")
        results.append({"seed": seed, "f1": f1, "delta": delta})

    f1s = np.array([r["f1"] for r in results])
    deltas = np.array([r["delta"] for r in results])
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1))
    n = len(deltas)
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_delta, scale=std_delta / np.sqrt(n))
    ci_excludes_zero = bool(ci_low > 0)
    meets_threshold = bool(mean_delta >= 0.01)
    verdict = "PASS" if (ci_excludes_zero and meets_threshold) else "FAIL"

    return {
        "target": target,
        "chance": chance,
        "results": results,
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s, ddof=1)),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_95": [float(ci_low), float(ci_high)],
        "ci_excludes_zero": ci_excludes_zero,
        "meets_threshold": meets_threshold,
        "verdict": verdict,
    }


def main() -> None:
    print("=" * 60)
    print("v4 structure_only > chance probe")
    print(f"Seeds: {SEEDS[0]}-{SEEDS[-1]}, targets: {TARGETS}")
    print("=" * 60)

    print("\nLoading v4_think graphs (structure_only)...")
    graph_paths = sorted(V4_FREE_TEXT_DIR.glob("*.json"))
    data_list = _load_graphs()
    print(f"Loaded {len(data_list)} graphs")

    all_results: dict[str, dict] = {}
    for target in TARGETS:
        print(f"\n--- Target: {target} ---")
        all_results[target] = run_target(target, data_list, graph_paths)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    for target in TARGETS:
        r = all_results[target]
        print(f"\n{target}:")
        print(f"  Mean F1:        {r['mean_f1']:.4f} +/- {r['std_f1']:.4f}")
        print(f"  Chance:          {r['chance']:.4f}")
        print(f"  Mean delta:      {r['mean_delta']:+.4f}")
        print(f"  95% CI:          [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")
        print(f"  CI excludes 0:   {r['ci_excludes_zero']}")
        print(f"  delta >= +0.01:  {r['meets_threshold']}")
        print(f"  VERDICT:         {r['verdict']}")

    # ── Persist ──────────────────────────────────────────────────────────
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    rp = PROBE_DIR / "results.json"
    with open(rp, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {rp}")

    # ── Append to results-log ────────────────────────────────────────────
    _append_to_log(all_results)


def _append_to_log(all_results: dict[str, dict]) -> None:
    log_path = Path(".claude/context/results-log.md")
    entry = "\n### Phase 2.6 — v4 structure_only > chance\n\n"
    entry += "**Date:** 2026-06-12 | **Protocol:** 10-seed frozen CI (42-51)\n\n"

    for target, r in all_results.items():
        if target == "cohort":
            target_label = "cohort (workforce/creatives/scientists)"
        elif target == "ai_adoption":
            target_label = "ai_adoption (tool_user/integrated)"
        else:
            target_label = "stance_ambivalence (low/med/high)"
        entry += f"**Target: {target_label}**\n\n"
        entry += f"- Mean F1: {r['mean_f1']:.4f} +/- {r['std_f1']:.4f}\n"
        entry += f"- Chance: {r['chance']:.4f}\n"
        entry += f"- Mean delta: {r['mean_delta']:+.4f}\n"
        entry += f"- 95% CI: [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]\n"
        entry += f"- CI excludes 0: {r['ci_excludes_zero']}\n"
        entry += f"- delta >= +0.01: {r['meets_threshold']}\n"
        entry += f"- **Verdict: {r['verdict']}**\n\n"

        entry += "| Seed | F1 | delta vs chance |\n"
        entry += "|------|----|----|\n"
        for s in r["results"]:
            entry += f"| {s['seed']} | {s['f1']:.4f} | {s['delta']:+.4f} |\n"
        entry += "\n"

    entry += "---\n"

    with open(log_path, "a") as f:
        f.write(entry)
    print(f"Verdict appended to {log_path}")


if __name__ == "__main__":
    main()
