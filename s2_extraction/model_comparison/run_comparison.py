"""Run the 3-model comparison experiment on 10 fixed transcripts.

Extracts graphs with each candidate model, computes automated quality
metrics, and produces a comparison table.  Results are cached per-model
so re-runs are fast.

Usage:
    uv run python extraction/model_comparison/run_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from s2_extraction.validator import validate_graph

load_dotenv(override=True)

# ── paths ────────────────────────────────────────────────────────────

SAMPLE_IDS_PATH = Path("s2_extraction/model_comparison/sample_ids.txt")
PROMPT_PATH = Path("s2_extraction/prompts/v1.txt")
TAGGED_DIR = Path("s1_data/tagged")
RESULTS_DIR = Path("s2_extraction/model_comparison/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── model configs ────────────────────────────────────────────────────

# api_type: "anthropic" uses the Anthropic SDK; "openai" uses raw HTTP.
# max_tokens: per-model output budget.  DeepSeek needs extra because
#   thinking tokens share the budget with visible output.
MODELS = [
    {
        "name": "Claude",
        "model_id": "claude-sonnet-4-6",
        "api_type": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "max_tokens": 4096,
    },
    {
        "name": "DeepSeek",
        "model_id": "deepseek-chat",
        "api_type": "openai",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "max_tokens": 8192,
        "json_mode": True,  # enables response_format={"type": "json_object"}
    },
    {
        "name": "Agnes",
        "model_id": "agnes-2.0-flash",
        "api_type": "openai",
        "api_key_env": "AGNES_API_KEY",
        "base_url": "https://apihub.agnes-ai.com/v1/chat/completions",
        "max_tokens": 4096,
    },
]

MAX_RETRIES = 3
RETRY_DELAYS = [2, 8, 32]


# ── helpers ──────────────────────────────────────────────────────────


def _load_sample_ids() -> list[str]:
    """Load the fixed 10-transcript sample list."""
    ids: list[str] = []
    for line in SAMPLE_IDS_PATH.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def _load_tagged() -> dict[str, dict]:
    """Load all tagged transcripts into a lookup dict."""
    records: dict[str, dict] = {}
    for path in sorted(TAGGED_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            records[rec["transcript_id"]] = rec
    return records


def _extract_anthropic(
    prompt: str,
    client: Anthropic,
    model_id: str,
    max_tokens: int,
) -> str | None:
    """Extract via Anthropic-compatible API.  Returns raw text or None."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                timeout=180,
            )
            text_blocks = [b for b in response.content if b.type == "text"]
            if not text_blocks:
                raise ValueError(f"no text block (blocks: {[b.type for b in response.content]})")
            return text_blocks[0].text
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
            else:
                print(f"    FAILED after {MAX_RETRIES} attempts: {exc}")
                return None
    return None


def _extract_openai(
    prompt: str,
    api_key: str,
    base_url: str,
    model_id: str,
    max_tokens: int,
    json_mode: bool = False,
) -> str | None:
    """Extract via OpenAI-compatible API.  Returns raw text or None."""
    messages = [{"role": "user", "content": prompt}]
    # DeepSeek JSON mode requires the word "json" in the prompt
    if json_mode:
        messages.insert(0, {"role": "system", "content": "You must output valid JSON."})

    payload: dict = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                base_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return content
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
            else:
                # Try to extract error detail from response
                detail = str(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        err_body = exc.read().decode("utf-8")
                        detail = err_body[:200]
                    except Exception:
                        pass
                print(f"    FAILED after {MAX_RETRIES} attempts: {detail}")
                return None
    return None


def _parse_graph_json(raw_text: str, transcript_id: str, model_id: str) -> dict | None:
    """Parse JSON from LLM response, validate, and attach metadata."""
    json_text = raw_text.strip()
    # Strip markdown fences
    if json_text.startswith("```"):
        nl = json_text.find("\n")
        if nl != -1:
            json_text = json_text[nl + 1 :]
        if json_text.rstrip().endswith("```"):
            json_text = json_text.rstrip()[: json_text.rstrip().rfind("```")]
    json_text = json_text.strip()

    try:
        graph = json.loads(json_text)
    except json.JSONDecodeError as exc:
        print(f"    JSON parse error: {exc}")
        return None

    graph["transcript_id"] = transcript_id
    graph["extraction_model"] = model_id
    violations = validate_graph(graph)
    graph["validation_violations"] = violations
    return graph


# ── automated metrics (approximate rubric R1-R5) ─────────────────────


def _compute_metrics(graphs: dict[str, dict]) -> dict:
    """Compute automated quality metrics across a set of graphs.

    These approximate the rubric criteria (R1-R5) as quantitative checks.
    Manual review is still needed for final scoring, but these flag outliers.
    """
    if not graphs:
        return {"error": "no graphs to analyse"}

    n_graphs = len(graphs)
    node_counts = [len(g["nodes"]) for g in graphs.values()]
    edge_counts = [len(g["edges"]) for g in graphs.values()]
    violation_counts = [len(g.get("validation_violations", [])) for g in graphs.values()]

    # R1 (size consistency): coefficient of variation of node count
    mean_nodes = sum(node_counts) / n_graphs
    std_nodes = (sum((n - mean_nodes) ** 2 for n in node_counts) / n_graphs) ** 0.5
    cv_nodes = std_nodes / mean_nodes if mean_nodes > 0 else float("inf")

    # R2 (ontology adherence): fraction of graphs with zero violations
    clean_graphs = sum(1 for v in violation_counts if v == 0)

    # R3 (bipolarity capture): mean bipolarity completeness
    bip_scores: list[float] = []
    for g in graphs.values():
        constructs = [n for n in g["nodes"] if n["type"] == "Construct"]
        if constructs:
            bip = sum(1.0 if c.get("bipolarity_complete") else 0.5 for c in constructs) / len(
                constructs
            )
            bip_scores.append(bip)
    mean_bip = sum(bip_scores) / len(bip_scores) if bip_scores else 0.0

    # R4 (hallucination rate proxy): fraction of edges with dangling refs
    # (these are caught by validator already, so use violation type counts)
    total_violations = sum(violation_counts)

    # R5 (relation typing accuracy): count of edges with unknown relation types
    unknown_rel_edges = sum(
        1
        for g in graphs.values()
        for v in g.get("validation_violations", [])
        if "unknown relation" in v
    )

    return {
        "n_graphs": n_graphs,
        "mean_nodes": round(mean_nodes, 1),
        "cv_nodes": round(cv_nodes, 3),  # R1 proxy — lower = more consistent
        "clean_graphs": f"{clean_graphs}/{n_graphs}",  # R2 proxy
        "mean_bipolarity": round(mean_bip, 3),  # R3 proxy
        "total_violations": total_violations,  # R4 proxy
        "unknown_rel_edges": unknown_rel_edges,  # R5 proxy
        "mean_edges": round(sum(edge_counts) / n_graphs, 1),
        "node_range": f"{min(node_counts)}-{max(node_counts)}",
        "edge_range": f"{min(edge_counts)}-{max(edge_counts)}",
    }


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    sample_ids = _load_sample_ids()
    tagged = _load_tagged()
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    # Verify all sample IDs exist
    missing = [tid for tid in sample_ids if tid not in tagged]
    if missing:
        print(f"ERROR: sample IDs not found in tagged data: {missing}")
        sys.exit(1)

    print(f"Model comparison: {len(sample_ids)} transcripts x {len(MODELS)} models")
    print(f"Sample IDs: {', '.join(sample_ids)}")
    print()

    all_results: dict[str, dict] = {}

    for model_cfg in MODELS:
        name = model_cfg["name"]
        api_type = model_cfg["api_type"]
        model_id = model_cfg["model_id"]
        max_tokens = model_cfg["max_tokens"]
        api_key = os.environ.get(model_cfg["api_key_env"], "")
        base_url = model_cfg["base_url"]

        if not api_key:
            print(f"=== {name} === SKIPPED: {model_cfg['api_key_env']} not set")
            print()
            continue

        print(f"=== {name} ({model_id}) === [api={api_type}, max_tokens={max_tokens}]")

        # Create client for Anthropic-compatible APIs
        client = None
        if api_type == "anthropic":
            client = Anthropic(api_key=api_key, base_url=base_url)

        graphs: dict[str, dict] = {}

        for i, tid in enumerate(sample_ids):
            rec = tagged[tid]
            prompt = prompt_template.replace("{transcript}", rec["formatted"])
            print(
                f"  [{i + 1}/{len(sample_ids)}] {tid} "
                f"({rec['split']}, {rec['n_human_turns']} human turns)..."
            )

            # Dispatch to the right API backend
            if api_type == "anthropic":
                if client is None:
                    raise ValueError("Anthropic client required for anthropic API")
                raw_text = _extract_anthropic(prompt, client, model_id, max_tokens)
            else:
                json_mode = model_cfg.get("json_mode", False)
                raw_text = _extract_openai(
                    prompt, api_key, base_url, model_id, max_tokens, json_mode=json_mode
                )

            if raw_text is None:
                print("    → FAILED")
                continue

            graph = _parse_graph_json(raw_text, tid, model_id)
            if graph:
                graphs[tid] = graph
                n_nodes = len(graph["nodes"])
                n_edges = len(graph["edges"])
                n_viol = len(graph.get("validation_violations", []))
                status = f"{n_nodes} nodes, {n_edges} edges"
                if n_viol:
                    status += f", {n_viol} violations ⚠"
                print(f"    → {status}")

                # Save individual result
                out_path = RESULTS_DIR / f"{name.lower()}_{tid}.json"
                out_path.write_text(
                    json.dumps(graph, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                print("    → FAILED")

        metrics = _compute_metrics(graphs)
        all_results[name] = metrics

        if "error" not in metrics:
            print(
                f"  Summary: {metrics['n_graphs']} graphs, "
                f"{metrics['mean_nodes']} nodes avg, "
                f"bip={metrics['mean_bipolarity']:.2f}, "
                f"{metrics['clean_graphs']} clean"
            )
        else:
            print(f"  Summary: {metrics['error']}")
        print()

    # ── comparison table ────────────────────────────────────────────
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    header = (
        f"{'Model':12s} {'Graphs':>7s} {'Nodes':>7s} {'Edges':>7s} "
        f"{'CV(N)':>7s} {'Bip':>6s} {'Clean':>6s} {'Viol':>5s}"
    )
    print(header)
    print("-" * 70)
    for name, m in all_results.items():
        if "error" in m:
            print(f"{name:12s} ERROR: {m['error']}")
            continue
        print(
            f"{name:12s} {m['n_graphs']:>7d} "
            f"{m['mean_nodes']:>7.1f} {m['mean_edges']:>7.1f} "
            f"{m['cv_nodes']:>7.3f} {m['mean_bipolarity']:>6.2f} "
            f"{m['clean_graphs']:>6s} {m['total_violations']:>5d}"
        )

    # Save comparison JSON
    summary_path = RESULTS_DIR / "comparison_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
