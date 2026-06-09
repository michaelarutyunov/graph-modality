"""Run DeepSeek vs Agnes demographic extraction on 20 transcripts.

Extracts career_stage and ai_adoption from human-only transcript text,
then compares the two models on agreement rate, uncertainty rate, and
reasoning quality.

Usage:
    uv run python extraction/model_comparison/run_demographics_comparison.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ── paths ────────────────────────────────────────────────────────────

SAMPLE_IDS_PATH = Path("s2_extraction/model_comparison/demographics_sample_ids.txt")
PROMPT_PATH = Path("s2_extraction/prompts/demographics_v1.txt")
TAGGED_DIR = Path("s1_data/tagged")
RESULTS_DIR = Path("s2_extraction/model_comparison/demographics_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── model configs ────────────────────────────────────────────────────

MODELS = [
    {
        "name": "DeepSeek",
        "model_id": "deepseek-chat",
        "api_type": "openai",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "max_tokens": 2048,
        "json_mode": True,
    },
    {
        "name": "Agnes",
        "model_id": "agnes-2.0-flash",
        "api_type": "openai",
        "api_key_env": "AGNES_API_KEY",
        "base_url": "https://apihub.agnes-ai.com/v1/chat/completions",
        "max_tokens": 2048,
    },
]

MAX_RETRIES = 3
RETRY_DELAYS = [2, 8, 32]
API_TIMEOUT = 120


# ── helpers ──────────────────────────────────────────────────────────


def _load_sample_ids() -> list[str]:
    """Load the 20-transcript sample list."""
    ids: list[str] = []
    for line in SAMPLE_IDS_PATH.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            ids.append(line)
    return ids


def _load_transcript(transcript_id: str) -> dict:
    """Load a tagged transcript and return human-only text."""
    for path in sorted(TAGGED_DIR.glob("*.jsonl")):
        if path.name == ".gitkeep":
            continue
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            record = json.loads(line)
            if record["transcript_id"] == transcript_id:
                return record
    raise FileNotFoundError(f"Transcript {transcript_id} not found")


def _build_human_text(record: dict) -> str:
    """Extract human-only turns from a tagged transcript."""
    turns = []
    for t in record["turns"]:
        if t["speaker"] == "Human":
            turns.append(f"[Human]: {t['text']}")
    return "\n\n".join(turns)


def _call_openai(
    model_cfg: dict,
    system_prompt: str,
    user_text: str,
) -> dict:
    """Call an OpenAI-compatible API with JSON mode.

    Returns:
        Parsed JSON response dict.
    """
    api_key = os.environ.get(model_cfg["api_key_env"], "")
    if not api_key:
        raise ValueError(f"API key env var {model_cfg['api_key_env']} not set")

    body: dict = {
        "model": model_cfg["model_id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": model_cfg["max_tokens"],
        "temperature": 0.0,
    }

    if model_cfg.get("json_mode"):
        body["response_format"] = {"type": "json_object"}

    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                model_cfg["base_url"],
                data=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                content = raw["choices"][0]["message"]["content"]
                return json.loads(content)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
            else:
                raise


def _run_model(model_cfg: dict, sample_ids: list[str]) -> dict[str, dict]:
    """Run one model on all sample transcripts.

    Returns:
        Dict mapping transcript_id → parsed extraction result.
    """
    name = model_cfg["name"]
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    cache_path = RESULTS_DIR / f"{name.lower()}_results.json"

    # Check cache
    if cache_path.exists():
        print(f"Loading cached {name} results...")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    results: dict[str, dict] = {}
    n = len(sample_ids)

    for i, tid in enumerate(sample_ids, 1):
        print(f"\n[{name}] {i}/{n}: {tid}")

        record = _load_transcript(tid)
        human_text = _build_human_text(record)

        # Truncate to ~8000 chars to stay within context limits
        # (most transcripts are 2000-5000 chars human-only)
        if len(human_text) > 8000:
            human_text = human_text[:8000] + "\n\n[... truncated for length]"

        user_message = (
            f"Transcript ID: {tid}\n\n"
            f"Below is the interviewee's responses from an interview about "
            f"AI usage in professional work. Extract career_stage and "
            f"ai_adoption following the schema.\n\n"
            f"{human_text}"
        )

        try:
            extraction = _call_openai(model_cfg, prompt, user_message)
            extraction["_model"] = name
            results[tid] = extraction
            cs = extraction.get("career_stage", {}).get("label", "?")
            ai = extraction.get("ai_adoption", {}).get("label", "?")
            print(f"  career_stage={cs}, ai_adoption={ai}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[tid] = {
                "transcript_id": tid,
                "career_stage": {"label": "error", "reasoning": str(e), "quotes": []},
                "ai_adoption": {"label": "error", "reasoning": str(e), "quotes": []},
                "_model": name,
                "_error": str(e),
            }

    # Cache
    cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n{name} results saved to {cache_path}")
    return results


# ── comparison ───────────────────────────────────────────────────────


def _compare(deepseek: dict, agnes: dict) -> None:
    """Compare DeepSeek and Agnes results."""
    common_ids = sorted(set(deepseek.keys()) & set(agnes.keys()))

    print("\n" + "=" * 70)
    print("DEEPSEEK vs AGNES — DEMOGRAPHIC EXTRACTION COMPARISON")
    print("=" * 70)

    # Per-transcript table
    print(
        f"\n{'transcript_id':<22} {'DS career':>12} {'AG career':>12} "
        f"{'agree?':>8}  {'DS AI':>12} {'AG AI':>12} {'agree?':>8}"
    )
    print("-" * 90)

    cs_agree = 0
    ai_agree = 0
    cs_ds_uncertain = 0
    cs_ag_uncertain = 0
    ai_ds_uncertain = 0
    ai_ag_uncertain = 0
    n = len(common_ids)

    for tid in common_ids:
        ds = deepseek[tid]
        ag = agnes[tid]

        ds_cs = ds.get("career_stage", {}).get("label", "?")
        ag_cs = ag.get("career_stage", {}).get("label", "?")
        ds_ai = ds.get("ai_adoption", {}).get("label", "?")
        ag_ai = ag.get("ai_adoption", {}).get("label", "?")

        cs_match = ds_cs == ag_cs
        ai_match = ds_ai == ag_ai

        if cs_match:
            cs_agree += 1
        if ai_match:
            ai_agree += 1
        if ds_cs == "uncertain":
            cs_ds_uncertain += 1
        if ag_cs == "uncertain":
            cs_ag_uncertain += 1
        if ds_ai == "uncertain":
            ai_ds_uncertain += 1
        if ag_ai == "uncertain":
            ai_ag_uncertain += 1

        print(
            f"{tid:<22} {ds_cs:>12} {ag_cs:>12} "
            f"{'✓' if cs_match else '✗':>8}  "
            f"{ds_ai:>12} {ag_ai:>12} "
            f"{'✓' if ai_match else '✗':>8}"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<35} {'Career Stage':>15} {'AI Adoption':>15}")
    print("-" * 67)
    print(f"{'Agreement rate':<35} {cs_agree / n:>14.1%}  {ai_agree / n:>14.1%}")
    print(
        f"{'DeepSeek uncertain rate':<35} {cs_ds_uncertain / n:>14.1%}  {ai_ds_uncertain / n:>14.1%}"
    )
    print(
        f"{'Agnes uncertain rate':<35} {cs_ag_uncertain / n:>14.1%}  {ai_ag_uncertain / n:>14.1%}"
    )

    # Class distribution comparison
    for attr, attr_name in [("career_stage", "Career Stage"), ("ai_adoption", "AI Adoption")]:
        print(f"\n--- {attr_name} class distribution ---")
        ds_counts: dict[str, int] = {}
        ag_counts: dict[str, int] = {}
        for tid in common_ids:
            ds_label = deepseek[tid].get(attr, {}).get("label", "?")
            ag_label = agnes[tid].get(attr, {}).get("label", "?")
            ds_counts[ds_label] = ds_counts.get(ds_label, 0) + 1
            ag_counts[ag_label] = ag_counts.get(ag_label, 0) + 1

        all_labels = sorted(set(list(ds_counts.keys()) + list(ag_counts.keys())))
        print(f"{'Label':<16} {'DeepSeek':>10} {'Agnes':>10}")
        print("-" * 38)
        for label in all_labels:
            print(f"{label:<16} {ds_counts.get(label, 0):>10} {ag_counts.get(label, 0):>10}")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    sample_ids = _load_sample_ids()
    print(f"Loaded {len(sample_ids)} sample transcript IDs")

    results: dict[str, dict[str, dict]] = {}

    for model_cfg in MODELS:
        name = model_cfg["name"]
        print(f"\n{'=' * 60}")
        print(f"Running {name} ({model_cfg['model_id']})...")
        print(f"{'=' * 60}")
        results[name.lower()] = _run_model(model_cfg, sample_ids)

    _compare(results["deepseek"], results["agnes"])

    # Print a few reasoning traces for qualitative review
    print("\n" + "=" * 70)
    print("QUALITATIVE SPOT-CHECK (first 3 transcripts)")
    print("=" * 70)
    for tid in sample_ids[:3]:
        print(f"\n--- {tid} ---")
        for model_name in ["DeepSeek", "Agnes"]:
            r = results[model_name.lower()][tid]
            cs = r.get("career_stage", {})
            ai = r.get("ai_adoption", {})
            print(f"\n  [{model_name}]")
            print(f"    career_stage: {cs.get('label')}")
            if cs.get("quotes"):
                print(f'    quote: "{cs["quotes"][0][:120]}..."')
            print(f"    ai_adoption: {ai.get('label')}")
            if ai.get("quotes"):
                print(f'    quote: "{ai["quotes"][0][:120]}..."')


if __name__ == "__main__":
    main()
