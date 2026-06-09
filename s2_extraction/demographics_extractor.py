"""Full-scale demographic extraction on all 1,250 transcripts via DeepSeek.

Extracts career_stage and ai_adoption from human-only transcript text
using the v2 demographics prompt.  Cache-first: re-running skips already-
extracted transcripts.

Usage:
    uv run python extraction/demographics_extractor.py
    uv run python extraction/demographics_extractor.py --limit 50   # test run
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ── paths ────────────────────────────────────────────────────────────

PROMPT_PATH = Path("s2_extraction/prompts/demographics_v1.txt")
TAGGED_DIR = Path("s1_data/tagged")
CACHE_DIR = Path("cache")
OUTPUT_PATH = CACHE_DIR / "demographics.jsonl"

# ── API config ───────────────────────────────────────────────────────

MODEL_ID = "deepseek-chat"
API_KEY_ENV = "DEEPSEEK_API_KEY"
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_TOKENS = 2048
API_TIMEOUT = 120
MAX_RETRIES = 4
RETRY_DELAYS = [2, 8, 32, 64]

# Rate limiting: requests per second
RATE_LIMIT = 0.8  # seconds between requests (1.25 req/s, safe margin below 1/s limit)

# ── Schema validation ────────────────────────────────────────────────

VALID_CAREER_STAGES = {"early", "mid", "late", "uncertain"}
VALID_AI_ADOPTION = {"novice", "tool_user", "integrated", "power_user", "uncertain"}


def _validate_extraction(extraction: dict, tid: str) -> list[str]:
    """Validate extraction structure. Returns list of warnings."""
    warnings: list[str] = []

    for attr, valid_labels in [
        ("career_stage", VALID_CAREER_STAGES),
        ("ai_adoption", VALID_AI_ADOPTION),
    ]:
        obj = extraction.get(attr)
        if obj is None:
            warnings.append(f"{tid}: missing '{attr}' field")
            continue
        label = obj.get("label", "")
        if label not in valid_labels:
            warnings.append(f"{tid}: invalid {attr} label '{label}'")
        if label != "uncertain" and not obj.get("quotes"):
            warnings.append(f"{tid}: {attr}={label} but no quotes provided")
        if not obj.get("reasoning"):
            warnings.append(f"{tid}: {attr} missing reasoning")

    return warnings


# ── helpers ──────────────────────────────────────────────────────────


def _load_all_transcript_ids() -> list[str]:
    """Load all transcript IDs from tagged data, preserving order."""
    ids: list[str] = []
    for path in sorted(TAGGED_DIR.glob("*.jsonl")):
        if path.name == ".gitkeep":
            continue
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            record = json.loads(line)
            ids.append(record["transcript_id"])
    return ids


def _load_transcript(transcript_id: str) -> dict:
    """Load a single tagged transcript record."""
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


def _load_cache() -> dict[str, dict]:
    """Load existing extractions from cache. Returns tid → extraction dict."""
    if not OUTPUT_PATH.exists():
        return {}
    cached: dict[str, dict] = {}
    for line in OUTPUT_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
            tid = entry.get("transcript_id", "")
            if tid:
                cached[tid] = entry
        except json.JSONDecodeError:
            continue
    return cached


def _call_deepseek(system_prompt: str, user_message: str) -> dict:
    """Call DeepSeek API with JSON mode. Returns parsed JSON response."""
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise ValueError(f"API key env var {API_KEY_ENV} not set")

    body = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                BASE_URL,
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
            last_error = e
            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            print(f"  Retry {attempt + 1}/{MAX_RETRIES} in {delay}s: {e}")
            time.sleep(delay)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")


def _print_distribution(results: dict[str, dict]) -> None:
    """Print class distribution tables."""
    cs_counts: dict[str, int] = {}
    ai_counts: dict[str, int] = {}
    errors = 0

    for _tid, entry in results.items():
        cs = entry.get("career_stage", {}).get("label", "?")
        ai = entry.get("ai_adoption", {}).get("label", "?")
        if cs == "error" or ai == "error":
            errors += 1
        cs_counts[cs] = cs_counts.get(cs, 0) + 1
        ai_counts[ai] = ai_counts.get(ai, 0) + 1

    n = len(results)

    print(f"\n{'=' * 60}")
    print(f"DISTRIBUTION ({n} transcripts")
    if errors:
        print(f"  Errors: {errors}")
    print(f"{'=' * 60}")

    for attr_name, counts, valid_set in [
        ("Career Stage", cs_counts, VALID_CAREER_STAGES),
        ("AI Adoption", ai_counts, VALID_AI_ADOPTION),
    ]:
        print(f"\n--- {attr_name} ---")
        print(f"{'Label':<16} {'Count':>8} {'%':>8}")
        print("-" * 34)
        for label in sorted(counts.keys()):
            pct = counts[label] / n * 100
            marker = " ✓" if label in valid_set else " ✗ INVALID"
            print(f"{label:<16} {counts[label]:>8} {pct:>7.1f}%{marker}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract demographics from all transcripts via DeepSeek"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N transcripts")
    args = parser.parse_args()

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    all_ids = _load_all_transcript_ids()
    total = len(all_ids)

    if args.limit:
        all_ids = all_ids[: args.limit]
        total = len(all_ids)

    # Load existing cache
    cache = _load_cache()
    pending = [tid for tid in all_ids if tid not in cache]
    already = total - len(pending)

    print(f"Total transcripts: {total}")
    print(f"Already cached:   {already}")
    print(f"To extract:        {len(pending)}")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_PATH}")

    if not pending:
        print("\nAll transcripts already cached. Nothing to do.")
        _print_distribution(cache)
        return

    warnings: list[str] = []
    start_time = time.time()
    tokens_estimate = 0  # rough estimate from response content length

    for i, tid in enumerate(pending, 1):
        pct = (already + i) / total * 100
        print(f"\n[{i}/{len(pending)}] ({pct:.0f}% total) {tid}")

        try:
            record = _load_transcript(tid)
            human_text = _build_human_text(record)

            # Truncate very long transcripts
            if len(human_text) > 8000:
                human_text = human_text[:8000] + "\n\n[... truncated]"

            user_message = (
                f"Transcript ID: {tid}\n\n"
                f"Below is the interviewee's responses from an interview about "
                f"AI usage in professional work. Extract career_stage and "
                f"ai_adoption following the schema.\n\n"
                f"{human_text}"
            )

            extraction = _call_deepseek(prompt, user_message)
            extraction["_model"] = MODEL_ID

            # Validate
            warns = _validate_extraction(extraction, tid)
            warnings.extend(warns)
            if warns:
                for w in warns:
                    print(f"  ⚠ {w}")

            cs = extraction.get("career_stage", {}).get("label", "?")
            ai = extraction.get("ai_adoption", {}).get("label", "?")
            print(f"  career_stage={cs}, ai_adoption={ai}")

            # Write incrementally (one line per transcript)
            cache[tid] = extraction
            with open(OUTPUT_PATH, "w") as f:
                for cached_tid in sorted(cache.keys()):
                    json.dump(cache[cached_tid], f, ensure_ascii=False)
                    f.write("\n")

            tokens_estimate += len(json.dumps(extraction))

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            error_entry = {
                "transcript_id": tid,
                "career_stage": {"label": "error", "reasoning": str(e), "quotes": []},
                "ai_adoption": {"label": "error", "reasoning": str(e), "quotes": []},
                "_model": MODEL_ID,
                "_error": str(e),
            }
            cache[tid] = error_entry
            with open(OUTPUT_PATH, "w") as f:
                for cached_tid in sorted(cache.keys()):
                    json.dump(cache[cached_tid], f, ensure_ascii=False)
                    f.write("\n")
            warnings.append(f"{tid}: EXTRACTION FAILED — {e}")

        # Rate limiting
        if i < len(pending):
            time.sleep(RATE_LIMIT)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Processed: {len(pending)} transcripts")
    print(f"Elapsed:   {elapsed:.0f}s ({elapsed / len(pending):.1f}s/transcript)")
    print(f"Output:    {OUTPUT_PATH}")
    print(f"Warnings:  {len(warnings)}")
    if warnings:
        for w in warnings:
            print(f"  - {w}")

    _print_distribution(cache)


if __name__ == "__main__":
    main()
