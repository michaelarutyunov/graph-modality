"""Extract concept graphs from speaker-tagged transcripts via LLM API.

Loads the versioned extraction prompt, calls the Anthropic API, caches
results, and runs structural validation.  Designed for batch operation
with idempotent cache-first behaviour.

Usage:
    uv run python extraction/extractor.py                       # all uncached
    uv run python extraction/extractor.py --tid work_0000       # single transcript
    uv run python extraction/extractor.py --split workforce     # one split
    uv run python extraction/extractor.py --limit 5             # first 5 uncached
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from extraction.validator import validate_graph

load_dotenv()

# ── paths ────────────────────────────────────────────────────────────

PROMPT_DIR = Path("extraction/prompts")
TAGGED_DIR = Path("data/tagged")
GRAPH_DIR = Path("data/graphs/free_text")
FAILED_LOG = Path("extraction/failed.txt")

# ── API config ───────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 8, 32]  # seconds — exponential-ish
API_TIMEOUT = 120  # seconds


# ── helpers ──────────────────────────────────────────────────────────


def _load_prompt(version: str = "v1") -> str:
    """Load the extraction prompt template from its versioned file."""
    path = PROMPT_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _build_prompt(prompt_template: str, formatted_transcript: str) -> str:
    """Insert the formatted transcript into the prompt template."""
    return prompt_template.replace("{transcript}", formatted_transcript)


def _extract_json_from_response(text: str) -> str:
    """Extract a JSON object from an LLM response that may include markdown fences."""
    text = text.strip()

    # Strip ```json … ``` fences if present
    if text.startswith("```"):
        # find the first newline after the opening fence
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        # strip closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: text.rstrip().rfind("```")]

    return text.strip()


# ── core extraction ──────────────────────────────────────────────────


def extract_one(
    transcript_id: str,
    formatted_transcript: str,
    client: Anthropic,
    model: str = DEFAULT_MODEL,
    prompt_version: str = "v1",
    split: str = "unknown",
) -> dict | None:
    """Extract a concept graph for a single transcript.

    Checks cache first.  Retries on API failure with backoff.
    Returns the parsed graph dict, or None on failure.
    """
    cache_path = GRAPH_DIR / f"{transcript_id}.json"

    # ── cache hit ──────────────────────────────────────────────────
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # ── build prompt ───────────────────────────────────────────────
    template = _load_prompt(prompt_version)
    prompt = _build_prompt(template, formatted_transcript)

    # ── call API with retry ────────────────────────────────────────
    raw_text: str | None = None
    last_error: str = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                timeout=API_TIMEOUT,
            )
            raw_text = response.content[0].text
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(
                    f"  [{transcript_id}] attempt {attempt + 1} failed "
                    f"({last_error[:80]}), retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                _log_failure(transcript_id, last_error)
                return None

    if raw_text is None:
        _log_failure(transcript_id, "all retries exhausted, no response")
        return None

    # ── parse ──────────────────────────────────────────────────────
    try:
        json_text = _extract_json_from_response(raw_text)
        graph = json.loads(json_text)
    except json.JSONDecodeError as exc:
        _log_failure(transcript_id, f"JSON decode error: {exc}")
        return None

    # ── attach metadata ────────────────────────────────────────────
    graph.setdefault("transcript_id", transcript_id)
    graph["split"] = split
    graph["extraction_model"] = model
    graph["prompt_version"] = prompt_version

    # ── validate ───────────────────────────────────────────────────
    violations = validate_graph(graph)
    graph["validation_violations"] = violations
    if violations:
        for v in violations:
            print(f"  [{transcript_id}] VIOLATION: {v}")

    # ── cache ──────────────────────────────────────────────────────
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

    return graph


# ── failure logging ──────────────────────────────────────────────────


def _log_failure(transcript_id: str, reason: str) -> None:
    """Append a failed extraction to the failure log."""
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{transcript_id}\t{reason}\n")


# ── batch extraction ─────────────────────────────────────────────────


def load_tagged_transcripts() -> dict[str, dict]:
    """Load all tagged transcripts from JSONL files into a dict keyed by ID."""
    records: dict[str, dict] = {}
    for path in sorted(TAGGED_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            records[rec["transcript_id"]] = rec
    return records


def extract_batch(
    transcript_ids: list[str],
    tagged: dict[str, dict],
    client: Anthropic,
    model: str = DEFAULT_MODEL,
    prompt_version: str = "v1",
) -> tuple[int, int]:
    """Extract graphs for a list of transcript IDs.

    Returns (success_count, failure_count).
    """
    success = 0
    failure = 0

    for i, tid in enumerate(transcript_ids):
        if tid not in tagged:
            print(f"  [{tid}] not found in tagged data — skipping")
            failure += 1
            continue

        rec = tagged[tid]
        print(
            f"[{i + 1}/{len(transcript_ids)}] {tid} "
            f"({rec['split']}, {rec['n_human_turns']} human turns)..."
        )

        result = extract_one(
            transcript_id=tid,
            formatted_transcript=rec["formatted"],
            client=client,
            model=model,
            prompt_version=prompt_version,
            split=rec["split"],
        )

        if result is None:
            failure += 1
        else:
            success += 1
            v_count = len(result.get("validation_violations", []))
            n_nodes = len(result.get("nodes", []))
            n_edges = len(result.get("edges", []))
            status = f"{n_nodes} nodes, {n_edges} edges"
            if v_count:
                status += f", {v_count} violations"
            print(f"  → {status}")

    return success, failure


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract concept graphs from tagged transcripts")
    parser.add_argument("--tid", type=str, help="Process a single transcript ID")
    parser.add_argument(
        "--split",
        type=str,
        choices=["workforce", "creatives", "scientists"],
        help="Process all transcripts from one split",
    )
    parser.add_argument("--limit", type=int, help="Process at most N uncached transcripts")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--prompt-version", type=str, default="v1", help="Prompt version to use (default: v1)"
    )
    parser.add_argument("--force", action="store_true", help="Re-extract even if cache exists")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    tagged = load_tagged_transcripts()

    if args.tid:
        if args.tid not in tagged:
            print(f"ERROR: transcript '{args.tid}' not found in tagged data")
            sys.exit(1)
        rec = tagged[args.tid]
        if args.force:
            cache_path = GRAPH_DIR / f"{args.tid}.json"
            cache_path.unlink(missing_ok=True)
        result = extract_one(
            transcript_id=args.tid,
            formatted_transcript=rec["formatted"],
            client=client,
            model=args.model,
            prompt_version=args.prompt_version,
            split=rec["split"],
        )
        if result is None:
            print("Extraction failed.")
            sys.exit(1)
        print(
            f"✓ {args.tid}: {len(result.get('nodes', []))} nodes, "
            f"{len(result.get('edges', []))} edges"
        )
        return

    # Build the work list
    ids: list[str] = []
    if args.split:
        ids = [tid for tid, rec in tagged.items() if rec["split"] == args.split]
    else:
        ids = sorted(tagged.keys())

    # Skip cached unless --force
    if not args.force:
        uncached = [tid for tid in ids if not (GRAPH_DIR / f"{tid}.json").exists()]
        skipped = len(ids) - len(uncached)
        if skipped:
            print(f"Skipping {skipped} cached transcripts (use --force to re-extract)")
        ids = uncached

    if args.limit:
        ids = ids[: args.limit]

    if not ids:
        print("No transcripts to process.")
        return

    print(
        f"Processing {len(ids)} transcripts with model={args.model}, prompt={args.prompt_version}"
    )
    success, failure = extract_batch(
        transcript_ids=ids,
        tagged=tagged,
        client=client,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    print(f"\nDone. {success} succeeded, {failure} failed.")


if __name__ == "__main__":
    main()
