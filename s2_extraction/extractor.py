"""Extract concept graphs from speaker-tagged transcripts via LLM API.

Supports Anthropic and OpenAI-compatible backends.  Caches results and
runs structural validation.  Cache-first: never re-extracts if output exists.

Usage:
    uv run python extraction/extractor.py                          # all uncached (DeepSeek)
    uv run python s2_extraction/extractor.py --backend anthropic      # Use Claude
    uv run python s2_extraction/extractor.py --backend agnes           # Use Agnes 2.0 Flash
    uv run python extraction/extractor.py --tid work_0000          # single transcript
    uv run python extraction/extractor.py --split workforce        # one split
    uv run python extraction/extractor.py --limit 100              # first N uncached
"""

from __future__ import annotations

import argparse
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

load_dotenv()

# ── paths ────────────────────────────────────────────────────────────

PROMPT_DIR = Path("s2_extraction/prompts")
TAGGED_DIR = Path("s1_data/tagged")
GRAPH_DIR = Path("s1_data/graphs/free_text")  # legacy v3 corpus — locked, never overwrite
FAILED_LOG = Path("s2_extraction/failed.txt")


def _graph_dir_for(prompt_version: str) -> Path:
    """Output directory for a prompt version.

    v3 keeps the legacy flat path (``s1_data/graphs/free_text``) for
    backward-compatibility with the locked corpus. Every other version is
    namespaced (``s1_data/graphs/<version>/free_text``) so a new extraction
    can never read or overwrite the locked v3 graphs.
    """
    if prompt_version == "v3":
        return GRAPH_DIR
    return Path(f"s1_data/graphs/{prompt_version}/free_text")


# ── backends ─────────────────────────────────────────────────────────

BACKENDS = {
    "deepseek": {
        "type": "openai",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "max_tokens": 8192,
        "json_mode": True,
    },
    "deepseek-think": {
        "type": "openai",
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        # thinking consumes output tokens; raise the budget so reasoning does not
        # crowd out the JSON answer (the Phase-1 truncation failure mode)
        "max_tokens": 16384,
        "json_mode": True,
        "extra_payload": {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        },
    },
    "anthropic": {
        "type": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "max_tokens": 4096,
    },
    "agnes": {
        "type": "openai",
        "model": "agnes-2.0-flash",
        "api_key_env": "AGNES_API_KEY",
        "base_url": "https://apihub.agnes-ai.com/v1/chat/completions",
        "max_tokens": 8192,
        "json_mode": False,  # Agnes does not support response_format={type: json_object}
        "system_message": (
            "You must output valid JSON only. "
            "No markdown fences, no preamble, no explanation — just the JSON object."
        ),
        "chat_template_kwargs": {
            "enable_thinking": True,  # recommended for reasoning/agent tasks
        },
    },
}

MAX_RETRIES = 3
RETRY_DELAYS = [2, 8, 32]
API_TIMEOUT = 180


# ── helpers ──────────────────────────────────────────────────────────


def _load_prompt(version: str = "v3") -> str:
    path = PROMPT_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: text.rstrip().rfind("```")]
    return text.strip()


def _log_failure(transcript_id: str, reason: str) -> None:
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{transcript_id}\t{reason}\n")


# ── API callers ──────────────────────────────────────────────────────


def _call_anthropic(prompt: str, client: Anthropic, model: str, max_tokens: int) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                timeout=API_TIMEOUT,
            )
            text_blocks = [b for b in response.content if b.type == "text"]
            if not text_blocks:
                raise ValueError(f"no text block (blocks: {[b.type for b in response.content]})")
            return text_blocks[0].text
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(RETRY_DELAYS[attempt])
    return None


def _call_openai(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    json_mode: bool = False,
    system_message: str | None = None,
    chat_template_kwargs: dict | None = None,
    extra_payload: dict | None = None,
) -> str | None:
    messages: list[dict] = [{"role": "user", "content": prompt}]
    # System message for JSON guidance — always recommended for structured output.
    # When json_mode is enabled, this is required by some providers (DeepSeek).
    # When json_mode is unavailable (Agnes), this is the primary JSON guarantee.
    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})
    elif json_mode:
        messages.insert(0, {"role": "system", "content": "You must output valid JSON."})

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if extra_payload:
        payload.update(extra_payload)

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
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(RETRY_DELAYS[attempt])
    return None


# ── core extraction ──────────────────────────────────────────────────


def extract_one(
    transcript_id: str,
    formatted_transcript: str,
    backend: dict,
    prompt_version: str = "v4",
    split: str = "unknown",
    domain: str = "AI's role in professional work",
    out_dir: Path | None = None,
    anthropic_client: Anthropic | None = None,
) -> dict | None:
    """Extract a concept graph for a single transcript.

    Checks cache first.  Retries on API failure with backoff.  Writes to
    ``out_dir`` (defaults to the version-namespaced dir for ``prompt_version``).
    """
    if out_dir is None:
        out_dir = _graph_dir_for(prompt_version)
    cache_path = out_dir / f"{transcript_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    template = _load_prompt(prompt_version)
    prompt = template.replace("{transcript}", formatted_transcript).replace("{domain}", domain)

    # Call API
    if backend["type"] == "anthropic":
        if anthropic_client is None:
            raise ValueError("Anthropic client required for anthropic backend")
        raw_text = _call_anthropic(
            prompt, anthropic_client, backend["model"], backend["max_tokens"]
        )
    else:
        raw_text = _call_openai(
            prompt,
            api_key=os.environ[backend["api_key_env"]],
            base_url=backend["base_url"],
            model=backend["model"],
            max_tokens=backend["max_tokens"],
            json_mode=backend.get("json_mode", False),
            system_message=backend.get("system_message"),
            chat_template_kwargs=backend.get("chat_template_kwargs"),
            extra_payload=backend.get("extra_payload"),
        )

    if raw_text is None:
        _log_failure(transcript_id, "API call failed after retries")
        return None

    # Parse
    try:
        json_text = _strip_markdown_fences(raw_text)
        graph = json.loads(json_text)
    except json.JSONDecodeError as exc:
        _log_failure(transcript_id, f"JSON decode error: {exc}")
        return None

    # Metadata — always use the true transcript_id
    graph["transcript_id"] = transcript_id
    graph["split"] = split
    graph["extraction_model"] = backend["model"]
    graph["prompt_version"] = prompt_version

    # Validate
    violations = validate_graph(graph)
    graph["validation_violations"] = violations
    if violations:
        for v in violations:
            print(f"  [{transcript_id}] VIOLATION: {v}")

    # Cache
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    return graph


# ── batch helpers ────────────────────────────────────────────────────


def load_tagged_transcripts() -> dict[str, dict]:
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
    backend: dict,
    prompt_version: str = "v4",
    domain: str = "AI's role in professional work",
    out_dir: Path | None = None,
    anthropic_client: Anthropic | None = None,
) -> tuple[int, int]:
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
            backend=backend,
            prompt_version=prompt_version,
            split=rec["split"],
            domain=domain,
            out_dir=out_dir,
            anthropic_client=anthropic_client,
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
        "--backend",
        type=str,
        default="deepseek",
        choices=["deepseek", "deepseek-think", "anthropic", "agnes"],
        help="Backend to use (default: deepseek)",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v4",
        help="Prompt version (default: v4)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="AI's role in professional work",
        help="Domain description for topic-neutral ontology "
        "(default: AI's role in professional work)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for graphs (default: version-namespaced, "
        "e.g. s1_data/graphs/v4/free_text; v3 keeps the legacy flat path)",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="sorted",
        choices=["sorted", "stratified"],
        help="Batch ordering: 'sorted' (alphabetical) or 'stratified' "
        "(round-robin across cohorts, so a --limit checkpoint samples all splits)",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract even if cache exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _graph_dir_for(args.prompt_version)
    print(f"Output dir: {out_dir}")

    backend = BACKENDS[args.backend]
    api_key = os.environ.get(backend["api_key_env"])
    if not api_key:
        print(f"ERROR: {backend['api_key_env']} not set in environment or .env")
        sys.exit(1)

    # Create Anthropic client only if needed
    anthropic_client = None
    if backend["type"] == "anthropic":
        anthropic_client = Anthropic(api_key=api_key, base_url=backend["base_url"])

    tagged = load_tagged_transcripts()

    # ── single transcript mode ─────────────────────────────────────
    if args.tid:
        if args.tid not in tagged:
            print(f"ERROR: transcript '{args.tid}' not found in tagged data")
            sys.exit(1)
        rec = tagged[args.tid]
        if args.force:
            (out_dir / f"{args.tid}.json").unlink(missing_ok=True)
        result = extract_one(
            transcript_id=args.tid,
            formatted_transcript=rec["formatted"],
            backend=backend,
            prompt_version=args.prompt_version,
            split=rec["split"],
            domain=args.domain,
            out_dir=out_dir,
            anthropic_client=anthropic_client,
        )
        if result is None:
            print("Extraction failed.")
            sys.exit(1)
        print(
            f"✓ {args.tid}: {len(result.get('nodes', []))} nodes, "
            f"{len(result.get('edges', []))} edges"
        )
        return

    # ── batch mode ─────────────────────────────────────────────────
    ids: list[str] = []
    if args.split:
        ids = [tid for tid, rec in tagged.items() if rec["split"] == args.split]
    elif args.order == "stratified":
        # round-robin interleave across cohorts so a --limit checkpoint samples
        # all splits proportionally-early (deterministic → resumable)
        from itertools import zip_longest

        by_split: dict[str, list[str]] = {}
        for tid in sorted(tagged.keys()):
            by_split.setdefault(tagged[tid]["split"], []).append(tid)
        ids = [tid for group in zip_longest(*by_split.values()) for tid in group if tid is not None]
    else:
        ids = sorted(tagged.keys())

    if not args.force:
        uncached = [tid for tid in ids if not (out_dir / f"{tid}.json").exists()]
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
        f"Processing {len(ids)} transcripts with "
        f"backend={args.backend} model={backend['model']} prompt={args.prompt_version}"
    )
    success, failure = extract_batch(
        transcript_ids=ids,
        tagged=tagged,
        backend=backend,
        prompt_version=args.prompt_version,
        domain=args.domain,
        out_dir=out_dir,
        anthropic_client=anthropic_client,
    )
    print(f"\nDone. {success} succeeded, {failure} failed.")


if __name__ == "__main__":
    main()
