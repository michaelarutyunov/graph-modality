"""Label stance_ambivalence on all transcripts using a non-DeepSeek model.

Two backends, neither is the DeepSeek graph extractor (breaks circularity):
  - agnes  : OpenAI-compatible, no JSON mode, free        -> agnes-2.0-flash
  - haiku  : Anthropic SDK                                  -> claude-haiku-4-5

Writes one line per transcript to cache/ambivalence_{model_tag}.jsonl.
Cache-first: re-running skips already-labeled transcripts.

Usage:
    PYTHONPATH=. uv run python s2_extraction/ambivalence_labeler.py --backend agnes
    PYTHONPATH=. uv run python s2_extraction/ambivalence_labeler.py --backend haiku
    PYTHONPATH=. uv run python s2_extraction/ambivalence_labeler.py --backend haiku --limit 3
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

PROMPT_PATH = Path("s2_extraction/prompts/ambivalence_v1.txt")
TAGGED_DIR = Path("s1_data/tagged")
CACHE_DIR = Path("cache")

BACKENDS = {
    "agnes": {
        "type": "openai",
        "model": "agnes-2.0-flash",
        "model_tag": "agnes",
        "api_key_env": "AGNES_API_KEY",
        "base_url": "https://apihub.agnes-ai.com/v1/chat/completions",
        "json_mode": False,
    },
    "haiku": {
        "type": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "model_tag": "haiku",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}

MAX_TOKENS = 1024
API_TIMEOUT = 120
MAX_RETRIES = 4
RETRY_DELAYS = [2, 8, 32, 64]
RATE_LIMIT = 0.8

VALID_LABELS = {"low", "med", "high", "uncertain"}


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: text.rstrip().rfind("```")]
    return text.strip()


def _validate_ambivalence(extraction: dict, tid: str) -> list[str]:
    warnings: list[str] = []
    obj = extraction.get("stance_ambivalence")
    if obj is None:
        return [f"{tid}: missing 'stance_ambivalence' field"]
    label = obj.get("label", "")
    if label not in VALID_LABELS:
        warnings.append(f"{tid}: invalid label '{label}'")
    if label != "uncertain" and not obj.get("quotes"):
        warnings.append(f"{tid}: label={label} but no quotes provided")
    if not obj.get("reasoning"):
        warnings.append(f"{tid}: missing reasoning")
    return warnings


def _build_human_text(record: dict) -> str:
    turns = [f"[Human]: {t['text']}" for t in record["turns"] if t["speaker"] == "Human"]
    return "\n\n".join(turns)


def _load_all_records() -> list[dict]:
    records: list[dict] = []
    for path in sorted(TAGGED_DIR.glob("*.jsonl")):
        if path.name == ".gitkeep":
            continue
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                records.append(json.loads(line))
    return records


def _load_cache(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    cached: dict[str, dict] = {}
    for line in output_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("transcript_id"):
                cached[entry["transcript_id"]] = entry
        except json.JSONDecodeError:
            continue
    return cached


def _call_openai(system_prompt: str, user_message: str, backend: dict) -> dict:
    api_key = os.environ.get(backend["api_key_env"], "")
    if not api_key:
        raise ValueError(f"API key env var {backend['api_key_env']} not set")
    messages = [
        {"role": "system", "content": system_prompt + "\n\nOutput valid JSON only."},
        {"role": "user", "content": user_message},
    ]
    body = {
        "model": backend["model"],
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                backend["base_url"],
                data=body_bytes,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                content = raw["choices"][0]["message"]["content"]
                return json.loads(_strip_fences(content))
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            last_error = e
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")


def _call_anthropic(system_prompt: str, user_message: str, backend: dict) -> dict:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ[backend["api_key_env"]])
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=backend["model"],
                max_tokens=MAX_TOKENS,
                system=system_prompt + "\n\nOutput valid JSON only. No markdown fences.",
                messages=[{"role": "user", "content": user_message}],
                timeout=API_TIMEOUT,
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return json.loads(_strip_fences(text))
        except Exception as e:
            last_error = e
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label stance_ambivalence")
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    backend = BACKENDS[args.backend]
    output_path = CACHE_DIR / f"ambivalence_{backend['model_tag']}.jsonl"
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    records = _load_all_records()
    if args.limit:
        records = records[: args.limit]

    cache = _load_cache(output_path)
    pending = [r for r in records if r["transcript_id"] not in cache]
    print(
        f"Backend: {backend['model']}  total={len(records)}  "
        f"cached={len(records) - len(pending)}  todo={len(pending)}"
    )

    caller = _call_openai if backend["type"] == "openai" else _call_anthropic
    warnings: list[str] = []
    for i, record in enumerate(pending, 1):
        tid = record["transcript_id"]
        human_text = _build_human_text(record)
        if len(human_text) > 8000:
            human_text = human_text[:8000] + "\n\n[... truncated]"
        user_message = f"Transcript ID: {tid}\n\n{human_text}"
        print(f"[{i}/{len(pending)}] {tid}")
        try:
            extraction = caller(prompt, user_message, backend)
            extraction["transcript_id"] = tid
            extraction["_model"] = backend["model"]
            warns = _validate_ambivalence(extraction, tid)
            warnings.extend(warns)
            for w in warns:
                print(f"  ⚠ {w}")
        except Exception as e:
            print(f"  ❌ {e}")
            extraction = {
                "transcript_id": tid,
                "stance_ambivalence": {"label": "error", "reasoning": str(e), "quotes": []},
                "_model": backend["model"],
                "_error": str(e),
            }
            warnings.append(f"{tid}: FAILED — {e}")
        cache[tid] = extraction
        with open(output_path, "w", encoding="utf-8") as f:
            for cid in sorted(cache):
                json.dump(cache[cid], f, ensure_ascii=False)
                f.write("\n")
        if i < len(pending):
            time.sleep(RATE_LIMIT)

    print(f"\nDone. Output: {output_path}  warnings={len(warnings)}")


if __name__ == "__main__":
    main()
