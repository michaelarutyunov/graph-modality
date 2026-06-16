"""Label the human-AI delegation boundary on the full corpus (selfpos_v1, multi-label).

Cache-first, one backend at a time (run Agnes and Haiku as separate processes). Neither is the
DeepSeek graph extractor (breaks circularity). Writes cache/selfpos_{tag}.jsonl, one line per
transcript. Re-running skips already-labeled transcripts.

    PYTHONPATH=. uv run python s2_extraction/selfpos_labeler.py --backend agnes
    PYTHONPATH=. uv run python s2_extraction/selfpos_labeler.py --backend haiku
    PYTHONPATH=. uv run python s2_extraction/selfpos_labeler.py --backend haiku --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=str(Path(__file__).resolve().parents[1] / ".env"), override=True)

PROMPT_PATH = Path("s2_extraction/prompts/selfpos_v1.txt")
TAGGED_DIR = Path("s1_data/tagged")
CACHE_DIR = Path("cache")

BACKENDS = {
    "agnes": {
        "type": "openai",
        "model": "agnes-2.0-flash",
        "tag": "agnes",
        "key": "AGNES_API_KEY",
        "url": "https://apihub.agnes-ai.com/v1/chat/completions",
        "temperature": 0.0,
    },
    "haiku": {
        "type": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "tag": "haiku",
        "key": "ANTHROPIC_API_KEY",
        "temperature": 0.0,
    },
}

VALID_RATIONALES = {
    "competence",
    "competence_compensate",
    "identity",
    "trust_reliability",
    "output_efficiency",
    "other",
}
VALID_BREADTH = {"low", "medium", "high"}
MAX_TOKENS = 1800
TIMEOUT = 60
RETRIES = 3
DELAYS = [2, 6, 15]
RATE_LIMIT = 0.5


def strip_fences(t: str) -> str:
    t = t.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: t.rstrip().rfind("```")]
    return t.strip()


def parse_json(content: str) -> dict:
    try:
        return json.loads(strip_fences(content))
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def call_openai(prompt: str, b: dict) -> dict:
    body = json.dumps(
        {
            "model": b["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful qualitative coder. Output valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": b["temperature"],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                b["url"],
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ[b['key']]}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return parse_json(
                    json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
                )
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            last = e
            time.sleep(DELAYS[min(attempt, len(DELAYS) - 1)])
    raise RuntimeError(f"openai failed: {last}")


def call_anthropic(prompt: str, b: dict) -> dict:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ[b["key"]])
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = client.messages.create(
                model=b["model"],
                max_tokens=MAX_TOKENS,
                temperature=b["temperature"],
                system="You are a careful qualitative coder. Output valid JSON only. No markdown fences.",
                messages=[{"role": "user", "content": prompt}],
                timeout=TIMEOUT,
            )
            return parse_json("".join(x.text for x in resp.content if x.type == "text"))
        except Exception as e:
            last = e
            time.sleep(DELAYS[min(attempt, len(DELAYS) - 1)])
    raise RuntimeError(f"anthropic failed: {last}")


def human_text(record: dict) -> str:
    turns = [t["text"] for t in record["turns"] if t["speaker"] == "Human"]
    txt = "\n\n".join(turns)
    return txt[:8000] + "\n\n[... truncated]" if len(txt) > 8000 else txt


def load_records() -> list[dict]:
    recs: list[dict] = []
    for path in sorted(TAGGED_DIR.glob("*.jsonl")):
        if path.name == ".gitkeep":
            continue
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                recs.append(json.loads(line))
    return recs


def load_cache(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            try:
                e = json.loads(line)
                if e.get("transcript_id"):
                    out[e["transcript_id"]] = e
            except json.JSONDecodeError:
                continue
    return out


def validate(rec: dict, tid: str) -> list[str]:
    w = []
    rp = rec.get("rationales_present")
    if rp is None:
        w.append(f"{tid}: missing rationales_present")
    elif any(r not in VALID_RATIONALES for r in rp):
        w.append(f"{tid}: invalid rationale in {rp}")
    if rec.get("delegation_breadth") not in VALID_BREADTH:
        w.append(f"{tid}: invalid delegation_breadth {rec.get('delegation_breadth')}")
    if rec.get("boundary_talk_depth") not in (1, 2, 3):
        w.append(f"{tid}: invalid boundary_talk_depth {rec.get('boundary_talk_depth')}")
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    b = BACKENDS[args.backend]
    out_path = CACHE_DIR / f"selfpos_{b['tag']}.jsonl"
    CACHE_DIR.mkdir(exist_ok=True)
    tpl = PROMPT_PATH.read_text(encoding="utf-8")
    records = load_records()
    if args.limit:
        records = records[: args.limit]
    cache = load_cache(out_path)
    pending = [r for r in records if r["transcript_id"] not in cache]
    print(
        f"backend={b['model']} total={len(records)} cached={len(records) - len(pending)} todo={len(pending)}",
        flush=True,
    )
    caller = call_openai if b["type"] == "openai" else call_anthropic

    n_fail, warns = 0, 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(pending, 1):
            tid = rec["transcript_id"]
            prompt = tpl.replace("{transcript}", human_text(rec))
            try:
                out = caller(prompt, b)
                out["transcript_id"] = tid
                out["_model"] = b["model"]
                for wmsg in validate(out, tid):
                    warns += 1
                    print("  WARN", wmsg, flush=True)
            except Exception as e:
                out = {"transcript_id": tid, "_model": b["model"], "_error": str(e)}
                n_fail += 1
                print(f"[{i}/{len(pending)}] {tid} FAIL: {e}", flush=True)
            else:
                if i % 25 == 0 or i == len(pending):
                    print(f"[{i}/{len(pending)}] {tid} ok", flush=True)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(RATE_LIMIT)
    print(f"DONE backend={b['tag']} failures={n_fail} warnings={warns} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
