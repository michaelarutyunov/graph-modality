"""B7 adjudication: Kimi rules the 208 delegation_breadth disputes (Agnes vs Haiku).

Uses the PROVEN Kimi k2.6 config from ambivalence_adjudicator (thinking DISABLED + temperature
0.6 + JSON mode) -- fast and reliable, unlike thinking-on which times out. For each dispute the
two candidate breadth ratings are shown anonymized + randomized (A/B) so the judge can't anchor
on a coder; Kimi reads the respondent's own words and picks which rating fits. Cache-first.

    PYTHONPATH=. uv run python s2_extraction/selfpos_adjudicator.py
    PYTHONPATH=. uv run python s2_extraction/selfpos_adjudicator.py --limit 3

Output: cache/selfpos_adjudications.jsonl (one line per dispute).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=str(Path(__file__).resolve().parents[1] / ".env"), override=True)

WORKLIST = Path("results/method_review/selfpos_consensus/worklist.jsonl")
TAGGED = Path("s1_data/tagged")
OUT = Path("cache/selfpos_adjudications.jsonl")
BASE_URL = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1/chat/completions")
MODEL = "kimi-k2.6"
API_KEY_ENV = "KIMI_API_KEY"
MAX_TOKENS = 1024
TIMEOUT = 120
RETRIES = 4
DELAYS = [3, 10, 30, 60]
RATE_LIMIT = 0.4

BREADTH_DEF = (
    "DELEGATION BREADTH = how much of their professional work the respondent cedes to AI overall:\n"
    "- low: they keep most things; AI is a narrow/occasional helper.\n"
    "- medium: a real mix of retained and ceded work.\n"
    "- high: they hand a large share of their work to AI."
)
SYSTEM = (
    "You are an expert qualitative adjudicator. Two coders assigned DIFFERENT delegation-breadth "
    "ratings to the same interview respondent. " + BREADTH_DEF + "\n\n"
    "Read ONLY the respondent's own words and decide which candidate rating fits better. "
    'Output strict JSON: {"chosen": "A" or "B", "reasoning": "<one or two sentences>"}.'
)


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


def call_kimi(user_message: str) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.6,  # required when thinking disabled
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                BASE_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ[API_KEY_ENV]}",
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
    raise RuntimeError(f"kimi failed: {last}")


def human_text(record: dict) -> str:
    turns = [t["text"] for t in record["turns"] if t["speaker"] == "Human"]
    txt = "\n\n".join(turns)
    return txt[:8000] + "\n\n[... truncated]" if len(txt) > 8000 else txt


def load_tagged() -> dict[str, dict]:
    recs = {}
    for p in sorted(TAGGED.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                r = json.loads(line)
                recs[r["transcript_id"]] = r
    return recs


def load_done() -> set[str]:
    if not OUT.exists():
        return set()
    return {
        json.loads(l)["transcript_id"]
        for l in OUT.read_text().strip().split("\n")
        if l and "transcript_id" in json.loads(l)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tagged = load_tagged()
    worklist = [json.loads(l) for l in WORKLIST.read_text().strip().split("\n") if l]
    worklist = [
        w for w in worklist if w["breadth_agnes"] != w["breadth_haiku"]
    ]  # breadth disputes only
    done = load_done()
    pending = [w for w in worklist if w["transcript_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"breadth disputes={len(worklist)} done={len(done)} todo={len(pending)}", flush=True)

    n_fail = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for i, w in enumerate(pending, 1):
            tid = w["transcript_id"]
            rng = random.Random(tid)  # deterministic A/B per transcript
            cand = {"agnes": w["breadth_agnes"], "haiku": w["breadth_haiku"]}
            order = ["agnes", "haiku"]
            rng.shuffle(order)
            label_for = {"A": cand[order[0]], "B": cand[order[1]]}  # A/B -> breadth value
            coder_for = {"A": order[0], "B": order[1]}
            user = (
                f"Candidate A breadth rating: {label_for['A']}\n"
                f"Candidate B breadth rating: {label_for['B']}\n\n"
                f"Respondent's words:\n{human_text(tagged[tid])}"
            )
            try:
                out = call_kimi(user)
                chosen = str(out.get("chosen", "")).strip().upper()
                if chosen not in ("A", "B"):
                    raise ValueError(f"bad chosen={chosen!r}")
                rec = {
                    "transcript_id": tid,
                    "cohort": w.get("cohort"),
                    "breadth_agnes": w["breadth_agnes"],
                    "breadth_haiku": w["breadth_haiku"],
                    "adjudicated_breadth": label_for[chosen],
                    "chosen_coder": coder_for[chosen],
                    "reasoning": out.get("reasoning", ""),
                    "_model": MODEL,
                }
            except Exception as e:
                rec = {"transcript_id": tid, "_error": str(e)}
                n_fail += 1
                print(f"[{i}/{len(pending)}] {tid} FAIL: {e}", flush=True)
            else:
                if i % 25 == 0 or i == len(pending):
                    print(f"[{i}/{len(pending)}] {tid} -> {rec['adjudicated_breadth']}", flush=True)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(RATE_LIMIT)
    print(f"DONE failures={n_fail} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
