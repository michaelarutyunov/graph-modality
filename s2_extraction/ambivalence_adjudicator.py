"""Adjudicate Agnes/Haiku stance_ambivalence disagreements via Kimi.

Reads the two cached label files, identifies transcripts where the labels
conflict, and asks Kimi (kimi-k2.6) to judge which annotation better fits the
ambivalence_v1 rubric. The judge sees the human transcript and both
annotators' labels, reasonings, and quotes; annotators are anonymized (A/B)
and their order is randomized per transcript.

Outputs:
    cache/ambivalence_adjudications.json
        {transcript_id: chosen_label} for resolved disagreements.
        Labels may be low, med, high, uncertain, or manual_review.
    cache/ambivalence_adjudication_details.jsonl
        One JSON object per adjudicated transcript, including the chosen
        annotator, Kimi's reasoning, and supporting quotes for auditability.

Usage:
    PYTHONPATH=. uv run python s2_extraction/ambivalence_adjudicator.py
    PYTHONPATH=. uv run python s2_extraction/ambivalence_adjudicator.py --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from s2_extraction.ambivalence_labeler import (
    API_TIMEOUT,
    CACHE_DIR,
    MAX_RETRIES,
    MAX_TOKENS,
    RATE_LIMIT,
    RETRY_DELAYS,
    TAGGED_DIR,
    VALID_LABELS,
    _build_human_text,
    _strip_fences,
)

load_dotenv(override=True)

BASE_URL = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1/chat/completions")
MODEL = "kimi-k2.6"
API_KEY_ENV = "KIMI_API_KEY"

ADJUDICATION_PATH = CACHE_DIR / "ambivalence_adjudications.json"
DETAILS_PATH = CACHE_DIR / "ambivalence_adjudication_details.jsonl"
DISAGREEMENT_PATH = CACHE_DIR / "ambivalence_disagreements.json"

VALID_JUDGE_LABELS = VALID_LABELS | {"manual_review"}


def _load_jsonl(path: Path) -> dict[str, dict]:
    """Load a jsonl file as a transcript_id -> record dict."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("transcript_id"):
            out[entry["transcript_id"]] = entry
    return out


def _load_tagged_records() -> dict[str, dict]:
    """Load all tagged transcripts keyed by transcript_id."""
    out: dict[str, dict] = {}
    for path in sorted(TAGGED_DIR.glob("*.jsonl")):
        if path.name == ".gitkeep":
            continue
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            record = json.loads(line)
            tid = record.get("transcript_id")
            if tid:
                out[tid] = record
    return out


def _compute_disagreements(
    a: dict[str, dict], b: dict[str, dict]
) -> list[tuple[str, str, str]]:
    """Return list of (tid, label_a, label_b) for transcripts where labels differ."""
    common = sorted(set(a) & set(b))
    disagreements: list[tuple[str, str, str]] = []
    for tid in common:
        la = a[tid]["stance_ambivalence"]["label"]
        lb = b[tid]["stance_ambivalence"]["label"]
        if la != lb:
            disagreements.append((tid, la, lb))
    return disagreements


def _load_existing_details(path: Path) -> dict[str, dict]:
    """Load previous adjudication details keyed by transcript_id."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = entry.get("transcript_id")
        if tid:
            out[tid] = entry
    return out


def _load_existing_adjudications(path: Path) -> dict[str, str]:
    """Load existing adjudication labels keyed by transcript_id."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _annotate_label(ann: dict) -> str:
    """Format a single annotator's label, reasoning, and quotes for the judge."""
    obj = ann["stance_ambivalence"]
    label = obj.get("label", "?")
    reasoning = obj.get("reasoning", "")
    quotes = obj.get("quotes", [])
    quotes_text = "\n".join(f"  - {q}" for q in quotes) if quotes else "  (none provided)"
    return (
        f"Label: {label}\n"
        f"Reasoning: {reasoning}\n"
        f"Supporting quotes:\n{quotes_text}"
    )


def _build_judge_prompt(
    rubric: str,
    record: dict,
    ann_a: dict,
    ann_b: dict,
) -> tuple[str, str, tuple[str, str]]:
    """Return (system_prompt, user_message, order) for a Kimi adjudication call.

    order is a tuple (model_for_a, model_for_b) so the caller can map the
    chosen annotator back to the actual model.
    """
    tid = record["transcript_id"]
    human_text = _build_human_text(record)

    # Randomize A/B order per transcript to avoid position bias.
    if random.random() < 0.5:
        first, second = ann_a, ann_b
    else:
        first, second = ann_b, ann_a
    order = (first["_model"], second["_model"])

    system_prompt = (
        "You are an expert judge reviewing annotations of attitudinal ambivalence "
        "toward AI in interview transcripts.\n\n"
        f"{rubric}\n\n"
        "You will be shown the interviewee's responses and two anonymized annotator "
        "judgments (Annotator A and Annotator B). Each annotator provides a label, "
        "reasoning, and supporting quotes. Your job is to decide which label better "
        "matches the rubric, not which annotator is more articulate. If neither label "
        "is defensible, respond with 'uncertain'. If the case is genuinely "
        "unresolvable from the text, respond with 'manual_review'.\n\n"
        "Respond with a single JSON object and no other text. The JSON must follow "
        "this exact schema:\n"
        '{\n'
        '  "chosen_label": "low|med|high|uncertain|manual_review",\n'
        '  "chosen_annotator": "A|B|none",\n'
        '  "reasoning": "Explain your choice step by step, citing specific quotes '
        'from the transcript.",\n'
        '  "supporting_quotes": ["quote 1", "quote 2"]\n'
        '}\n\n'
        'Use "chosen_annotator": "none" when you select "uncertain" or "manual_review".'
    )

    user_message = (
        f"Transcript ID: {tid}\n\n"
        "[Human interviewee responses only]:\n"
        f"{human_text}\n\n"
        "---\n\n"
        "Annotator A:\n"
        f"{_annotate_label(first)}\n\n"
        "Annotator B:\n"
        f"{_annotate_label(second)}\n\n"
        "Which label (A or B) better fits the rubric? Output only the JSON object."
    )

    return system_prompt, user_message, order


def _call_kimi(system_prompt: str, user_message: str, dry_run: bool = False) -> dict:
    """Call the Kimi chat completions endpoint and return parsed JSON."""
    if dry_run:
        return {
            "chosen_label": "manual_review",
            "chosen_annotator": "none",
            "reasoning": "dry-run mode: no API call made",
            "supporting_quotes": [],
        }

    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise ValueError(f"API key env var {API_KEY_ENV} not set")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    body = {
        "model": MODEL,
        "messages": messages,
        # Kimi k2.6 requires temperature=0.6 when thinking is disabled.
        "temperature": 0.6,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
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
                return json.loads(_strip_fences(content))
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            last_error = e
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")


def _parse_judge_output(
    raw: dict, tid: str, order: tuple[str, str]
) -> dict:
    """Validate and normalize a Kimi judge response.

    order maps (model_for_A, model_for_B).
    """
    label = raw.get("chosen_label", "")
    if label not in VALID_JUDGE_LABELS:
        label = "manual_review"

    chosen = raw.get("chosen_annotator", "none")
    if chosen not in {"A", "B", "none"}:
        chosen = "none"

    model_chosen: str | None = None
    if chosen == "A":
        model_chosen = order[0]
    elif chosen == "B":
        model_chosen = order[1]

    return {
        "transcript_id": tid,
        "chosen_label": label,
        "chosen_annotator": chosen,
        "chosen_model": model_chosen,
        "reasoning": raw.get("reasoning", ""),
        "supporting_quotes": raw.get("supporting_quotes", []) or [],
        "adjudication_model": MODEL,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adjudicate stance_ambivalence disagreements with Kimi"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only adjudicate N disagreements"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts but do not call the Kimi API",
    )
    args = parser.parse_args()

    a = _load_jsonl(CACHE_DIR / "ambivalence_agnes.jsonl")
    b = _load_jsonl(CACHE_DIR / "ambivalence_haiku.jsonl")
    records = _load_tagged_records()
    disagreements = _compute_disagreements(a, b)

    existing_details = _load_existing_details(DETAILS_PATH)
    adjudications = _load_existing_adjudications(ADJUDICATION_PATH)

    rubric = (Path(__file__).parent / "prompts" / "ambivalence_v1.txt").read_text(
        encoding="utf-8"
    )

    pending = [
        (tid, la, lb)
        for tid, la, lb in disagreements
        if tid not in existing_details and tid not in adjudications
    ]
    if args.limit:
        pending = pending[: args.limit]

    print(
        f"Disagreements: {len(disagreements)} | "
        f"already adjudicated: {len(existing_details)} | "
        f"pending: {len(pending)}"
    )
    if not pending:
        print("Nothing to adjudicate.")
        return

    # Persist disagreement worklist for reference.
    DISAGREEMENT_PATH.write_text(
        json.dumps(
            [
                {"transcript_id": tid, "agnes": la, "haiku": lb}
                for tid, la, lb in disagreements
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    new_adjudications: dict[str, str] = {}
    new_details: list[dict] = []
    errors: list[str] = []

    for i, (tid, _, _) in enumerate(pending, 1):
        record = records.get(tid)
        if record is None:
            errors.append(f"{tid}: missing tagged record")
            continue

        ann_a = a.get(tid)
        ann_b = b.get(tid)
        if ann_a is None or ann_b is None:
            errors.append(f"{tid}: missing annotation")
            continue

        system_prompt, user_message, order = _build_judge_prompt(
            rubric, record, ann_a, ann_b
        )

        print(f"[{i}/{len(pending)}] {tid} -> Kimi", flush=True)
        try:
            raw = _call_kimi(system_prompt, user_message, dry_run=args.dry_run)
            detail = _parse_judge_output(raw, tid, order)
        except Exception as e:
            print(f"  ❌ {e}")
            errors.append(f"{tid}: {e}")
            detail = {
                "transcript_id": tid,
                "chosen_label": "manual_review",
                "chosen_annotator": "none",
                "chosen_model": None,
                "reasoning": f"API/parse error: {e}",
                "supporting_quotes": [],
                "adjudication_model": MODEL,
            }

        new_details.append(detail)
        adjudications[tid] = detail["chosen_label"]
        new_adjudications[tid] = detail["chosen_label"]

        # Append detail immediately for crash-resume.
        DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DETAILS_PATH, "a", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False)
            f.write("\n")

        # Persist adjudications after every call.
        ADJUDICATION_PATH.write_text(
            json.dumps(adjudications, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if i < len(pending):
            time.sleep(RATE_LIMIT)

    print(
        f"\nDone. New adjudications: {len(new_adjudications)} | "
        f"errors: {len(errors)}"
    )
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
