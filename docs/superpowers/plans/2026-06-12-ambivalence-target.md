# Ambivalence Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an endogenous, lexically-non-obvious ordinal target `stance_ambivalence ∈ {low, med, high}`, labeled independently of the DeepSeek graph extractor (by Agnes + Haiku with user-adjudicated disagreements), and re-run the null-ladder / structure-only / text probes on it.

**Architecture:** A standalone dual-backend labeler reads raw human turns and emits ordinal ambivalence labels from two non-DeepSeek models. A consensus step computes agreement/κ, accepts agreed labels, and produces a disagreement worklist for manual adjudication, writing a single `cache/ambivalence.jsonl`. The new target then plugs into the existing frozen-embedding harness through the same seams `ai_adoption` uses (`_load_*_labels`, `make_split`, `slice_modalities`, the two probes).

**Tech Stack:** Python 3.11, `uv`, stdlib `urllib` (OpenAI-compatible Agnes), Anthropic SDK (Haiku), scikit-learn (`cohen_kappa_score`, `f1_score`), PyTorch + PyG (existing probes), pytest.

**Spec:** `.claude/context/ambivalence-target.md`

**Conventions:** All commands prefixed `PYTHONPATH=. uv run`. Tests set `BEADS_DB` to a temp path if they touch beads (these don't). Results written to new files, never overwriting.

---

### Task 1: Ambivalence prompt file

**Files:**
- Create: `s2_extraction/prompts/ambivalence_v1.txt`
- Test: `tests/test_ambivalence_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ambivalence_prompt.py
from pathlib import Path

PROMPT = Path("s2_extraction/prompts/ambivalence_v1.txt")


def test_prompt_exists_and_defines_ordinal_scheme():
    text = PROMPT.read_text(encoding="utf-8")
    # three ordinal levels present
    for level in ('"low"', '"med"', '"high"'):
        assert level in text, f"missing level {level}"
    # the JSON output key the labeler/loader depend on
    assert "stance_ambivalence" in text
    # holistic-judgment guardrail: must forbid counting graph features
    assert "do not count" in text.lower() or "not by counting" in text.lower()
    # evidence requirement
    assert "quote" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_prompt.py -v`
Expected: FAIL — `FileNotFoundError` (prompt file does not exist).

- [ ] **Step 3: Create the prompt file**

```text
You are an annotator reading an interview transcript about how a person uses AI in their professional work. You will read ONLY the interviewee's responses (labeled [Human]). The interviewer's questions are NOT shown.

Your job: judge the person's ATTITUDINAL AMBIVALENCE toward AI — the degree of UNRESOLVED TENSION in their overall stance. This is a holistic judgment about the person, formed from their own words. Do NOT count anything; weigh how their positive and negative views relate.

Almost everyone mentions both upsides and downsides of AI. The question is NOT whether both appear, but whether the person has RESOLVED them into a coherent stance or HOLDS THEM IN TENSION.

## Levels

- **low** — Coherent one-sided stance. Clearly pro-AI OR clearly skeptical, with little internal tension. Downsides (if mentioned) are minor caveats that do not unsettle the overall position. Evidence: "AI has completely changed how I work and I rely on it daily" (with no real reservation); or "I just don't trust these tools for my work" (with no real enthusiasm).

- **med** — Leans one way but genuinely acknowledges the other side. A clear net position exists, but the person gives real weight to the opposing considerations. Evidence: "I use it a lot and it helps, though I'm careful because it gets things wrong in ways that matter."

- **high** — Genuinely torn. Substantial pro AND con held in unresolved tension; the person oscillates, qualifies heavily, or explicitly cannot settle. Evidence: "I love what it lets me do but I'm honestly conflicted — some days it feels essential, other days I worry it's eroding the craft."

- **uncertain** — Not enough about the person's attitude to judge tension at all. Use sparingly.

## Rules

1. Evidence-first: every non-uncertain label MUST include at least one direct quote from a [Human] turn. Quote exact words.
2. Judge tension, not sentiment. A strongly negative but COHERENT view is **low**, not high. Ambivalence is about conflict between views, not about how positive or negative they are.
3. Do not infer from profession, vocabulary sophistication, or topic. Use only what the person says about AI.
4. Reasoning is mandatory: explain step by step, citing the quotes.

## Output format

Respond with a single JSON object. No other text, no markdown fences.

{
  "transcript_id": "<the transcript ID provided>",
  "stance_ambivalence": {
    "label": "low|med|high|uncertain",
    "reasoning": "Step-by-step reasoning. If not uncertain, include at least one direct quote from [Human] turns.",
    "quotes": ["quote 1", "quote 2"]
  }
}

If a quote contains double quotes, escape them with backslash: \"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_prompt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s2_extraction/prompts/ambivalence_v1.txt tests/test_ambivalence_prompt.py
git commit -m "feat: ambivalence_v1 prompt — ordinal stance-tension rubric"
```

---

### Task 2: Dual-backend ambivalence labeler

Produces per-model labels into `cache/ambivalence_{model_tag}.jsonl`. Reuses the
`demographics_extractor.py` self-contained pattern but parametrized by backend, supporting
Agnes (OpenAI-compatible, no JSON mode) and Haiku (Anthropic SDK).

**Files:**
- Create: `s2_extraction/ambivalence_labeler.py`
- Test: `tests/test_ambivalence_labeler.py`

- [ ] **Step 1: Write the failing test (pure helpers only — no API calls)**

```python
# tests/test_ambivalence_labeler.py
from s2_extraction.ambivalence_labeler import (
    _validate_ambivalence,
    _strip_fences,
    _build_human_text,
)


def test_strip_fences_removes_markdown():
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


def test_build_human_text_keeps_only_human_turns():
    record = {
        "turns": [
            {"speaker": "AI", "text": "as a creative professional, tell me..."},
            {"speaker": "Human", "text": "I use AI daily."},
            {"speaker": "Human", "text": "But I worry about it."},
        ]
    }
    out = _build_human_text(record)
    assert "as a creative professional" not in out
    assert "[Human]: I use AI daily." in out
    assert "[Human]: But I worry about it." in out


def test_validate_flags_invalid_label_and_missing_quotes():
    bad = {"transcript_id": "t1", "stance_ambivalence": {"label": "extreme", "quotes": [], "reasoning": "x"}}
    warns = _validate_ambivalence(bad, "t1")
    assert any("invalid" in w for w in warns)

    no_quotes = {"transcript_id": "t1", "stance_ambivalence": {"label": "high", "quotes": [], "reasoning": "x"}}
    warns = _validate_ambivalence(no_quotes, "t1")
    assert any("no quotes" in w for w in warns)


def test_validate_passes_uncertain_without_quotes():
    ok = {"transcript_id": "t1", "stance_ambivalence": {"label": "uncertain", "quotes": [], "reasoning": "thin"}}
    assert _validate_ambivalence(ok, "t1") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_labeler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 's2_extraction.ambivalence_labeler'`.

- [ ] **Step 3: Write the labeler**

```python
# s2_extraction/ambivalence_labeler.py
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
    body = {"model": backend["model"], "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0.0}
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
        except Exception as e:  # noqa: BLE001 - retry on any API/parse error
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
    print(f"Backend: {backend['model']}  total={len(records)}  cached={len(records) - len(pending)}  todo={len(pending)}")

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
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {e}")
            extraction = {"transcript_id": tid, "stance_ambivalence": {"label": "error", "reasoning": str(e), "quotes": []}, "_model": backend["model"], "_error": str(e)}
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_labeler.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint, then commit**

```bash
ruff check s2_extraction/ambivalence_labeler.py tests/test_ambivalence_labeler.py --fix
ruff format s2_extraction/ambivalence_labeler.py tests/test_ambivalence_labeler.py
git add s2_extraction/ambivalence_labeler.py tests/test_ambivalence_labeler.py
git commit -m "feat: dual-backend (Agnes/Haiku) ambivalence labeler"
```

---

### Task 3: Consensus, Cohen's κ, and disagreement worklist

Merges the two per-model label files into agreement stats and a consensus file. Agreed labels are
accepted; disagreements are written to a worklist for the user to adjudicate, then merged back.

**Files:**
- Create: `s2_extraction/ambivalence_consensus.py`
- Test: `tests/test_ambivalence_consensus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ambivalence_consensus.py
from s2_extraction.ambivalence_consensus import (
    compute_agreement,
    merge_consensus,
)


def _entry(tid, label):
    return {"transcript_id": tid, "stance_ambivalence": {"label": label, "quotes": [], "reasoning": ""}}


def test_compute_agreement_rate_and_kappa():
    a = {"t1": _entry("t1", "low"), "t2": _entry("t2", "high"), "t3": _entry("t3", "med")}
    b = {"t1": _entry("t1", "low"), "t2": _entry("t2", "med"), "t3": _entry("t3", "med")}
    stats = compute_agreement(a, b)
    assert stats["n_common"] == 3
    assert stats["n_agree"] == 2
    assert abs(stats["agreement_rate"] - 2 / 3) < 1e-9
    assert "kappa" in stats
    assert stats["disagreements"] == ["t2"]


def test_merge_consensus_accepts_agreements_and_applies_adjudications():
    a = {"t1": _entry("t1", "low"), "t2": _entry("t2", "high")}
    b = {"t1": _entry("t1", "low"), "t2": _entry("t2", "med")}
    adjudications = {"t2": "high"}  # user resolved the disagreement
    final = merge_consensus(a, b, adjudications)
    assert final["t1"]["stance_ambivalence"]["label"] == "low"
    assert final["t1"]["stance_ambivalence"]["source"] == "consensus"
    assert final["t2"]["stance_ambivalence"]["label"] == "high"
    assert final["t2"]["stance_ambivalence"]["source"] == "adjudicated"


def test_merge_consensus_skips_unadjudicated_disagreements():
    a = {"t2": _entry("t2", "high")}
    b = {"t2": _entry("t2", "med")}
    final = merge_consensus(a, b, adjudications={})
    assert "t2" not in final  # unresolved disagreements are excluded, not guessed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_consensus.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write the consensus module**

```python
# s2_extraction/ambivalence_consensus.py
"""Merge two per-model ambivalence label files into a consensus label set.

Agreement cases are accepted directly. Disagreements are written to a worklist
(cache/ambivalence_disagreements.json) for manual adjudication; the user fills in
resolved labels in cache/ambivalence_adjudications.json, then this script merges
them into the final cache/ambivalence.jsonl.

Usage:
    # 1. after both labelers have run:
    PYTHONPATH=. uv run python s2_extraction/ambivalence_consensus.py --report
    # 2. user edits cache/ambivalence_adjudications.json (worklist -> labels)
    # 3. produce final consensus file:
    PYTHONPATH=. uv run python s2_extraction/ambivalence_consensus.py --finalize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

CACHE_DIR = Path("cache")
PATH_A = CACHE_DIR / "ambivalence_agnes.jsonl"
PATH_B = CACHE_DIR / "ambivalence_haiku.jsonl"
DISAGREE_PATH = CACHE_DIR / "ambivalence_disagreements.json"
ADJUDICATE_PATH = CACHE_DIR / "ambivalence_adjudications.json"
FINAL_PATH = CACHE_DIR / "ambivalence.jsonl"

# excluded from agreement/consensus — not part of the ordinal scale
NON_ORDINAL = {"uncertain", "error"}


def _label(entry: dict) -> str:
    return entry["stance_ambivalence"]["label"]


def _load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            e = json.loads(line)
            out[e["transcript_id"]] = e
    return out


def compute_agreement(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Agreement rate, Cohen's kappa, and the list of disagreeing ids.

    Only transcripts where BOTH models gave an ordinal label (low/med/high) count.
    """
    common = sorted(
        tid
        for tid in set(a) & set(b)
        if _label(a[tid]) not in NON_ORDINAL and _label(b[tid]) not in NON_ORDINAL
    )
    la = [_label(a[t]) for t in common]
    lb = [_label(b[t]) for t in common]
    n_agree = sum(1 for x, y in zip(la, lb) if x == y)
    disagreements = [t for t, x, y in zip(common, la, lb) if x != y]
    kappa = float(cohen_kappa_score(la, lb)) if len(common) > 1 else float("nan")
    return {
        "n_common": len(common),
        "n_agree": n_agree,
        "agreement_rate": n_agree / len(common) if common else float("nan"),
        "kappa": kappa,
        "disagreements": disagreements,
    }


def merge_consensus(
    a: dict[str, dict], b: dict[str, dict], adjudications: dict[str, str]
) -> dict[str, dict]:
    """Build the final label set.

    - both ordinal and equal      -> accept (source='consensus')
    - disagree but adjudicated     -> use adjudicated label (source='adjudicated')
    - disagree and not adjudicated -> EXCLUDE (never guessed)
    - any non-ordinal in either    -> EXCLUDE
    """
    final: dict[str, dict] = {}
    for tid in sorted(set(a) & set(b)):
        la, lb = _label(a[tid]), _label(b[tid])
        if la in NON_ORDINAL or lb in NON_ORDINAL:
            if tid in adjudications:
                final[tid] = {"transcript_id": tid, "stance_ambivalence": {"label": adjudications[tid], "source": "adjudicated"}}
            continue
        if la == lb:
            final[tid] = {"transcript_id": tid, "stance_ambivalence": {"label": la, "source": "consensus"}}
        elif tid in adjudications:
            final[tid] = {"transcript_id": tid, "stance_ambivalence": {"label": adjudications[tid], "source": "adjudicated"}}
        # else: unresolved disagreement -> excluded
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="compute agreement + write disagreement worklist")
    parser.add_argument("--finalize", action="store_true", help="merge adjudications into final cache/ambivalence.jsonl")
    args = parser.parse_args()

    a, b = _load_jsonl(PATH_A), _load_jsonl(PATH_B)

    if args.report:
        stats = compute_agreement(a, b)
        print(json.dumps({k: v for k, v in stats.items() if k != "disagreements"}, indent=2))
        print(f"disagreements: {len(stats['disagreements'])}")
        worklist = {
            tid: {
                "agnes": _label(a[tid]),
                "haiku": _label(b[tid]),
                "agnes_reasoning": a[tid]["stance_ambivalence"].get("reasoning", ""),
                "haiku_reasoning": b[tid]["stance_ambivalence"].get("reasoning", ""),
                "RESOLVE_TO": "",  # user fills: low|med|high
            }
            for tid in stats["disagreements"]
        }
        DISAGREE_PATH.write_text(json.dumps(worklist, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Worklist -> {DISAGREE_PATH}. Fill RESOLVE_TO, copy into {ADJUDICATE_PATH} as {{tid: label}}.")

    if args.finalize:
        adjudications = json.loads(ADJUDICATE_PATH.read_text(encoding="utf-8")) if ADJUDICATE_PATH.exists() else {}
        final = merge_consensus(a, b, adjudications)
        with open(FINAL_PATH, "w", encoding="utf-8") as f:
            for tid in sorted(final):
                json.dump(final[tid], f, ensure_ascii=False)
                f.write("\n")
        print(f"Final consensus labels: {len(final)} -> {FINAL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_consensus.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, then commit**

```bash
ruff check s2_extraction/ambivalence_consensus.py tests/test_ambivalence_consensus.py --fix
ruff format s2_extraction/ambivalence_consensus.py tests/test_ambivalence_consensus.py
git add s2_extraction/ambivalence_consensus.py tests/test_ambivalence_consensus.py
git commit -m "feat: ambivalence consensus + Cohen's kappa + disagreement worklist"
```

---

### Task 4: Label loader + split/slice wiring

Adds `stance_ambivalence` to the dataset seams so the existing harness can consume it.

**Files:**
- Modify: `s4_encoding/build_dataset.py` (add constants + `_load_ambivalence_labels`)
- Modify: `s5_classification/repeated_eval.py:39-44` and `:85-90` (target dispatch in `make_split` and `slice_modalities`)
- Test: `tests/test_ambivalence_labels.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ambivalence_labels.py
import json

import s4_encoding.build_dataset as bd


def test_load_ambivalence_labels_maps_ordinal_and_skips_nonordinal(tmp_path, monkeypatch):
    f = tmp_path / "ambivalence.jsonl"
    rows = [
        {"transcript_id": "t1", "stance_ambivalence": {"label": "low"}},
        {"transcript_id": "t2", "stance_ambivalence": {"label": "med"}},
        {"transcript_id": "t3", "stance_ambivalence": {"label": "high"}},
        {"transcript_id": "t4", "stance_ambivalence": {"label": "uncertain"}},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(bd, "AMBIVALENCE_PATH", f)

    labels = bd._load_ambivalence_labels()
    assert labels == {"t1": 0, "t2": 1, "t3": 2}  # uncertain skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_labels.py -v`
Expected: FAIL — `AttributeError: module 's4_encoding.build_dataset' has no attribute 'AMBIVALENCE_PATH'`.

- [ ] **Step 3a: Add constants + loader to `s4_encoding/build_dataset.py`**

Find the existing block (near line 34-39):

```python
DEMOGRAPHICS_PATH = CACHE_DIR / "demographics.jsonl"
...
AI_LABEL_MAP = {"tool_user": 0, "integrated": 1}
AI_EXCLUDED = {"novice", "power_user"}
```

Add immediately after it:

```python
AMBIVALENCE_PATH = CACHE_DIR / "ambivalence.jsonl"
AMBIVALENCE_LABEL_MAP = {"low": 0, "med": 1, "high": 2}
```

Then add this function next to `_load_ai_adoption_labels`:

```python
def _load_ambivalence_labels() -> dict[str, int]:
    """Load stance_ambivalence labels: low->0, med->1, high->2.

    Reads the consensus file (cache/ambivalence.jsonl). Non-ordinal labels
    (uncertain/error) are absent from the consensus file by construction, but
    we skip defensively. Returns transcript_id -> integer ordinal label.
    """
    if not AMBIVALENCE_PATH.exists():
        raise FileNotFoundError(
            f"Ambivalence consensus file not found at {AMBIVALENCE_PATH}. "
            "Run the labeler + consensus --finalize first."
        )
    labels: dict[str, int] = {}
    with open(AMBIVALENCE_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            raw = record["stance_ambivalence"]["label"]
            if raw in AMBIVALENCE_LABEL_MAP:
                labels[record["transcript_id"]] = AMBIVALENCE_LABEL_MAP[raw]
    print(f"Ambivalence labels: {len(labels)} transcripts")
    return labels
```

- [ ] **Step 3b: Wire target dispatch in `s5_classification/repeated_eval.py`**

At the top import block, extend the existing import:

```python
from s4_encoding.build_dataset import _load_ai_adoption_labels, _load_ambivalence_labels
```

In `make_split` (the `if target == ...` chain near line 39) add a branch before the `else`:

```python
    elif target == "stance_ambivalence":
        ids_to_labels = _load_ambivalence_labels()
```

In `slice_modalities` (the `if target == ...` chain near line 85) add the same branch before the `else`:

```python
    elif target == "stance_ambivalence":
        labels_dict = _load_ambivalence_labels()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_labels.py -v`
Expected: PASS.
Run regression: `PYTHONPATH=. uv run pytest tests/test_repeated_eval.py -v`
Expected: PASS (no behavior change for existing targets).

- [ ] **Step 5: Typecheck, lint, commit**

```bash
uv run pyright s4_encoding/build_dataset.py s5_classification/repeated_eval.py
ruff check s4_encoding/build_dataset.py s5_classification/repeated_eval.py tests/test_ambivalence_labels.py --fix
git add s4_encoding/build_dataset.py s5_classification/repeated_eval.py tests/test_ambivalence_labels.py
git commit -m "feat: wire stance_ambivalence target into label loader + split/slice"
```

---

### Task 5: Wire target into the two probes (+ dynamic chance baseline)

The 3-class ordinal chance baseline is unknown until labels exist, so compute it from the label
distribution instead of hardcoding (cohort/ai_adoption keep their existing constants).

**Files:**
- Modify: `s5_classification/null_ladder.py` (`_load_labels`, `_chance_baseline`, `_class_names`)
- Modify: `s5_classification/structure_only_probe.py` (`_load_labels`, `TARGETS`, `CHANCE`, chance use)
- Test: `tests/test_ambivalence_target_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ambivalence_target_wiring.py
from sklearn.metrics import f1_score

from s5_classification.null_ladder import _class_names
from s5_classification.structure_only_probe import majority_class_macro_f1


def test_class_names_for_ambivalence():
    assert _class_names("stance_ambivalence") == ["low", "med", "high"]


def test_majority_class_macro_f1_matches_sklearn():
    y = [0, 0, 0, 1, 2]  # majority class 0
    expected = f1_score(y, [0] * len(y), average="macro")
    assert abs(majority_class_macro_f1(y) - expected) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_target_wiring.py -v`
Expected: FAIL — `ImportError: cannot import name 'majority_class_macro_f1'` and `_class_names` raises on the new target.

- [ ] **Step 3a: Edit `s5_classification/structure_only_probe.py`**

Extend the constants near line 49-51:

```python
TARGETS = ["ai_adoption", "cohort", "stance_ambivalence"]

CHANCE = {"cohort": 0.2959, "ai_adoption": 0.3367}  # stance_ambivalence computed dynamically
```

Add the helper (top-level, after imports) and a label branch:

```python
def majority_class_macro_f1(y) -> float:
    """Macro-F1 of always predicting the most frequent class (chance baseline)."""
    from collections import Counter

    from sklearn.metrics import f1_score

    majority = Counter(int(v) for v in y).most_common(1)[0][0]
    return float(f1_score(list(y), [majority] * len(y), average="macro"))
```

In `_load_labels`, add before the `raise`:

```python
    elif target == "stance_ambivalence":
        from s4_encoding.build_dataset import _load_ambivalence_labels

        return _load_ambivalence_labels()
```

In `run_target`, replace the chance lookup `CHANCE[target]` with a resolved value. Find where
`delta = f1 - CHANCE[target]` (≈ line 169) and the summary `"chance": CHANCE[target]` (≈ line 187).
Just above the seed loop in `run_target` add:

```python
    chance = CHANCE.get(target)
    if chance is None:  # ordinal target without a precomputed constant
        chance = majority_class_macro_f1(list(ids_to_labels.values()))
```

Then use `chance` in place of `CHANCE[target]` in both the `delta = f1 - chance` line and the
`"chance": chance` summary entry.

- [ ] **Step 3b: Edit `s5_classification/null_ladder.py`**

In `_load_labels` add before the `raise` (≈ line 61):

```python
    elif target == "stance_ambivalence":
        from s4_encoding.build_dataset import _load_ambivalence_labels

        return _load_ambivalence_labels()
```

In `_class_names` add before the `raise` (≈ line 80):

```python
    elif target == "stance_ambivalence":
        return ["low", "med", "high"]
```

In `_chance_baseline` add before the `raise` (≈ line 71). Import the helper from the probe to
avoid duplication:

```python
    elif target == "stance_ambivalence":
        from s5_classification.structure_only_probe import majority_class_macro_f1

        return majority_class_macro_f1(list(_load_labels(target).values()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_ambivalence_target_wiring.py -v`
Expected: PASS (2 tests).
Run full suite: `PYTHONPATH=. uv run pytest -q`
Expected: PASS (no regressions).

- [ ] **Step 5: Typecheck, lint, commit**

```bash
uv run pyright s5_classification/null_ladder.py s5_classification/structure_only_probe.py
ruff check s5_classification/null_ladder.py s5_classification/structure_only_probe.py tests/test_ambivalence_target_wiring.py --fix
git add s5_classification/null_ladder.py s5_classification/structure_only_probe.py tests/test_ambivalence_target_wiring.py
git commit -m "feat: stance_ambivalence in null_ladder + structure_only (dynamic chance)"
```

---

### Task 6: Produce the labels (operational — API spend + manual adjudication)

No unit tests; this runs the pipeline built in Tasks 1-3 and produces `cache/ambivalence.jsonl`.

- [ ] **Step 1: Smoke-test each backend on 3 transcripts**

```bash
PYTHONPATH=. uv run python s2_extraction/ambivalence_labeler.py --backend agnes --limit 3
PYTHONPATH=. uv run python s2_extraction/ambivalence_labeler.py --backend haiku --limit 3
```
Expected: each prints 3 labels with no `❌`/`⚠` and writes `cache/ambivalence_agnes.jsonl` /
`cache/ambivalence_haiku.jsonl`. Inspect a couple of `reasoning`/`quotes` fields by eye for
sanity before spending the full run.

- [ ] **Step 2: Run the full corpus on both backends**

```bash
PYTHONPATH=. uv run python s2_extraction/ambivalence_labeler.py --backend agnes
PYTHONPATH=. uv run python s2_extraction/ambivalence_labeler.py --backend haiku
```
Expected: 1,250 labels each (cache-first; re-runs resume). Note any persistent `error` labels.

- [ ] **Step 3: Agreement report + disagreement worklist**

```bash
PYTHONPATH=. uv run python s2_extraction/ambivalence_consensus.py --report
```
Expected: prints `n_common`, `agreement_rate`, `kappa`; writes `cache/ambivalence_disagreements.json`.
Record the κ and agreement rate (used in results-log).

- [ ] **Step 4: Adjudicate disagreements (manual)**

Open `cache/ambivalence_disagreements.json`, read both models' reasoning/quotes per transcript,
and create `cache/ambivalence_adjudications.json` mapping each disagreeing id to your resolved
ordinal label, e.g.:

```json
{ "work_0142": "high", "creativity_0070": "med" }
```

Optional tripwire (spec §3): also eyeball ~10 random *agreement* cases for shared bias.

- [ ] **Step 5: Finalize the consensus file**

```bash
PYTHONPATH=. uv run python s2_extraction/ambivalence_consensus.py --finalize
```
Expected: writes `cache/ambivalence.jsonl` (agreements + adjudicated). Print line count =
n_agree + n_adjudicated.

- [ ] **Step 6: Record the label distribution + chance baseline**

```bash
PYTHONPATH=. uv run python -c "
from collections import Counter
from s4_encoding.build_dataset import _load_ambivalence_labels
from s5_classification.structure_only_probe import majority_class_macro_f1
labels = _load_ambivalence_labels()
print('distribution:', Counter(labels.values()))
print('chance macro-F1:', round(majority_class_macro_f1(list(labels.values())), 4))
"
```
Expected: a non-degenerate 3-class distribution (none of low/med/high near-zero). If one level is
near-empty, STOP and revisit the rubric (the construct may not be tri-modal) before Task 7.

- [ ] **Step 7: Commit the prompt-side artifacts only**

`cache/` is gitignored, so only commit the (already-committed) code. Nothing to add here unless
the rubric was edited; if so, bump to `ambivalence_v2.txt` (never edit a used prompt in place,
per project convention) and re-run.

---

### Task 7: Run the probes on `stance_ambivalence` and record results

- [ ] **Step 1: Build the per-seed splits (sanity)**

```bash
PYTHONPATH=. uv run python -c "
from s5_classification.repeated_eval import make_split
tr, va, te = make_split('stance_ambivalence', 42)
print('seed42 sizes:', len(tr), len(va), len(te))
"
```
Expected: ~70/15/15 split of the consensus-labeled n.

- [ ] **Step 2: structure_only probe**

```bash
PYTHONPATH=. uv run python s5_classification/structure_only_probe.py
```
Expected: prints structure_only mean F1 ± CI and delta-vs-chance for `stance_ambivalence`
(and the existing targets). Capture the JSON it writes.

- [ ] **Step 3: null-ladder (histogram vs GINEConv)**

```bash
PYTHONPATH=. uv run python s5_classification/null_ladder.py --target stance_ambivalence
```
Expected: prints GINEConv vs histogram mean F1, mean delta, 95% CI, per-seed table, PASS/FAIL.

- [ ] **Step 4: text baseline on the new target (self-contained)**

`repeated_eval.py` is a library (no CLI) and `repeated_run.py`'s `TARGETS` is hardcoded, so run
the text arm directly via the library functions. This mirrors the structure_only protocol (10
seeds, LogisticRegression, macro-F1) but on the frozen SBERT text embedding:

```bash
PYTHONPATH=. uv run python -c "
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from s5_classification.repeated_eval import get_split_data
from s5_classification.structure_only_probe import majority_class_macro_f1
from s4_encoding.build_dataset import _load_ambivalence_labels

chance = majority_class_macro_f1(list(_load_ambivalence_labels().values()))
f1s = []
for seed in range(42, 52):
    tr, _va, te = get_split_data('stance_ambivalence', seed)
    clf = LogisticRegression(max_iter=2000).fit(tr['text_emb'], tr['labels'])
    f1s.append(f1_score(te['labels'], clf.predict(te['text_emb']), average='macro'))
f1s = np.array(f1s)
lo, hi = np.percentile(f1s, [2.5, 97.5])
print(f'text macro-F1: {f1s.mean():.4f} +/- {f1s.std():.4f}  CI=[{lo:.4f},{hi:.4f}]')
print(f'chance: {chance:.4f}  delta_vs_chance: {f1s.mean()-chance:+.4f}')
"
```
Expected: text macro-F1 with CI on `stance_ambivalence`. The spec §5 hypothesis is that text
struggles here (small delta over chance) — i.e. ambivalence is NOT lexically recoverable, leaving
honest headroom for the graph arm.

- [ ] **Step 5: Record results in the results log**

Append a section to `.claude/context/results-log.md`:

```markdown
### Phase 2.6+ — Ambivalence target (stance_ambivalence)

**Date:** <date> | **Protocol:** 10-seed frozen CI (42-51)
**Label source:** Agnes + Haiku, neither = DeepSeek graph extractor.
**Inter-model agreement:** <rate>, Cohen's kappa = <k>. Adjudicated disagreements: <n>.
**Label distribution:** low=<>, med=<>, high=<>. Chance macro-F1 = <>.

| Arm | Mean macro-F1 | 95% CI | delta | Verdict |
|-----|---------------|--------|-------|---------|
| text (SBERT)        | | | vs chance | |
| structure_only      | | | vs chance | PASS/FAIL |
| histogram (null)    | | | — | |
| GINEConv (alt)      | | | vs histogram | PASS/FAIL |

**Interpretation:** <which of spec §5's three outcomes>. Topology claim rests on
GINEConv > histogram.
```

- [ ] **Step 6: Commit results**

```bash
git add .claude/context/results-log.md
git commit -m "results: stance_ambivalence target — null-ladder + structure_only + text"
```

- [ ] **Step 7: Update beads (epic 3ee)**

```bash
bd update graph-modality-3ee --notes="Ambivalence target built + run; see results-log Phase 2.6+. Gate re-evaluated on lexically-clean endogenous target."
bd vc commit -m "ambivalence target results recorded"
```

---

## Self-Review

**Spec coverage:** §1 construct → Task 1 (rubric). §2 dual-backend independent labeling → Tasks 2, 6. §3 consensus/κ/adjudication/optional tripwire → Tasks 3, 6. §4 experiment integration → Tasks 4, 5, 7. §5 honest framing → Task 7 results template. Disposition of prior targets → unchanged code paths preserved (cohort/ai_adoption still work). All covered.

**Placeholder scan:** No "TBD/TODO". The only deliberately-blank values are the user's adjudication labels (Task 6 Step 4) and the results numbers (Task 7 Step 5), which are runtime outputs, not plan gaps. The text-arm step (Task 7 Step 4) was verified against the real code: `repeated_eval.py` is a library (no CLI), so the step calls `get_split_data` directly rather than a nonexistent `--target` flag.

**Type consistency:** `stance_ambivalence` key, `{low,med,high,uncertain}` labels, `AMBIVALENCE_LABEL_MAP`, `majority_class_macro_f1`, `compute_agreement`/`merge_consensus`, `_load_ambivalence_labels`, `_class_names` — names used identically across Tasks 2-7. `model_tag` ("agnes"/"haiku") drives both the per-model cache filenames (Task 2) and the consensus input paths (Task 3) consistently.
