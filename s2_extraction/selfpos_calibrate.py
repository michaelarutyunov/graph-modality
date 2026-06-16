"""B2 calibration bake-off for the delegation-boundary coding rubric (selfpos_v1, multi-label).

Runs a model panel (Agnes / Haiku / Kimi) over the 10 B1 reference transcripts using
prompts/selfpos_v1.txt, then scores each model against the human reference
(results/method_review/selfpos_calib/reference.jsonl).

Rationale is MULTI-LABEL: scoring is Jaccard overlap of the rationales_present set
(robust to forced-choice noise), plus per-category presence agreement, plus
delegation_breadth / boundary_talk_depth agreement.

Decision rule (pinned, bead graph-modality-zb9, refined 2026-06-16): pick the CHEAPEST panel
model whose mean Jaccard vs reference on rationales_present is >= 0.60. Prefer the cheap pair
Agnes + Haiku if both pass; Kimi is fallback only.

    PYTHONPATH=. uv run python s2_extraction/selfpos_calibrate.py

DeepSeek excluded (it is the graph extractor -> circularity).
"""

from __future__ import annotations

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
OUT_DIR = Path("results/method_review/selfpos_calib")
REF_PATH = OUT_DIR / "reference.jsonl"

CALIB_IDS = [
    "work_0150",
    "work_0350",
    "work_0550",
    "work_0750",
    "creativity_0020",
    "creativity_0055",
    "creativity_0110",
    "science_0020",
    "science_0055",
    "science_0110",
]

# cheapest -> most expensive (selection rule walks this order)
PANEL = [
    {
        "tag": "agnes",
        "type": "openai",
        "model": "agnes-2.0-flash",
        "key": "AGNES_API_KEY",
        "url": "https://apihub.agnes-ai.com/v1/chat/completions",
        "temperature": 0.0,
    },
    {
        "tag": "haiku",
        "type": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "key": "ANTHROPIC_API_KEY",
        "temperature": 0.0,
    },
    # Kimi dropped from re-run: k2.6 thinking model times out at 60s and is not needed if the
    # cheap pair passes. To re-enable as a fallback, restore this entry AND raise TIMEOUT to ~300.
    # {"tag": "kimi", "type": "openai", "model": "kimi-k2.6", "key": "KIMI_API_KEY",
    #  "url": os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1/chat/completions"),
    #  "temperature": 1.0},
]

RATIONALE_CATS = [
    "competence",
    "competence_compensate",
    "identity",
    "trust_reliability",
    "output_efficiency",
    "other",
]
CORE_CATS = {"competence", "identity", "trust_reliability"}  # Gate-1 load-bearing categories
JACCARD_THRESHOLD = 0.60  # applied to CORE-category Jaccard
MAX_TOKENS = 1800
TIMEOUT = 60
RETRIES = 3
DELAYS = [2, 6, 15]


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
                raw = json.loads(resp.read().decode("utf-8"))
                return parse_json(raw["choices"][0]["message"]["content"])
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            last = e
            time.sleep(DELAYS[min(attempt, len(DELAYS) - 1)])
    raise RuntimeError(f"openai call failed: {last}")


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
    raise RuntimeError(f"anthropic call failed: {last}")


def human_text(record: dict) -> str:
    turns = [t["text"] for t in record["turns"] if t["speaker"] == "Human"]
    txt = "\n\n".join(turns)
    return txt[:8000] + "\n\n[... truncated]" if len(txt) > 8000 else txt


def load_records() -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for path in sorted(TAGGED_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                r = json.loads(line)
                recs[r["transcript_id"]] = r
    return recs


def rset(record: dict) -> set | None:
    """Distinct rationale set for a record, or None if the model errored."""
    if record is None or "_error" in record:
        return None
    rp = record.get("rationales_present")
    if rp is None:  # derive from tasks
        rp = []
        for t in record.get("tasks", []):
            rp += t.get("rationales", []) or ([t["rationale"]] if t.get("rationale") else [])
    return {str(x).strip().lower() for x in rp if x}


def jaccard(a: set | None, b: set | None) -> float | None:
    if a is None or b is None:
        return None
    if not a and not b:
        return 1.0
    return round(len(a & b) / len(a | b), 3) if (a | b) else 1.0


def core_only(s: set | None) -> set | None:
    return None if s is None else (s & CORE_CATS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tpl = PROMPT_PATH.read_text(encoding="utf-8")
    records = load_records()
    ref = {
        json.loads(l)["transcript_id"]: json.loads(l)
        for l in REF_PATH.read_text().strip().split("\n")
    }

    panel: dict[str, dict[str, dict]] = {}
    for b in PANEL:
        tag = b["tag"]
        caller = call_openai if b["type"] == "openai" else call_anthropic
        panel[tag] = {}
        for i, tid in enumerate(CALIB_IDS, 1):
            prompt = tpl.replace("{transcript}", human_text(records[tid]))
            try:
                out = caller(prompt, b)
                print(f"[{tag} {i}/10] {tid} OK")
            except Exception as e:
                out = {"_error": str(e)}
                print(f"[{tag} {i}/10] {tid} FAIL: {e}")
            out["transcript_id"] = tid
            panel[tag][tid] = out
            time.sleep(0.6)
        (OUT_DIR / f"panel_{tag}.jsonl").write_text(
            "\n".join(json.dumps(panel[tag][t], ensure_ascii=False) for t in CALIB_IDS) + "\n"
        )

    ref_sets = {t: rset(ref[t]) for t in CALIB_IDS}

    def score(tag: str) -> dict:
        recs = panel[tag]
        jac = [jaccard(core_only(rset(recs[t])), core_only(ref_sets[t])) for t in CALIB_IDS]
        jac_ok = [j for j in jac if j is not None]
        mean_jac = round(sum(jac_ok) / len(jac_ok), 3) if jac_ok else None
        fj = [jaccard(rset(recs[t]), ref_sets[t]) for t in CALIB_IDS]
        fj_ok = [j for j in fj if j is not None]
        mean_fj = round(sum(fj_ok) / len(fj_ok), 3) if fj_ok else None
        per_cat = {}
        for cat in RATIONALE_CATS:
            hits = 0
            n = 0
            for t in CALIB_IDS:
                ms, rs = rset(recs[t]), ref_sets[t]
                if ms is None or rs is None:
                    continue
                n += 1
                hits += (cat in ms) == (cat in rs)
            per_cat[cat] = round(hits / n, 3) if n else None
        breadth = round(
            sum(
                ("_error" not in recs[t])
                and recs[t].get("delegation_breadth") == ref[t].get("delegation_breadth")
                for t in CALIB_IDS
            )
            / len(CALIB_IDS),
            3,
        )
        depth = round(
            sum(
                ("_error" not in recs[t])
                and recs[t].get("boundary_talk_depth") == ref[t].get("boundary_talk_depth")
                for t in CALIB_IDS
            )
            / len(CALIB_IDS),
            3,
        )
        return {
            "n_ok": sum("_error" not in recs[t] for t in CALIB_IDS),
            "mean_jaccard_vs_ref": mean_jac,
            "mean_jaccard_full_vs_ref": mean_fj,
            "per_category_presence_agreement": per_cat,
            "delegation_breadth_agreement": breadth,
            "boundary_talk_depth_agreement": depth,
        }

    vs_ref = {b["tag"]: score(b["tag"]) for b in PANEL}

    tags = [b["tag"] for b in PANEL]
    inter = {}
    for i, t1 in enumerate(tags):
        for t2 in tags[i + 1 :]:
            js = [
                jaccard(core_only(rset(panel[t1][t])), core_only(rset(panel[t2][t])))
                for t in CALIB_IDS
            ]
            js = [j for j in js if j is not None]
            inter[f"{t1}__{t2}"] = round(sum(js) / len(js), 3) if js else None

    chosen, decision = None, ""
    for b in PANEL:  # cheapest first
        mj = vs_ref[b["tag"]]["mean_jaccard_vs_ref"]
        if mj is not None and mj >= JACCARD_THRESHOLD:
            chosen = b["tag"]
            break
    cheap_ok = [
        t
        for t in ("agnes", "haiku")
        if (vs_ref.get(t, {}).get("mean_jaccard_vs_ref") or 0) >= JACCARD_THRESHOLD
    ]
    if len(cheap_ok) == 2:
        pair = ["agnes", "haiku"]
        decision = (
            "Cheap pair Agnes+Haiku BOTH pass core-Jaccard>=0.60 -> Gate-1 pair; Kimi not needed."
        )
    elif len(cheap_ok) == 1:
        pair = cheap_ok
        decision = (
            f"Only {cheap_ok[0]} passes core-Jaccard>=0.60. Per ladder: Kimi-only fallback "
            "(re-enable Kimi with TIMEOUT~300) or stop+Sonnet estimate - USER decision."
        )
    else:
        pair = []
        decision = (
            "Neither cheap model passes core-Jaccard>=0.60. Per ladder: Kimi-only "
            "(re-enable Kimi) or stop+Sonnet estimate - USER decision."
        )

    summary = {
        "n_transcripts": len(CALIB_IDS),
        "rationale_scoring": "CORE-category (competence/identity/trust) Jaccard; full-set secondary",
        "jaccard_threshold": JACCARD_THRESHOLD,
        "vs_reference": vs_ref,
        "inter_model_jaccard": inter,
        "chosen_cheapest_pass": chosen,
        "gate1_pair": pair,
        "decision": decision,
        "note": "n=10 calibration; Jaccard de-noises the multi-label forced-choice problem.",
    }
    (OUT_DIR / "calibration.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== DECISION ===\n" + decision)
    print(
        json.dumps(
            {
                t: {
                    "mean_jaccard": vs_ref[t]["mean_jaccard_vs_ref"],
                    "competence_presence_agr": vs_ref[t]["per_category_presence_agreement"][
                        "competence"
                    ],
                    "breadth_agr": vs_ref[t]["delegation_breadth_agreement"],
                }
                for t in tags
            },
            indent=2,
        )
    )
    print("inter-model Jaccard:", json.dumps(inter))


if __name__ == "__main__":
    main()
