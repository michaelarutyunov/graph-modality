# extraction-log.md

> **Purpose:** track extraction prompt version history, model comparison results, extraction statistics, and any deviations or incidents during graph extraction. This is the canonical record for Phase 1.

---

## prompt version history

| version | file | date | change summary | rationale |
|---|---|---|---|---|
| v1 | `s2_extraction/prompts/v1.txt` | 2026-06-07 | Initial extraction prompt | Based on ENGINEERING.md §5.2. Four entity types (Construct, Value, Stance, CognitiveStyleMarker), four relation types (SERVES, EXPRESSED_VIA, MODULATED_BY, CONFLICTS_WITH). Human-turn-only node extraction. Maximum 2 CSM per transcript. |

---

## dataset verification

| date | finding |
|---|---|
| 2026-06-07 | Confirmed dataset uses `Assistant:` (opening turn), `AI:` (subsequent AI turns), `User:` (human turns) — NOT `Human:` as ENGINEERING.md §5.1 originally stated. ENGINEERING.md to be corrected. |
| 2026-06-07 | 1,250 transcripts confirmed: workforce 1,000, creatives 125, scientists 125. Turns range: 15–55 per transcript. |

---

## manual review (6 transcripts, 2026-06-07)

Reviewed 6 graphs across 3 splits (2 workforce, 2 creatives, 1 scientist, plus work_0000 from earlier test). All passed structural validation with zero violations.

| metric | value |
|---|---|
| transcripts reviewed | 6 (work_0000, work_0001, work_0002, creativity_0000, creativity_0001, science_0000) |
| extraction model | claude-sonnet-4-6 |
| prompt version | v1 |
| mean nodes | 13.5 (range 13–14) |
| mean edges | 13.7 (range 12–17) |
| bipolarity completeness | 100% (all Constructs fully bipolar) |
| CSM count | 2 per graph (at ceiling, consistent) |
| CONFLICTS_WITH edges | present in all 6 graphs (9 total) |
| Stance→Value violations | 0 |
| validation failures | 0 / 6 |

**Qualitative observations:**
- Construct labels are specific and well-grounded in transcript spans
- All graphs exhibit the expected hub-and-spoke structure (Values as high-degree hubs)
- CONFLICTS_WITH edges appear in every graph — higher than expected for short interviews; suggests the prompt may be encouraging conflict identification
- CSM count exactly 2 in every graph — LLM may be "filling the quota" rather than genuinely identifying style markers
- Prompt v1 appears solid; no iteration needed before model comparison

**Decision:** Prompt v1 is locked. Proceed to model comparison experiment.

---

## model comparison

> *To be populated after the comparison experiment (`graph-modality-7xc`).*

| model | R1 (size consistency) | R2 (ontology) | R3 (bipolarity) | R4 (hallucination) | R5 (relation typing) | total |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

**Winner:** Claude (claude-sonnet-4-6) — DeepSeek failed to produce any valid graphs.  
**DeepSeek analysis:** 0/10 transcripts produced valid graphs. Failure modes: (a) response contained only thinking blocks, no text output (5/10), (b) JSON truncated mid-string — model hit output limits or lost track of JSON structure (5/10). The `deepseek-v4-pro` thinking variant is unsuitable for structured JSON extraction. A non-thinking variant (`deepseek-v4-flash`) may work but was not tested.  
**Agnes:** skipped — API endpoint unknown.  
**Tiebreaker used:** N/A (one-sided result)

**Sample IDs:** `s2_extraction/model_comparison/sample_ids.txt` (4 workforce, 3 creatives, 3 scientists)

**Claude metrics (10/10 after retry of science_0060):**
- Mean nodes: 15.1 (range 11–21)
- Mean edges: 15.8 (range 12–21)
- CV(N): 0.198 (low variance = consistent extraction)
- Bipolarity: 100% (all Constructs fully bipolar)
- Validation violations: 0
- Transient failure: science_0060 (empty response on first attempt; succeeded on retry)

---

## scale extraction stats

**Completed 2026-06-08.** DeepSeek (deepseek-chat) via OpenAI-compatible endpoint with JSON mode. Prompt v3 (two-shot examples).

| metric | value |
|---|---|
| transcripts processed | 1,250 |
| successful extractions | 1,250 (100%) |
| failed extractions | 0 |
| extraction model | deepseek-chat |
| prompt version | v3 |
| mean nodes per graph | 14.9 |
| mean edges per graph | 13.6 |
| graphs with violations | 4 (0.3%) |
| failure log | `s2_extraction/failed.txt` is empty (0 entries) |
| backend | OpenAI-compatible, `response_format={"type": "json_object"}`, `max_tokens=8192` |
|---|---|
| transcripts processed | — |
| successful extractions | — |
| failed extractions | — |
| invalid graphs (validator violations) | — |
| mean nodes per graph | — |
| mean edges per graph | — |
| mean bipolarity score | — |
| CONFLICTS_WITH edges found | — |

---

## incidents and deviations

> *Record any unexpected behaviour, API issues, or decisions that depart from the plan.*

### 2026-06-07: `ANTHROPIC_BASE_URL` env var conflict

The global `.env` contains `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic/...` (from the Claude Code harness config). The Anthropic Python SDK reads this env var and routes ALL calls to DeepSeek's API, which rejects real Anthropic API keys with 401. Fixed by passing `base_url="https://api.anthropic.com"` explicitly to `Anthropic()` constructor in `extractor.py`. The global env var is left untouched — it's needed for Claude Code's own operation.

### 2026-06-07: Transcript ID hallucination bug

`validate_graph()` used `graph.setdefault("transcript_id", transcript_id)` to attach metadata. When the LLM included a `transcript_id` in its JSON output (which it often hallucinates), `setdefault` preserved the LLM's value instead of the true dataset ID. This caused duplicate IDs across different splits (e.g. three graphs with `interview_001`). Fixed by using direct assignment `graph["transcript_id"] = transcript_id`. Affected cache files were deleted and re-extracted.
