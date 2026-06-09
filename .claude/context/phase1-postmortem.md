# phase1-postmortem.md

> **Purpose:** retrospective on the extraction phase (Phase 1) — what went well, what went wrong,
> key decisions, and lessons for future extraction pipelines.
>
> **Created:** 2026-06-09 | **Phase 4 — Synthesis**

---

## 1. Summary

Phase 1 extracted concept graphs from 1,250 Anthropic Interviewer transcripts using DeepSeek (deepseek-chat)
via the OpenAI-compatible endpoint with JSON mode. Extraction completed with 100% success rate, 0.3%
validation violation rate, and a mean of 14.9 nodes / 13.6 edges per graph. The extraction prompt
evolved from v1 (no examples) to v3 (two-shot: workforce + scientist), with each version preserved
and never deleted.

---

## 2. What went well

### 2.1 Prompt versioning discipline

Prompts live as numbered text files in `extraction/prompts/` (v1.txt, v2.txt, v3.txt). Each version
is preserved immutably — no overwriting, no deletion. This made it trivial to trace which prompt
produced which graphs and to compare versions systematically. The version number is the sole source
of truth; there is no separate registry or database.

**Lesson:** File-system-based versioning with a simple numbering convention is sufficient for research
pipelines with <10 prompt iterations. No need for a prompt registry or database.

### 2.2 Cache-everything architecture

Every extraction checks `data/graphs/free_text/{tid}.json` before making an API call. This made
re-running the pipeline zero-cost — if a transcript was already extracted, it was skipped. The
cache was also critical for recovery: when the API config was wrong (see §3.1), we could fix the
config and re-run only the failed transcripts without re-extracting the successful ones.

**Lesson:** Per-item caching with existence checks is the single highest-ROI engineering decision
in an API-dependent pipeline. It costs ~10 lines of code and saves hours of re-extraction and
dollars of API costs.

### 2.3 DeepSeek reliability (once correctly configured)

After switching to the OpenAI-compatible endpoint with JSON mode (see §3.1), DeepSeek produced
1,250 graphs with zero extraction failures and only 4 validation violations (0.3%). The graphs
are structurally consistent: mean 14.9 nodes, 13.6 edges, with tight distributions. DeepSeek
is fast (~3-5s per transcript) and cost-effective compared to Claude.

**Lesson:** DeepSeek via the OpenAI-compatible endpoint is a viable, reliable extraction backend
for structured JSON tasks at scale.

### 2.4 Validation as a non-blocking check

The validator runs after every extraction but logs violations rather than raising exceptions.
This meant that 4 graphs with minor violations (e.g., missing negative pole) were preserved
rather than discarded, and could be inspected manually. A blocking validator would have
required re-extraction or manual intervention for those 4 cases.

**Lesson:** Non-blocking validation with logged violations is the right pattern for research
pipelines where edge cases should be inspected, not discarded.

### 2.5 Three-model comparison informed the scale decision

The 10-transcript comparison (Claude, DeepSeek, Agnes) with a structured rubric (5 criteria,
scored 1-3) gave us confidence in DeepSeek before committing to 1,250 extractions. The
comparison was cheap (~30 minutes, $2-3 in API costs) and prevented a potentially costly
wrong choice.

**Lesson:** A small, fixed-sample model comparison with a structured rubric is a high-value
gate before scale extraction.

---

## 3. What went wrong

### 3.1 Anthropic-compatible endpoint with DeepSeek (CRITICAL)

**Problem:** Initial extraction attempts used DeepSeek's Anthropic-compatible Messages endpoint.
This endpoint forces thinking mode (`thinking: "enabled"`) which interleaves reasoning tokens
with the JSON output, causing truncation. Zero successful extractions.

**Root cause:** DeepSeek's Anthropic-compatible endpoint does not support disabling thinking mode.
The thinking tokens are injected into the response stream and corrupt the JSON structure.

**Fix:** Switched to DeepSeek's OpenAI-compatible endpoint (`deepseek-chat`) with
`response_format={"type": "json_object"}`. This produces clean JSON output without
thinking-mode interference.

**Detection:** Caught during the model comparison experiment (10 transcripts, 0/10 success)
before scale extraction. If we had gone straight to scale, we would have burned through
API credits with zero usable output.

**Lesson:** Always run a small validation batch (5-10 transcripts) before committing to a
scale extraction run. The cost of the validation batch is negligible; the cost of a failed
scale run is not.

### 3.2 JSON truncation with large transcripts

**Problem:** Two transcripts in the model comparison sample (>20k characters) produced
truncated JSON output from DeepSeek even with the OpenAI endpoint.

**Fix:** Added retry logic with exponential backoff (2s, 8s, 32s). On retry, the model
sometimes produces complete output. For the two persistently truncated transcripts, a
second attempt succeeded.

**Lesson:** Retry with backoff is necessary but not always sufficient. For very long
transcripts, consider chunking or summarisation pre-processing.

### 3.3 Interviewer confound (Phase 3, but detected extraction-side)

**Problem:** The AI interviewer uses cohort-specific opening scripts ("...using AI in your
**creative** work" vs "...using AI in your **scientific** work"), injecting the cohort label
into every transcript. This created a trivial classification signal (val macro-F1 = 1.0
across all routes) that masked any genuine graph-topology signal.

**Detection:** Caught during Phase 3 classification, not Phase 1 extraction. The ceiling
effect was so perfect (F1 = 1.0000) that it immediately triggered investigation.

**Fix:** `text_encoder.py` was updated to encode only human turns (`speaker_filter="Human"`),
stripping AI interviewer speech. All embeddings and models were recomputed.

**Lesson:** When working with AI-conducted interviews, the interviewer's language is a
confound. Always verify that classification signal comes from the interviewee, not the
interviewer. A perfect or near-perfect classification result should trigger confound
investigation, not celebration.

### 3.4 Ceiling effects in graph metrics

**Problem:** Two of the four pre-registered hypotheses (H3, H4) were untestable due to
ceiling effects in the extracted graphs. H3 (bipolarity): 99.8% of constructs have both
poles defined (mean bipolarity score = 0.997). H4 (CSM count): 99.8% of graphs have
exactly 2 CSMs (the ontology ceiling). These metrics have zero variance, making statistical
tests meaningless.

**Root cause:** The extraction prompt and validation constraints are strong enough that the
extractor produces near-ceiling compliance. This is good for graph quality but bad for
hypothesis testing — the metrics that would differentiate cohorts are forced to uniformity.

**Lesson:** Pre-register hypotheses on metrics with genuine expected variance. If a metric
is constrained by the ontology (e.g., max 2 CSMs, bipolarity required), it cannot serve as
a dependent variable. Consider relaxing constraints or using continuous rather than binary
metrics.

---

## 4. Key Decisions

| decision | rationale | consequence |
|---|---|---|
| OpenAI-compatible endpoint for DeepSeek | Anthropic endpoint forces thinking mode, corrupts JSON | Clean extraction, 0 failures |
| JSON mode (`response_format={"type": "json_object"}`) | Guarantees valid JSON output | No post-processing needed |
| Prompt v3 (two-shot) | Improved ontology adherence over v1/v2 | Better bipolarity capture, lower violation rate |
| Cache before API call | Idempotent re-runs, cost savings | Trivial recovery from config errors |
| Non-blocking validation | Preserve edge cases for inspection | 4 graphs with violations kept, not discarded |
| Human-only text encoding | Remove interviewer confound | Genuine classification task, F1 dropped from 1.0 → 0.82-0.84 |

---

## 5. Lessons for Future Extraction Pipelines

1. **Validate the API config on 5 transcripts before scale.** The 10-transcript comparison
   caught two critical issues (thinking mode, JSON truncation) that would have been
   catastrophic at scale.

2. **Cache per item, not per batch.** Per-transcript caching means a single failure doesn't
   require re-running the whole batch.

3. **Version prompts as files, not in code.** File-system versioning is simpler, more
   auditable, and works with git diff.

4. **Check for ceiling effects before pre-registering hypotheses.** If the ontology
   enforces a constraint (max 2 CSMs, bipolarity required), those metrics cannot
   differentiate groups.

5. **Strip interviewer speech when the interviewer knows the target label.** AI-conducted
   interviews with cohort-specific scripts create a confound that swamps genuine signal.

6. **Non-blocking validation > blocking validation for research.** You want to inspect
   edge cases, not discard them.

---

## 6. Cross-References

- [results-log.md](results-log.md) — Phase 3 classification results and Phase 4 structural analysis
- [graph-schema.md](graph-schema.md) — Graph JSON schema and ontology constraints
- [extraction-log.md](extraction-log.md) — Prompt version history and model comparison details
- [CHARTER.md](../CHARTER.md) — Research questions and hypotheses
- [ENGINEERING.md](../ENGINEERING.md) — Full technical specification
