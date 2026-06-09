# ADR 0001: Extraction Model Selection

- **Status:** Accepted
- **Date:** 2026-06-07
- **Phase:** 1 (Extraction)

## Context

Graph extraction requires an LLM to parse unstructured interview transcripts into a structured JSON concept graph conforming to the project ontology (4 entity types, 4 relation types, 6 structural constraints). Three candidate models were available: Claude (Anthropic API), DeepSeek (DeepSeek API), and Agnes (OpenAI-compatible API).

The model must:
1. Produce valid JSON matching the graph schema
2. Handle transcripts up to 27k characters
3. Maintain consistent graph size and ontology adherence across 1,250 transcripts
4. Be cost-effective at scale

## Decision

**Selected DeepSeek (`deepseek-chat`) via OpenAI-compatible endpoint with JSON mode (`response_format={"type": "json_object"}`)** for production extraction.

### Model comparison results (10 fixed transcripts)

| Model | Success Rate | Mean Nodes | Mean Edges | Violations | Notes |
|---|---|---|---|---|---|
| Claude (claude-sonnet-4-6) | 10/10 | 15.1 | 15.8 | 0 | Richest graphs, most expensive |
| DeepSeek (Anthropic endpoint) | 0/10 | — | — | — | **Failed:** thinking blocks + JSON truncation |
| DeepSeek (OpenAI endpoint, JSON mode) | 3/3 | 14–18 | 13–19 | 0 | Correct config, reliable |
| Agnes (agnes-2.0-flash) | 3/3 | 11–15 | 10–13 | 0–1 | Leaner graphs, works reliably |

### Critical finding: endpoint matters

DeepSeek exposes two API endpoints:
- **Anthropic-compatible endpoint** (`deepseek-v4-pro`): forces thinking mode, which causes JSON truncation. 0/10 success rate. **Unusable.**
- **OpenAI-compatible endpoint** (`deepseek-chat`): supports `response_format={"type": "json_object"}` for guaranteed JSON output. 100% success rate.

## Rationale

1. **Cost:** DeepSeek is ~10× cheaper than Claude for production extraction (1,250 transcripts)
2. **Quality parity:** DeepSeek and Claude produce comparable graph sizes and violation rates
3. **JSON reliability:** The OpenAI endpoint's JSON mode guarantees parseable output
4. **Claude as reference:** Claude was retained as the quality baseline for model comparison rubric scoring

## Consequences

- All 1,250 graphs extracted with DeepSeek; Claude used only for comparison scoring
- The Anthropic-compatible DeepSeek endpoint is unusable for structured extraction tasks
- Prompt v3 (two-shot examples) was tuned for DeepSeek's output style
- Agnes produces leaner graphs (fewer nodes/edges) — may miss subtle constructs

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Claude for all 1,250 | 10× cost with marginal quality improvement |
| Agnes for production | Leaner graphs risk missing constructs; lower rubric scores |
| DeepSeek Anthropic endpoint | 0% success rate due to JSON truncation |
| Multi-model ensemble | Adds complexity without clear quality benefit at scale |
