# 0005 — Adjudicating stance_ambivalence disagreements with Kimi

- **Status:** Accepted
- **Date:** 2026-06-14
- **Supersedes:** none
- **Related:** `s2_extraction/ambivalence_labeler.py`, `s2_extraction/ambivalence_adjudicator.py`, `s2_extraction/ambivalence_consensus.py`; `docs/superpowers/specs/2026-06-12-ambivalence-target-design.md`; `.claude/context/results-log.md`

## Context

The `stance_ambivalence` target was introduced as a cleaner primary label than `ai_adoption` for testing whether graph structure carries signal beyond text. It is ordinal (`low` / `med` / `high`) and lexically non-obvious: the label is meant to capture *unresolved attitudinal tension*, not raw sentiment or topic.

To avoid circularity, the labeler uses two models that are **not** the DeepSeek graph extractor:

- **Agnes** (`agnes-2.0-flash`) via OpenAI-compatible endpoint.
- **Haiku** (`claude-haiku-4-5-20251001`) via Anthropic SDK.

Both models output `label`, `reasoning`, and supporting `quotes` under the `ambivalence_v1.txt` rubric.

Initial inter-annotator agreement:

- Common transcripts: 1,250
- Agreements: 973
- Disagreements: 277 (22.2%)
- Agreement rate: 77.8%
- Cohen's κ: 0.504 (moderate)

The disagreements had to be reconciled without dropping records, because the research question requires the full 1,250-graph corpus.

## Decision

Use **Kimi k2.6** as a third-party judge over the 277 disagreements.

The judge is shown:

1. The full human-only transcript.
2. The exact `ambivalence_v1.txt` rubric.
3. The two conflicting annotations, **anonymized** as "Annotator A" and "Annotator B".
4. Each annotator's reasoning and supporting quotes.

Kimi must return a single JSON object:

```json
{
  "chosen_label": "low|med|high|uncertain|manual_review",
  "chosen_annotator": "A|B|none",
  "reasoning": "...",
  "supporting_quotes": ["..."]
}
```

Implementation details:

- Order of A/B is randomized per transcript to avoid position bias.
- Kimi k2.6 is called with `temperature=0.6`, `thinking={"type": "disabled"}`, and `response_format={"type": "json_object"}` (the model requires these parameters).
- A detailed audit log is written to `cache/ambivalence_adjudication_details.jsonl`.
- The final consensus file `cache/ambivalence.jsonl` merges the 973 direct agreements with the Kimi-resolved disagreements.
- `_load_ambivalence_labels()` skips any `uncertain` or `manual_review` labels while keeping them in the consensus file for auditability.

## Consequences

- All 1,250 transcripts have a usable `stance_ambivalence` label.
- Final distribution: `med=843`, `low=352`, `high=55`.
- The adjudication process is fully auditable: every disagreement can be traced to the two original annotations, Kimi's reasoning, and the chosen quotes.
- The label is now ready for the existing `null_ladder.py` and `structure_only_probe.py` pipelines.
- Class imbalance is severe (`high` = 4.4%); this is documented as a known limitation rather than hidden by relabeling.

## Alternatives considered

1. **Run Kimi as a third independent labeler and take majority vote.** Rejected: it ignores the existing reasonings and quotes, and would still leave 3-way ties requiring manual review.
2. **Manual adjudication of all 277 disagreements.** Rejected: too time-consuming for the expected gain; Kimi judge quality is sufficient given the rubric and evidence.
3. **Drop disagreed records.** Rejected: the research design requires the full 1,250-graph corpus; dropping 22% would bias the sample.
4. **Researcher relabeling of "borderline" med cases to high.** Rejected: it would be post-hoc data dredging, invalidating the rubric and the adjudication audit trail.

## Open risks

- The `med` class dominates. Any classifier may collapse toward `med`. Class weighting, stratified sampling, and per-class metrics are recommended.
- The `high` class is small (n=55), so statistical power for detecting graph-vs-text differences on `high` vs. others is limited.
- The ordinal thresholds remain somewhat arbitrary; future work could treat ambivalence as continuous via ordinal regression or collect finer-grained annotations.
