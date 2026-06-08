# Demographic Extraction — DeepSeek vs Agnes Comparison

**Date:** 2026-06-08
**Sample:** 20 transcripts (8 workforce, 6 creatives, 6 scientists), stratified random, seed=42
**Prompt:** `extraction/prompts/demographics_v1.txt`

## Quantitative results

| Metric | Career Stage | AI Adoption |
|---|---|---|
| Agreement rate | 40.0% | 90.0% |
| DeepSeek uncertain rate | 65.0% | 0.0% |
| Agnes uncertain rate | 10.0% | 0.0% |

### Career stage class distribution

| Label | DeepSeek | Agnes |
|---|---|---|
| early | 5 | 4 |
| mid | 2 | 14 |
| late | 0 | 0 |
| uncertain | 13 | 2 |

### AI adoption class distribution

| Label | DeepSeek | Agnes |
|---|---|---|
| experienced | 18 | 16 |
| novice | 2 | 3 |
| power_user | 0 | 1 |
| uncertain | 0 | 0 |

## Qualitative analysis

Agnes over-assigns "mid" career stage (14/20 = 70%). It labels someone as "mid" based on merely having a job — e.g., "I work as a QA tester" → mid. This violates the prompt's instruction to find explicit experience/timeline markers.

DeepSeek correctly defaults to "uncertain" when timeline evidence is absent (13/20 = 65%). Its "mid" assignments (2) are backed by genuine evidence like team management or role descriptions indicating established career.

Both models agree strongly on AI adoption (90%), showing the attribute is well-defined and evidence is abundant in transcripts.

## Decision

**Selected model: DeepSeek (deepseek-chat)**

Reasons:
1. Follows instructions faithfully — defaults to "uncertain" when evidence is absent
2. The prompt explicitly says "uncertain is safe" — DeepSeek respects this, Agnes ignores it
3. AI adoption agreement (90%) shows DeepSeek CAN extract evidence when it exists — the career stage uncertainty is genuine lack of signal, not model failure
4. DeepSeek is cheaper (~$2-3 for 1,250 transcripts vs ~$5-10 for Agnes)

## Next steps

1. Run DeepSeek on all 1,250 transcripts
2. Cache results as `cache/demographics.jsonl`
3. Create classification task for career_stage and ai_adoption
4. Evaluate text-only, graph-only, and text+graph routes on new targets
