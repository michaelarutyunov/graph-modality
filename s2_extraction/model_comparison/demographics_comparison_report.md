# Demographic Extraction — DeepSeek vs Agnes Comparison

**Date:** 2026-06-08
**Sample:** 20 transcripts (8 workforce, 6 creatives, 6 scientists), stratified random, seed=42
**Prompt:** `s2_extraction/prompts/demographics_v1.txt` (v2 — refined AI adoption classes)

## Prompt evolution

### v1 → v2: AI adoption class refinement

The initial 4-class scheme (novice/experienced/power_user/uncertain) collapsed 90% of the sample into "experienced". Analysis of the "experienced" transcripts revealed a natural split along **depth of integration**:

- **tool_user**: AI as a time-saver for bounded tasks. Could work without it.
- **integrated**: AI woven into daily process. Has methods, strategies, opinions. Would notice its absence.

The v2 prompt adds a signal table distinguishing the two classes (one-shot vs iterative, default interface vs custom configs, minor inconvenience vs workflow disruption).

## Quantitative results (v2 prompt)

| Metric | Career Stage | AI Adoption |
|---|---|---|
| Agreement rate | 45.0% | 85.0% |
| DeepSeek uncertain rate | 60.0% | 0.0% |
| Agnes uncertain rate | 10.0% | 0.0% |

### AI adoption class distribution

| Label | DeepSeek | Agnes |
|---|---|---|
| novice | 1 (5%) | 1 (5%) |
| tool_user | 8 (40%) | 11 (55%) |
| integrated | 11 (55%) | 8 (40%) |
| power_user | 0 (0%) | 0 (0%) |
| uncertain | 0 (0%) | 0 (0%) |

The distribution is now well-spread (~40/55 tool_user/integrated split), suitable for classification. Zero power_users in 20 samples is expected — the bar (API-level usage, custom code, programmatic integration) is appropriately high; a few will emerge in the full 1,250.

### Career stage class distribution

| Label | DeepSeek | Agnes |
|---|---|---|
| early | 5 | 4 |
| mid | 3 | 14 |
| late | 0 | 0 |
| uncertain | 12 | 2 |

Agnes continues to over-assign "mid" (70% vs DeepSeek's 15%). DeepSeek's 60% uncertain rate is correct behavior — most interviewees don't mention career timeline markers.

## Qualitative analysis (v2)

### AI adoption agreement (85%)

The 3 disagreements are all DeepSeek=integrated vs Agnes=tool_user (creativity_0011, science_0027, science_0075). DeepSeek correctly identifies iterative refinement and workflow integration that Agnes misses:

- `science_0027`: "I didn't use AI until I had a version of the game that I could use/test — the purpose of AI was to spot bugs" → DeepSeek sees the deliberate workflow design, Agnes only sees task-level usage.

### Borderline cases correctly classified

- `work_0142` ("I am the AI Pioneer within the team"): DeepSeek → integrated. Correct — advocating within a team is not the same as building custom pipelines. Falls short of power_user.
- `work_0228` (Custom GPT, fed employer guidelines): DeepSeek → integrated. Correct — a Custom GPT is a ChatGPT feature, not API-level integration.

### Career stage: Agnes still over-assigns "mid"

Agnes labels "I work as a QA tester" → mid. This is the same pattern as v1 — Agnes conflates "has a job" with "established professional" while ignoring the prompt's requirement for explicit experience/timeline evidence.

## Decision

**Selected model: DeepSeek (deepseek-chat)**

Reasons (unchanged from v1, reinforced by v2):
1. Follows "uncertain is safe" instruction — 60% uncertain on career stage where evidence is genuinely absent
2. Makes finer-grained AI adoption distinctions — catches iterative refinement and workflow integration that Agnes misses
3. The new AI adoption scheme (tool_user/integrated) produces a well-balanced distribution (~40/55) suitable for multi-class classification
4. Estimated cost: ~$2-3 for 1,250 transcripts

## Next steps

1. Run DeepSeek on all 1,250 transcripts with prompt v2
2. Cache results as `cache/demographics.jsonl`
3. Create classification tasks for career_stage (4-class) and ai_adoption (5-class)
4. Evaluate text-only, graph-only (2b/3b), and text+graph routes on new targets
5. Compare: does graph modality add more value on these targets than on professional cohort?
