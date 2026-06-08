# Demographic Variable Selection for Alternative Classification Targets

> **Status:** design rationale — created 2026-06-08
> **Related:** [[graph-schema]], [[results-log]]

---

## Problem

The professional-cohort classification task is dominated by text signal (0.82 test F1 text-only), leaving minimal headroom for graph features to demonstrate additive value. A new classification target is needed where language is informative but not dominant.

## Selection criteria

A good extraction target must satisfy four properties:

| # | Criterion | Definition |
|---|---|---|
| 1 | **Endogenous** | Naturally revealed when someone talks about work — not an incidental personal detail |
| 2 | **Graph-relevant** | There's a plausible mechanism for concept graph topology to differ across classes |
| 3 | **Moderate text predictability** | Text alone should achieve ~0.5-0.7 F1, leaving headroom for graphs (if text gets 0.9 or 0.2, the target is useless) |
| 4 | **Natural uncertainty** | "I don't know" / "uncertain" is a valid answer, not a measurement failure |

## Variables considered

### Career stage → SELECTED

| Criterion | Assessment |
|---|---|
| Endogenous | ✅ Strong. Career narrative ("30 years", "just started", "managing a team") is integral to how people describe their work |
| Graph-relevant | ✅ Plausible. Late-career professionals have denser, more hierarchically structured concept graphs reflecting accumulated expertise; early-career have sparser, more exploratory structures |
| Text predictability | Moderate. Temporal markers are lexical but career-stage inference from work descriptions is non-trivial |
| Natural uncertainty | ✅ Some transcripts lack clear temporal markers — "uncertain" is a legitimate class |

**Classes:** `early` / `mid` / `late` / `uncertain`

### AI adoption stage → SELECTED

| Criterion | Assessment |
|---|---|
| Endogenous | ✅ Strongest possible. The interviews ARE about AI usage — every transcript contains rich evidence |
| Graph-relevant | ✅ Strong. Power-users likely have more AI-concept nodes (Constructs about "workflow integration", Stances about "trust"), while novices have simpler AI-related subgraphs |
| Text predictability | Moderate. "I rarely use it" vs "I built a custom pipeline" is lexically distinct, but intermediate levels require inference |
| Natural uncertainty | ✅ "I've tried it a few times but not regularly" — ambiguous, legitimately uncertain |

**Classes:** `novice` / `experienced` / `power_user` / `uncertain`

### Gender — REJECTED

| Criterion | Assessment |
|---|---|
| Endogenous | ❌ Incidental. Gender references ("my wife", "as a woman in tech") occur in only a small fraction of transcripts |
| Graph-relevant | ❌ No plausible mechanism for gender to shape concept graph topology |
| Text predictability | Very low. Most transcripts would be "uncertain" — not a useful classification task |
| Natural uncertainty | Forced. "Uncertain" would be the majority class, making the variable uninformative |

### Education level — NOT SELECTED (future candidate)

Credentials (PhD, MA, college dropout) are extractable but:
- Heavily correlated with cohort (most scientists have PhDs, most workforce don't)
- Redundant with career stage for many professionals
- Could be revisited if career stage + AI adoption prove insufficient

## Implementation plan

1. **Test phase** (bead `lxk`): DeepSeek vs Agnes on 20 random transcripts
2. **Full extraction** (follow-up): Selected model on all 1,250 transcripts
3. **New classification targets**: Evaluate text-only, graph-only, and text+graph on career stage and AI adoption
4. **Compare**: Does graph modality add more value on these targets than on professional cohort?

## Prompt design principles

- **Evidence-first**: Every label must cite specific quotes from the transcript
- **Default to uncertain**: Better to miss data than to guess
- **No stereotyping**: Career stage from experience markers, not age guesswork; AI adoption from described behavior, not job title heuristics
- **Reasoning trace**: The model's reasoning is part of the output, not just a debugging aid
