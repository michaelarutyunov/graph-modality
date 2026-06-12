# Design: Attitudinal-Ambivalence Target (replaces `ai_adoption` as primary endogenous target)

**Date:** 2026-06-12
**Status:** Approved (design), pending implementation plan
**Related:** epic `graph-modality-3ee` (Phase 2.6), `docs/METHOD_REVIEW.md`, `.claude/context/results-log.md` (Phase 2.6 sections)

---

## Motivation

The Phase 2.6 results showed:

- **Null-ladder FAIL on ai_adoption** — typed relational wiring (GINEConv) does not beat a
  bag-of-types histogram (Δ=+0.0028, CI=[−0.0172, +0.0228]).
- **structure_only PASS** (+0.2022 over chance) — but this is suspect, because `ai_adoption`
  is produced by **DeepSeek**, the *same* model that produces the graphs. "Graph predicts
  target" is partly an extractor agreeing with itself (circularity; METHOD_REVIEW concern #1,
  sharpened because the label — not just the features — is LLM-extracted).

Deeper problem with the existing targets: they are **lexically obvious**. `cohort` leaks
profession vocabulary into human responses (un-strippable), so SBERT wins by keyword-matching
and there is no honest headroom for graph structure. `ai_adoption` is a mushy ~50/50 binary
(49.8% integrated / 48.2% tool_user) collapsed from a near-empty 4-point scale, labeled by a
single zero-shot DeepSeek call (κ≈0.69 vs Agnes on n=20, all disagreements one-directional).

**Target-selection criterion (from the user):** endogenous to the AI-usage topic, but
**lexically non-obvious** — two people using nearly the same vocabulary should be able to land
on opposite labels, so surface text has no shortcut and graph topology gets a fair test.

## Construct choice: attitudinal ambivalence

Among candidate constructs, **attitudinal ambivalence toward AI** best fits the criterion:
ambivalence is *structural, not lexical*. Two people can both discuss AI's benefits and
drawbacks with identical vocabulary; what differs is whether those views are held in
**unresolved tension** (ambivalent) or **resolved into a coherent stance** (one-sided). That
difference lives in *how* pro and con concepts connect — exactly what `CONFLICTS_WITH` edges
and mixed/ambivalent Stance valence encode. (Risk/agency attitude was rejected as too lexical;
cognitive style as too exogenous/personal.)

### Signal-existence validation (completed 2026-06-12)

Probe over all 1,250 v4_think graphs (graph-derived proxy, used **only** to confirm the
construct varies — never as the label):

| Signal | Spread | Verdict |
|---|---|---|
| `CONFLICTS_WITH` edges | 29.6% of graphs ≥1, median 0, max 3 | too sparse alone |
| `ambivalent` valence | 49 nodes in entire corpus | negligible |
| `mixed`/`ambivalent` stance per graph | 47% have ≥1 | moderate |
| both pos AND neg stance present | **97% of graphs** | "any ambivalence?" is **degenerate** |
| **neg+mixed fraction of stances (balance)** | mean 0.47, median 0.50, **sd 0.17** | **well-spread continuous signal** |

**Conclusion:** the construct that *varies* is the **balance/tilt** of stance (one-sided →
mixed → conflicted), not the presence of conflict. This is naturally **ordinal**, validating a
low/med/high scheme over a binary. The graph-derived proxy is a signal-existence check only;
the actual label is sourced independently (see §2) to avoid circularity.

---

## §1 — The construct definition

Ordinal `stance_ambivalence ∈ {low, med, high}`: the degree of **unresolved tension** in the
person's overall stance toward AI in their professional work. Judged **holistically from the
person's own words** — never by counting graph features.

- **low** — coherent one-sided stance (clearly pro *or* clearly skeptical); little internal
  tension.
- **med** — leans one way but genuinely acknowledges the other side; some mixed feeling.
- **high** — genuinely torn: substantial pro *and* con held in unresolved tension
  ("it depends", "I love it but…", oscillation between enthusiasm and concern).

The rubric is **anchored a priori** (from this definition + the proxy distribution's natural
cut points), because there is no hand-labeled dev set to calibrate from (see §3). Worked
example quotes for each level are written into the prompt.

## §2 — Label production (breaks circularity)

Two labelers, **neither is DeepSeek** (which produced the graphs):

- **Agnes** (free) — full corpus (1,250).
- **Haiku 4.5** — full corpus. This is a ~200-token label call, not graph extraction; full
  corpus is affordable. (The ≤30-transcript budget cap in memory `phase3-extraction-budget`
  applies to expensive Sonnet/Claude **graph** re-extraction, not short label calls.)

Both labelers:
- read **raw human turns only** — never the graph, never each other's output;
- use the same anchored ordinal rubric (§1) with worked example quotes;
- output `{transcript_id, stance_ambivalence: {label, reasoning, quotes}}` — same schema shape
  as `s2_extraction/prompts/demographics_v1.txt`, so it reuses the
  `demographics_extractor.py` pattern (new prompt file `ambivalence_v1.txt`, second backend
  for Haiku).

Truncation: keep the existing 8000-char human-text cap (affects 4.3% of transcripts); document
as a minor one-directional bias.

## §3 — Consensus & adjudication

- **Agree** → accept the agreed label.
- **Disagree** → **user adjudicates** (expected minority). Tooling surfaces disagreements as a
  worklist showing both models' label + reasoning + quotes side by side.
- Report observed **agreement rate + Cohen's κ** as the label-quality number.
- **No pre-committed gold set** (user decision). Known limitation: agreement cases are trusted
  without human review, so *correlated* bias between the two models is invisible.
- **Optional tripwire** (recommended, ~5 min): spot-check ~10 random *agreement* cases for
  shared calibration bias. Marked optional in the plan.

## §4 — Integration into the experiment

Same frozen-embedding harness; new target column `stance_ambivalence`. Re-run the existing
ladder with ambivalence as the label:

- **Text arm** (SBERT, human turns) → does flat text struggle? (hypothesis: yes — no lexical
  shortcut).
- **structure_only / GINEConv** → does graph predict it?
- **Null-ladder** (11-dim bag-of-types histogram vs GINEConv 128-dim), 10-seed frozen CI,
  same PASS criterion as `graph-modality-7sa` (CI excludes 0 AND mean Δ ≥ +0.01) — unchanged.

Chance baseline: report majority-class macro-F1 for the 3-level ordinal (recompute once label
distribution is known).

## §5 — Honest expectation-setting

This target does **not** guarantee a topology win. What it fixes is the *text confound*: it
gives the graph modality a fair fight by removing the keyword shortcut that let text dominate
`cohort`/`ai_adoption`. The genuine topology claim **still rests on GINEConv > histogram** on
this target. Three outcomes, all publishable:

1. Text struggles, **GINEConv > histogram** → the clean "structurally distinct modality"
   result.
2. Text struggles, histogram ≈ GINEConv → signal is *distributional stance-balance* (valence
   counts), not wiring — honest, and stronger than the ai_adoption story.
3. Text predicts it well → ambivalence was not lexically hidden after all; learned cheaply,
   stop.

## Disposition of prior targets

- `ai_adoption` — demoted to **secondary** (retained for comparison; its structure_only PASS is
  flagged as possibly circular).
- `cohort` — remains secondary/sanity (interviewer-protocol confounded).
- `stance_ambivalence` — **new primary** endogenous target.

## Out of scope (YAGNI)

- No multi-model **graph** re-extraction (separate, budget-capped line).
- No new canonicalisation (labels are transcript-level, not node-level).
- No human gold set beyond adjudicated disagreements + optional 10-case tripwire.
- Risk-attitude / cognitive-style targets not pursued.
