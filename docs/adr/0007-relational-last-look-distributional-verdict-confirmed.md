# 0007 — Relational last-look: distributional verdict empirically confirmed

- **Status:** Accepted
- **Date:** 2026-06-15
- **Supersedes:** none (confirms and hardens ADR-0006; does not change its verdict)
- **Related:** ADR-0006 (Phase 6 distributional verdict), ADR-0004 (edge-signal validity),
  ADR-0005 (ambivalence adjudication); `.claude/context/results-log.md`
  ("Relational last-look" section); epic `graph-modality-yki` (closed here) with children
  `graph-modality-8gf` (B1), `graph-modality-744` (B2), `graph-modality-621` (RGCN)

## Context

ADR-0006 concluded the graph modality's signal is **distributional node-attribute** (stance
valence / concept-label semantics), **not relational/topological**, and in its *Alternatives
considered* (#4) **rejected pursuing a stronger relational encoder** on the evidence then
available (topology arms at chance; `label_bag` without edges beating the GIN with edges).

Before accepting that verdict as final, we ran a small, **pre-scoped** last look — explicitly
guarded against p-hacking (the relational hypothesis was favoured; macro-F1 variance is
dominated by the `high` class, n=55). Three tests under the frozen 10-seed CI protocol (seeds
42–51, class-weighted, `stance_ambivalence`, v4_think corpus; PASS = CI excludes 0 AND mean
Δ macro-F1 ≥ +0.01), with a decision gate: only escalate to the expensive RGCN encoder if the
two cheap tests came back null.

1. **B1 — endpoint-aware conflict edges.** Ambivalence is definitionally "holding conflicting
   stances," so `CONFLICTS_WITH` is the one relation theory privileges. A bag-of-types
   histogram already counts conflict edges (and GINEConv failed to beat it), so B1's
   contribution was an **endpoint-valence-aware** feature a histogram cannot represent:
   conflict edges joining two `Construct`s of **opposite dominant stance valence** (valence
   reached via `Stance --EXPRESSED_VIA--> Construct`).
2. **B2 — lexical-vs-conceptual control.** Re-ran the `label_bag` probe on **free-text** vs
   **canonical** labels to localise whether the distributional label signal is surface wording
   or concept identity.
3. **RGCN (Tier C, gated).** A per-relation-weighted encoder (`RGCNConv`, 6 relations) — a
   strictly stronger edge-type test than GINEConv's additive edge features — added as a 4th arm
   to the existing `h_edge.py` structure-only ladder so untyped/typed/rgcn train under identical
   capacity, objective, protocol, and splits.

## Decision

Record the last-look result and act on it. **The relational hypothesis remains rejected; the
ADR-0006 distributional verdict stands, now empirically defended at the ceiling of reasonable
encoders.**

1. **B1 — relational structure adds nothing net of circularity.** Against the full 30-dim
   distributional baseline (LR, paired): adding the genuinely-relational opposite-valence
   feature gives Δ = +0.0098, CI [+0.001, +0.019] — **FAIL** (below the +0.01 bar) — and is
   **worse** than the plain conflict *count* (relational − count = −0.018, CI excludes 0). The
   only qualifying gain came from the raw `n_conflict_edges` count (+0.0275), which is
   **near-circular**: a count of conflict edges predicting *ambivalence* is nearly measuring
   the label twice. Strip the circularity and relational structure carries no qualifying signal.
2. **B2 — the signal is conceptual, not lexical.** Free-text (0.403) ≈ canonical (0.398);
   paired Δ = +0.0046, CI [−0.0094, +0.0187] (spans 0). Canonicalisation neither helps nor
   hurts → the distributional node-attribute signal is **synonym-invariant / conceptual**, not
   surface-wording. This **strengthens** the ADR-0006 positive (stats > text is not a lexical
   artifact).
3. **RGCN — per-relation weighting does not recover topology.** The decisive contrast
   **rgcn − untyped = +0.0234, CI [−0.0057, +0.0526] — FAIL** (CI spans 0). All edge-axis arms
   hover near chance (no_edges 0.278 / untyped 0.274 / typed 0.274 / rgcn 0.298; chance 0.269).
   RGCN beats GINEConv marginally (b: +0.0241, CI excludes 0) but that reflects GINEConv
   underperforming, not relational structure being recovered — the relational claim is contrast
   (a), and it fails. Crucially RGCN has **more** capacity than untyped (239,488 vs 166,784
   params) yet still fails to beat it: a loss-with-more-capacity cannot be attacked as
   under-parameterisation.

**Three independent angles (endpoint-aware features, a lexical control, and a per-relation GNN)
all confirm: the graph modality's signal is distributional node-attribute, not relational.**

## Consequences

- **ADR-0006 is confirmed, not amended.** The defensible claim is unchanged: *LLM-extracted
  concept-graph node-attribute statistics carry predictive signal flat text embeddings do not
  recover, on a lexically-non-obvious, independently-labelled target.* The last look adds two
  hardenings: the positive is **conceptual not lexical** (B2), and the relational null survives
  the strongest fair encoder (RGCN) and the theory-privileged endpoint-aware feature (B1).
- **Epic `graph-modality-yki` closed.** B1, B2 closed on their results; RGCN executed and closed.
- **The relational/edge thread is closed for this project.** ADR-0006 alternative #4 ("pursue a
  stronger relational encoder") was rejected on argument; it has now been **falsified
  empirically**. No further GNN-architecture search is warranted on this target/corpus — it
  would be p-hacking a triangulated null.
- **Methodological note (recorded for reuse).** B1's first pass *passed* against a thin 9-dim
  valence baseline (+0.042) and would have "confirmed" the relational hypothesis; the
  per-feature decomposition against the full distributional baseline revealed the gain was a
  near-circular conflict-count artifact. The lesson: test incremental features against the
  **strongest** distributional baseline, and decompose to check no single feature is circular
  with the label.

## Alternatives considered

1. **Accept ADR-0006 without the last look.** Defensible, but the relational hypothesis had not
   been given its *strongest fair* instrument (per-relation weights) or its
   *theory-privileged* feature (endpoint-aware conflict valence). Running them converts "rejected
   on argument" into "rejected on evidence."
2. **Open-ended architecture search (attention readout, positional encodings, graph
   transformers).** Rejected as p-hacking: with two cheap tests and a more-capable RGCN all
   null, additional capacity has nothing to capture; the decision gate exists precisely to stop
   here.
3. **Treat B1's first PASS as a reopen.** Rejected: it was a thin-baseline + circularity
   artifact (see methodological note); the corrective contrast against full stats is the valid
   test and it fails.

## Open risks

- Unchanged from ADR-0006: single dataset (Anthropic Interviewer), single graph extractor
  (DeepSeek); `high` class small (n=55). The one sensible future check remains the
  cross-extractor replication of **stats > text** (not topology) — a fresh budget-capped bead
  (≤30 transcripts), not a reopen of the relational thread.
- The edge-axis arms sit at/near chance under structure-only features by construction; this is
  the fair isolation of the edge axis, but it also means the corpus offers little structural
  variance for *any* encoder — itself consistent with (and possibly a partial mechanism for)
  the distributional verdict.
