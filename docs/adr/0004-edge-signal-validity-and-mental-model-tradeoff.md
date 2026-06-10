# 0004 — Edge-signal validity and the mental-model trade-off

- **Status:** Accepted (decision framing; implementation pending epic `graph-modality-ztg`)
- **Date:** 2026-06-10
- **Supersedes:** none
- **Related:** ADR-0001 (extraction model), ADR-0003 (target-agnostic encoders); `docs/METHOD_REVIEW.md`; results-log Method-Review Phases 1–2

## Context

Method-Review Phases 1–2 produced two hard findings:

1. Under the frozen 10-seed CI protocol, the only surviving text-vs-fusion gain on the
   primary target (`ai_adoption`) is **text+stats** (Δ=+0.0142, CI excludes 0). The GIN
   `text+graph` delta does **not** survive (CI includes 0).
2. The disentanglement ablation showed **`full_gin < label_bag`** on both targets — the
   GIN embedding is strictly worse than a training-free mean-pool of its own input labels.

These looked like a death sentence for the "graph is a distinct modality" claim. But they
were measured on **v3 graphs**, where edge type is a **deterministic function of the
endpoint node-type pair** (`SERVES`≡Construct→Value, `EXPRESSED_VIA`≡Stance→Construct,
etc.). On such graphs a message-passing GNN already sees node types, so edge types carry
almost no independent information. **The edge/relational hypothesis was therefore never
fairly tested.** v4 adds `SUBSUMES` (Value→Value) and `IMPLIES` (Construct→Construct),
the first relations that create same-type edges with non-deterministic type, making the
hypothesis testable.

## The central trade-off

Adding relations to break type-determinism creates a tension between two goods:

- **Ontological flexibility / testability** — richer relations break determinism and let
  the edge hypothesis be tested (and potentially carry stronger signal).
- **Construct validity** — does the richer graph still measure the *respondent's* mental
  model, or does it measure the *LLM's* response to a more permissive prompt?

Two readings of v4 are observationally identical at design time:

- **Reading A (valid):** v3 was too rigid and artificially flattened relational structure
  that genuinely exists in cognition; the new relations *recover* suppressed signal.
- **Reading B (invalid):** the relations were added *because the hypothesis needs
  variance* — tuning the instrument to produce the desired result (circular).

After a run of null results, motivated reasoning pulls toward B while narrating A.

## Decision

1. **v4's direction is defensible because the new relations are theory-grounded, not
   manufactured.** `SUBSUMES` ← means-end-chain / laddering theory (Gutman 1982) posits
   value hierarchies; `IMPLIES` ← Kelly's Personal Construct Theory (1955) treats
   constructs as an implicative network. The theoretical warrant predates and is
   independent of the hypothesis test — that is the line between Reading A and Reading B.

2. **The validity risk relocates to the new relations and v4 under-protects them.**
   `SUBSUMES`/`IMPLIES` are the most *inferential, least-grounded* relations (no
   respondent states them explicitly; the LLM infers them from its world-model) yet v4
   requires only a free-text rationale, no grounding span. The worst failure mode is
   **cohort-stereotyped confabulation**: the LLM imposes "scientist-shaped" implication
   networks given scientist framing, leaking the label through model prior rather than
   respondent cognition (review concerns #1 + #8 fused) — producing a *positive but
   invalid* edge result.

3. **Mitigations (required before any edge-signal claim):**
   - Make `SUBSUMES`/`IMPLIES` **auditable**: their rationale must cite the grounding
     spans of *both* endpoint nodes and state what licenses the inference. (Does not
     require respondents to state meta-relations — avoids collapsing back to v3
     determinism — but creates an audit trail.)
   - Tag every edge `grounding: explicit | inferred` so downstream can **weight or ablate**
     inferential edges and measure how much signal rides on confabulation-prone relations.
   - Post-extraction confound check: v4 removes the CSM cap and encourages exhaustiveness,
     inflating graph size. Regress node/edge count on transcript length per cohort — if
     size tracks length and length tracks cohort, that is a confound, not cognition.

4. **The mental-model question is resolved empirically, not a priori.** Rather than guess
   which ontology ("minimal/distributional" vs "moderate/grounded-relational" vs
   "maximal/theoretical") carries the strongest signal, extract **once** with the richest
   *auditable* schema, tagging groundedness and relation type as first-class dimensions,
   then run **nested ablations** under the 10-seed protocol: distributional-only →
   + grounded edges → + inferred edges. The rung at which signal stops increasing answers
   the mental-model question as a measured result. Each ontology revision costs a full
   re-extraction, so iterate downstream, not by re-extracting.

## Consequences

- v4 must be amended (auditable inferred edges + `grounding` tag) before the full 1,250
  extraction; validator and `graph-schema.md` updated in lockstep.
- `graph_dataset.py` must preserve span text **and** canonical label **and** type so the
  node-content axis is ablatable downstream (mirrors the edge-groundedness axis).
- The defensible claim ladder is unchanged from the review: (1) LLM-extracted graphs carry
  signal → (2) not reducible to pooled label semantics → (3) robust across extractors →
  (4) survives vocabulary masking. Each ablation/replication buys one rung.
- Honest expectation: text may simply win on these targets; the graph's realistic prize is
  the complementary GRAPH-UNIQUE sliver, not dominance.

## Alternatives considered

- **Rebuild the ontology from scratch.** Rejected: v3's four entity types
  (Construct/Value/Stance/CSM) are grounded, extractable, and theory-backed — the node
  layer was never the problem. The defect was the *relational* layer (over-deterministic
  and groundedness-blind), which amended-v4 addresses.
- **Maximal theoretical ontology** (full Kellyan implication grid + laddered hierarchies).
  Rejected: on ~10-turn transcripts current models fill richness with plausible priors;
  maximises confabulation and measures the model, not the person.
- **Stay on v3 / declare topology dead.** Rejected as premature: the v3 null is an artifact
  of edge-type determinism, not evidence against relational signal.
