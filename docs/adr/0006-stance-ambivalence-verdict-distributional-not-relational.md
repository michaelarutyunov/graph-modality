# 0006 — Phase 6 verdict: graph signal is distributional, not relational

- **Status:** Accepted
- **Date:** 2026-06-14
- **Supersedes:** none (closes the Phase 6 line; supersedes the v3-corpus Phase 6 numbers)
- **Related:** ADR-0003 (target-agnostic encoders), ADR-0004 (edge-signal validity), ADR-0005
  (ambivalence adjudication); `.claude/context/ambivalence-target.md`;
  `.claude/context/results-log.md` (Phase 6 + Phase 6 — v4 sections); epics
  `graph-modality-vpm` (Phase 6), `graph-modality-6k9` (Phase 7 replication, closed here)

## Context

Phase 6 retested the project's founding hypothesis — that concept-graph structure is a
**structurally distinct modality** carrying signal text cannot recover — after the v3 results
were found to be confounded:

- **Circularity (METHOD_REVIEW #1):** the old primary target `ai_adoption` was labelled by
  DeepSeek, the same model that produced the graphs → "graph predicts target" was partly the
  extractor agreeing with itself.
- **Lexical obviousness:** `cohort` leaks profession vocabulary into text (SBERT wins by
  keyword), leaving no honest headroom for graph structure.
- **Edge determinism (ADR-0004):** v3 made edge type a deterministic function of endpoint node
  types, so the relational/edge hypothesis was never fairly tested.

The response (ADR-0005 + design `.claude/context/ambivalence-target.md`): a new endogenous,
**lexically-non-obvious** primary target `stance_ambivalence` (low/med/high), **independently
labelled** (Agnes + Haiku + Kimi adjudication — none is DeepSeek), tested under the frozen
10-seed CI protocol with class-weighted loss (imbalance: med=843/low=352/high=55).

Two hypotheses:
- **H_fusion:** fusion(text + graph) > max(single modality).
- **H_edge:** an edge-typed encoder beats a node-only one (2-D ablation: edge axis × feature
  axis), on the v4 edge ontology (SUBSUMES/IMPLIES) built to break v3's determinism.

All tests were re-run on a single re-canonicalised **v4 corpus** (P6.6) after the first pass was
found to mix v3 (stats/GIN) and v4 (edge axis) graphs.

## Decision

Record the Phase 6 verdict and act on it. On the v4 corpus, with circularity and lexical
confounds controlled (full results in `results-log.md` "Phase 6 — v4"):

1. **Modality-distinctness — SUPPORTED.** Graph stats (0.433) beat text (0.367) by +0.066, 95%
   CI [+0.016, +0.116] (excludes 0). On a target with no lexical shortcut, distributional graph
   **node-attributes** carry signal text cannot recover.
2. **Complementarity (H_fusion) — NOT SUPPORTED.** No fusion arm significantly beats the best
   single modality (stats): text+stats − stats = +0.020, CI [−0.010, +0.051] (spans 0). Graph
   **subsumes** text rather than complementing it.
3. **Relational/edge hypothesis (H_edge) — REJECTED.** Edge presence *and* type add nothing
   (edge axis at chance: typed ≈ untyped ≈ histogram ≈ 0.27). `label_bag` (no edges, 0.402)
   **beats** `full_gin` (with edges, 0.285) by 0.117; `structure_only` (type+degree) sits at
   chance. The learned graph embedding's signal is **label/attribute semantics, not wiring**
   (METHOD_REVIEW #4 vindicated). The v4 edge ontology did not rescue the relational claim.

**The signal carried by the graph modality is distributional node-attribute (stance valence /
concept-label semantics), not relational/topological structure.**

## Consequences

- **Claim restated at the supported strength.** Defensible: *LLM-extracted concept-graph
  node-attribute statistics carry predictive signal that flat text embeddings do not recover,
  on a lexically-non-obvious, independently-labelled target.* NOT defensible: that graph
  **topology/wiring** adds information, or that graph and text are **complementary**.
- **Phase 7 replication epic `graph-modality-6k9` is closed as NOT-APPLICABLE.** It was scoped
  to replicate the **topology delta** across extractors via the v4 null-ladder. There is no
  topology delta — H_edge is dead on two corpora — so there is nothing to replicate. The
  budget-capped multi-model extraction is not spent.
- **Residual caveat / the one sensible future check.** The surviving claim (stats > text) still
  rests on graphs that DeepSeek extracted; stance valence is DeepSeek's judgement
  (METHOD_REVIEW #1 applies to the *features*, even though the *label* is now independent). The
  meaningful remaining validation is therefore **not** topology replication but a cross-extractor
  check that *the distributional stats > text result survives a different graph extractor*. This
  is a new, narrower question; if pursued, file it as a fresh budget-capped bead (≤30 transcripts
  per memory `phase3-extraction-budget`), not a reopen of `6k9`.
- **v4 is the corpus of record.** `canonical_map_v4.json` is locked; encoders read v4. v3
  artifacts remain immutable for provenance.

## Alternatives considered

1. **Replicate topology across extractors anyway (run `6k9` as written).** Rejected: replicating
   a null is not informative, and METHOD_REVIEW's own sequencing says replication is "the big
   spend — only commit once the deltas are real." The topology delta is zero.
2. **Reopen/repurpose `6k9` to replicate stats > text now.** Deferred, not done: it is a
   different question from the epic's charter; cleaner to close `6k9` honestly and file a fresh,
   narrowly-scoped bead if the project continues.
3. **Keep the v3-corpus numbers.** Rejected: they mixed v3 (stats/GIN) and v4 (edge) graphs and
   carried an unresolved structure_only-vs-chance contradiction. The v4 redo resolved it
   (structure_only at chance on both corpora).
4. **Pursue a stronger relational encoder (deeper GNN, attention over edges).** Rejected on the
   evidence: with structure-only features the topology arms are at chance and `label_bag`
   without edges already beats the GIN with edges — more relational capacity has nothing to
   capture on this target.

## Open risks

- Single dataset (Anthropic Interviewer), single extractor (DeepSeek) for the graphs. The
  distributional result is not yet shown to generalise across datasets or extractors.
- `high` class is small (n=55; 8 in test) — per-class power is limited; the verdict rests on
  macro-F1 over 10 seeds with class weighting, not on `high` alone.
- The distributional signal is concentrated in stance-valence features that are themselves
  LLM-extracted; the residual-circularity check above is the way to close this.
