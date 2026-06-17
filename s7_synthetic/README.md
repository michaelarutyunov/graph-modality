# s7_synthetic — synthetic pipeline demonstration

A small, self-contained demo that illustrates **when a graph representation can and cannot
help over text**, respecting the data-processing inequality (DPI). It is the constructive
coda to the real-data work in `s1_data … s6_notebooks/`.

> **This is a controlled fiction, not evidence about the world.** It uses numeric channels
> only — **no prose is generated, no LLM is involved**, and no claim is made that these
> encodings resemble real interviews. It illustrates the *logic* of the hypothesis; it does
> not test it on reality. That would require a human study (independent elicitation), which is
> not on the roadmap.

## Background — why this exists

The real-data phase (`s1–s6`) asked whether concept graphs extracted from interview
transcripts are a *distinct modality* that beats text. The verdict (ADR-0006/0007): the graph
modality's signal is **distributional (node-attribute / label semantics), not relational /
topological** — edges added nothing; a no-edge label-bag beat the GIN.

The deeper reason that result was almost inevitable is the **data-processing inequality**: if a
graph `G` is *derived from* text `T` (`G = f(T)`), and the label is also a function of `T`, then
`I(G; Y) ≤ I(T; Y)` — the graph cannot carry more information about the target than the text it
came from. So "graph beats text" is false *in principle* in that setup. The only defensible
escapes are:

1. **Extractability** — the graph can package the *same* information in a form a model extracts
   more easily (better inductive bias / sample efficiency). No new information; easier to use.
2. **Independent elicitation** — if the graph is gathered *independently* of the text (a
   separate task, not derived from it), it can carry genuinely non-redundant information.

This demo makes both escapes concrete.

## What `synthetic_demo.py` does

It samples synthetic latent DAGs, observes them through numeric "text" and "graph" channels,
generates a label from the latent structure, and runs two demonstrations.

### Panel A — Extractability (rig-proof)

- The **"text" representation is the full flattened adjacency** of the latent DAG — it contains
  *strictly more* information than the handful of graph-topology features (which are a lossy
  function of the adjacency). So the graph view **cannot smuggle in extra signal**; the DPI is
  respected by construction.
- The label depends on a **non-linear structural quantity** (longest-path length).
- A **weak learner** (logistic regression) does much better with the pre-computed topology
  features than with the raw adjacency — it cannot reconstruct longest-path from raw edges.
- A **strong learner** (gradient boosting) closes most of the gap on the raw adjacency —
  *empirically confirming the information was there all along.*
- **Conclusion:** the graph's advantage is **extraction, not information.**

### Panel B — Independent elicitation (the open question, as a knob)

- Two **noisy observations** of the same latent DAG: a "text" view and a "graph" view.
- A complementarity knob **`c`** interpolates from `c=0` (the graph observes *exactly* what the
  text observed — a graph *derived* from text, the DPI case, **no gain**) to `c=1` (the graph is
  an **independent** observation that can recover structure the text dropped).
- We plot the **incremental** predictive value of adding the graph over text alone, vs `c`.
- **Reality sits somewhere on this axis and we do not know where** — that is the empirical
  question a human study would answer. The plot characterises the *dependence*; it does not
  assert a point. The effect is deliberately left at its honest (modest, uncertain) size.

## Run

```bash
uv run python s7_synthetic/synthetic_demo.py
```

Outputs `s7_synthetic/synthetic_demo.png` (two-panel figure) and a numeric summary to stdout.
Runs in ~1 minute (CPU only). Knobs are the constants at the top of the script
(`N_NODES`, `EDGE_RHO`, `A_SAMPLES`, `B_P_OBSERVE`, `B_REPEATS`, `B_GRID`). Reproducible via
`SEED = 42`.

## Representative results (seed 42)

| Panel A — test AUROC | text (full adjacency) | graph (topology features) | gap |
|---|---|---|---|
| weak (logistic regression) | 0.814 | 0.880 | **+0.066** (extractability advantage) |
| strong (gradient boosting) | 0.855 | 0.875 | **+0.020** (gap closes — info was in text) |

Panel B: incremental AUROC of `text+graph` over `text` rises from **+0.000 at `c=0`** (derived
graph, DPI — no gain, by construction) to **~+0.03 at `c=1`** (independent elicitation), with a
wide confidence band — the honest, modest size of the effect.

## How this connects to the rest of the repo

- `s1_data … s6_notebooks/` — the real-data attempt (the receipts). **Frozen.**
- `docs/adr/0006`, `0007` — the negative verdict (distributional, not relational).
- `docs/adr/0008` — the round-3 delegation-boundary ontology (designed to give the hypothesis
  its fairest shot; never extracted — the project pivoted here instead).
- **this folder** — the constructive illustration of *what would be required* for a graph to
  add value, and the rigorous reason a text-derived graph cannot.
