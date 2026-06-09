# ruff: noqa

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    """Import marimo and expose `mo` for all downstream cells.

    This is the single shared import cell — only `mo` is returned as a
    cell output because it's the one object every other cell needs.
    Stdlib modules (json, pathlib) are imported locally in each cell
    with an underscore prefix (_json, _Path) to avoid Marimo's
    "redefines variables" error when multiple cells import the same
    module name.
    """
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    """Render the notebook title and model-comparison section header.

    Checks whether structured rubric scores exist on disk
    (extraction/model_comparison/results/).  If present, shows a
    "loading" message; otherwise explains that only the qualitative
    validation report is available.  Uses mo.vstack to combine the
    heading and body into one displayed output.
    """
    from pathlib import Path as _Path

    _results_dir = _Path("s2_extraction/model_comparison/results")
    _has_structured = _results_dir.exists() and any(_results_dir.iterdir())

    _heading = mo.md("""
    # Notebook 01 — Extraction Review

    ## Model Comparison
    """)
    _body = (
        mo.md("Structured rubric scores found. Loading...")
        if _has_structured
        else mo.md("""
        **Note:** Structured rubric comparison data is not available.
        The model comparison was run during Phase 1, but quantitative
        rubric scores were not persisted to disk. Below is a summary
        from the qualitative validation report.
        """)
    )
    mo.vstack([_heading, _body])
    return


@app.cell
def _(mo):
    """Display a static summary table of the 3-model extraction comparison.

    Pinned from the Phase 1 qualitative validation: Claude, DeepSeek,
    and Agnes were each tested on 3 transcripts.  DeepSeek (via the
    OpenAI-compatible endpoint + JSON mode) was chosen for the full
    1,250-transcript run.
    """
    mo.md("""
    ### Comparison Summary (3 transcripts × 3 models)

    | Model | Endpoint | Outcome |
    |-------|----------|---------|
    | Claude (claude-sonnet-4-6) | Anthropic API | ✅ 3/3 valid graphs |
    | DeepSeek (deepseek-chat) | OpenAI-compatible + JSON mode | ✅ 3/3 valid graphs |
    | Agnes (agnes-2.0-flash) | OpenAI-compatible | ✅ 3/3 valid graphs |

    **Key finding:** DeepSeek via the OpenAI-compatible endpoint with
    `response_format={"type": "json_object"}` produced reliable structured
    output, while the Anthropic-compatible endpoint (with forced thinking
    mode) failed completely (0/10 valid graphs due to JSON truncation).

    **Scale extraction:** DeepSeek (deepseek-chat) was selected for the
    full 1,250-transcript run based on cost and reliability.
    """)
    return


@app.cell
def _(mo):
    """Link to the full validation report on disk.

    Points users to extraction/model_comparison/validation_report.md
    for per-transcript inspection notes from the 3-model comparison.
    """
    mo.md("""
    ### Validation Report

    See `extraction/model_comparison/validation_report.md` for the full
    qualitative analysis including per-transcript graph inspection notes.
    """)
    return


@app.cell
def _():
    """Load every free-text concept graph into memory.

    Reads all JSON files from data/graphs/free_text/ (one per transcript,
    produced by extraction/extractor.py).  Each file contains nodes,
    edges, transcript_id, split (cohort), and validation_violations.
    The `graphs` list is returned for downstream cells to aggregate
    and inspect.  Uses `import json as _json` (underscore-prefixed)
    so Marimo treats the import as cell-private.
    """
    import json as _json

    from pathlib import Path as _Path

    _repo_root = _Path(__file__).parent.parent
    _paths = sorted(_repo_root.glob("s1_data/graphs/free_text/*.json"))
    graphs = []
    for _p in _paths:
        with open(_p) as _f:
            graphs.append(_json.load(_f))
    len(graphs)
    return (graphs,)


@app.cell
def _(graphs, mo):
    """Compute corpus-level summary statistics and display them as a table.

    Iterates over all loaded graphs, counting nodes by type (Construct,
    Value, Stance, CognitiveStyleMarker) and edges per graph.  Builds a
    Polars DataFrame with per-graph metrics, then formats aggregate
    stats (total graphs, mean nodes/edges, violation rate) into a
    Markdown table via mo.vstack.
    """
    import polars as _pl

    _rows = []
    for _g in graphs:
        _nodes = _g.get("nodes", [])
        _edges = _g.get("edges", [])
        _type_counts = {"Construct": 0, "Value": 0, "Stance": 0, "CognitiveStyleMarker": 0}
        for _n in _nodes:
            _t = _n.get("type", "")
            if _t in _type_counts:
                _type_counts[_t] += 1
        _rows.append(
            {
                "transcript_id": _g["transcript_id"],
                "split": _g["split"],
                "n_nodes": len(_nodes),
                "n_edges": len(_edges),
                "n_construct": _type_counts["Construct"],
                "n_value": _type_counts["Value"],
                "n_stance": _type_counts["Stance"],
                "n_csm": _type_counts["CognitiveStyleMarker"],
                "violations": len(_g.get("validation_violations", [])),
            }
        )

    df_meta = _pl.DataFrame(_rows)

    mo.vstack(
        [
            mo.md("## Corpus-Level Summary"),
            mo.md(f"""
        | Metric | Value |
        |--------|-------|
        | Total graphs | {df_meta.height} |
        | Mean nodes | {df_meta["n_nodes"].mean():.1f} |
        | Mean edges | {df_meta["n_edges"].mean():.1f} |
        | Graphs with violations | {df_meta.filter(df_meta["violations"] > 0).height} |
        | Violation rate | {df_meta["violations"].mean() * 100:.1f}% |
        """),
        ]
    )
    return (df_meta,)


@app.cell
def _(df_meta):
    """Display per-cohort (workforce / creatives / scientists) breakdown.

    Groups df_meta by `split` and computes mean node/edge counts and
    node-type breakdowns per cohort.  The resulting Polars DataFrame
    is the last expression so Marimo renders it as an interactive table.
    """
    import polars as _pl

    _cohort = (
        df_meta.group_by("split")
        .agg(
            [
                _pl.col("n_nodes").mean().round(2).alias("avg_nodes"),
                _pl.col("n_edges").mean().round(2).alias("avg_edges"),
                _pl.col("n_construct").mean().round(2).alias("avg_constructs"),
                _pl.col("n_value").mean().round(2).alias("avg_values"),
                _pl.col("n_stance").mean().round(2).alias("avg_stances"),
                _pl.col("n_csm").mean().round(2).alias("avg_csms"),
                _pl.col("violations").sum().alias("total_violations"),
                _pl.len().alias("count"),
            ]
        )
        .sort("split")
    )
    _cohort
    return


@app.cell
def _(mo):
    """Section heading for the interactive graph inspector.

    Purely presentational — displays the section title and instructions
    for the dropdown-driven transcript inspection below.
    """
    mo.md(
        "## Interactive Graph Inspection\n\nSelect a transcript to inspect its extracted concept graph:"
    )
    return


@app.cell
def _(graphs, mo):
    """Build a dropdown selector for transcript IDs.

    Creates a mo.ui.dropdown populated with the first 200 transcript
    IDs from the loaded graphs.  The dropdown value is used by the
    downstream "load selected graph" cell to find and display that
    transcript's concept graph.
    """
    _options = {_g["transcript_id"]: _g["transcript_id"] for _g in graphs[:200]}
    _default = list(_options.keys())[0] if _options else ""
    dropdown = mo.ui.dropdown(options=_options, label="Transcript ID", value=_default)
    dropdown
    return (dropdown,)


@app.cell
def _(dropdown, mo):
    """Find and display the graph for the selected transcript.

    Scans all JSON files in data/graphs/free_text/ for a matching
    transcript_id (linear scan — stops at first match).  When found,
    displays the transcript's cohort, node/edge counts, and any
    validation violations using mo.vstack.  The `selected_graph`
    dict is returned so downstream cells can render its nodes, edges,
    and grounding spans.
    """
    import json as _json

    from pathlib import Path as _Path

    _repo_root = _Path(__file__).parent.parent
    _selected_id = dropdown.value
    selected_graph = None
    for _p in _repo_root.glob("s1_data/graphs/free_text/*.json"):
        with open(_p) as _f:
            _g = _json.load(_f)
        if _g["transcript_id"] == _selected_id:
            selected_graph = _g
            break

    if selected_graph:
        _v = selected_graph.get("validation_violations", [])
        _violations_md = (
            mo.vstack([mo.md(f"⚠️ **Violations:** {len(_v)}")] + [mo.md(f"- {_vi}") for _vi in _v])
            if _v
            else mo.md("✅ No validation violations")
        )
        mo.vstack(
            [
                mo.md(f"**Selected:** {_selected_id} ({selected_graph['split']})"),
                mo.md(
                    f"Nodes: {len(selected_graph['nodes'])}, Edges: {len(selected_graph['edges'])}"
                ),
                _violations_md,
            ]
        )
    else:
        mo.md(f"Graph {_selected_id} not found")
    return (selected_graph,)


@app.cell
def _(mo, selected_graph):
    """Render the node table for the selected graph.

    Extracts the `nodes` array from selected_graph, builds a Polars
    DataFrame, and selects id, type, label (plus valence and
    bipolarity_complete if present).  Guarded by `selected_graph is
    not None` so it renders nothing until a transcript is chosen.
    """
    if selected_graph is not None:
        import polars as _pl

        _nodes = selected_graph.get("nodes", [])
        if _nodes:
            _df_nodes = _pl.DataFrame(_nodes)
            _cols = ["id", "type", "label"]
            if "valence" in _df_nodes.columns:
                _cols.append("valence")
            if "bipolarity_complete" in _df_nodes.columns:
                _cols.append("bipolarity_complete")
            mo.vstack([mo.md("### Nodes"), _df_nodes.select(_cols)])
        else:
            mo.md("No nodes in this graph")
    return


@app.cell
def _(mo, selected_graph):
    """Render the edge table for the selected graph.

    Extracts the `edges` array from selected_graph (each edge has
    source, target, relation) and displays it as a Polars DataFrame.
    Guarded by `selected_graph is not None`.
    """
    if selected_graph is not None:
        import polars as _pl

        _edges = selected_graph.get("edges", [])
        if _edges:
            mo.vstack([mo.md("### Edges"), _pl.DataFrame(_edges)])
        else:
            mo.md("No edges in this graph")
    return


@app.cell
def _(mo, selected_graph):
    """Render grounding spans for the selected graph's nodes.

    Each node may carry a `grounding_span` — the original transcript
    excerpt that the LLM cited when extracting that concept.  This
    cell collects all non-empty spans, truncates label/grounding text
    for readability, and displays them in a table.  Useful for
    verifying that extracted concepts are grounded in the source text.
    """
    if selected_graph is not None:
        _spans = []
        for _n in selected_graph.get("nodes", []):
            _span = _n.get("grounding_span", "")
            if _span:
                _spans.append(
                    {
                        "id": _n.get("id", ""),
                        "type": _n.get("type", ""),
                        "label": _n.get("label", "")[:40],
                        "grounding": _span[:100],
                    }
                )

        if _spans:
            import polars as _pl

            mo.vstack([mo.md("### Grounding Spans"), _pl.DataFrame(_spans)])
        else:
            mo.md("### Grounding Spans\n\nNo grounding spans available")
    return


@app.cell
def _(mo):
    """Footer — notebook identity and data source note."""
    mo.md("---\n\n*Notebook 01 — Extraction Review. Data: free-text concept graphs (n=1,250).*")
    return


if __name__ == "__main__":
    app.run()
