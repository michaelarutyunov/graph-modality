# ruff: noqa
import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import glob
    import json
    from pathlib import Path

    import marimo as mo
    return json, glob, Path, mo


@app.cell
def __(json, glob, Path, mo):
    """Check for structured model comparison results."""
    _results_dir = Path("extraction/model_comparison/results")
    _has_structured = _results_dir.exists() and any(_results_dir.iterdir())

    mo.md("""
    # Notebook 01 — Extraction Review

    ## Model Comparison
    """)

    if _has_structured:
        mo.md("Structured rubric scores found. Loading...")
    else:
        mo.md("""
        **Note:** Structured rubric comparison data is not available.
        The model comparison was run during Phase 1, but quantitative
        rubric scores were not persisted to disk. Below is a summary
        from the qualitative validation report.
        """)
    return _has_structured,


@app.cell
def __(mo):
    """Qualitative model comparison summary."""
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
def __(mo):
    """Validation report link."""
    mo.md("""
    ### Validation Report

    See `extraction/model_comparison/validation_report.md` for the full
    qualitative analysis including per-transcript graph inspection notes.
    """)
    return


@app.cell
def __(json, glob, Path):
    """Load all free-text graphs for inspection."""
    _repo_root = Path(__file__).parent.parent
    _paths = sorted(glob.glob(str(_repo_root / "data/graphs/free_text/*.json")))
    graphs = []
    for _p in _paths:
        with open(_p) as _f:
            graphs.append(json.load(_f))
    len(graphs)
    return graphs,


@app.cell
def __(graphs, mo):
    """Corpus-level summary statistics."""
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
        _rows.append({
            "transcript_id": _g["transcript_id"],
            "split": _g["split"],
            "n_nodes": len(_nodes),
            "n_edges": len(_edges),
            "n_construct": _type_counts["Construct"],
            "n_value": _type_counts["Value"],
            "n_stance": _type_counts["Stance"],
            "n_csm": _type_counts["CognitiveStyleMarker"],
            "violations": len(_g.get("validation_violations", [])),
        })

    df_meta = _pl.DataFrame(_rows)

    mo.md("## Corpus-Level Summary")
    mo.md(f"""
    | Metric | Value |
    |--------|-------|
    | Total graphs | {df_meta.height} |
    | Mean nodes | {df_meta['n_nodes'].mean():.1f} |
    | Mean edges | {df_meta['n_edges'].mean():.1f} |
    | Graphs with violations | {df_meta.filter(df_meta['violations'] > 0).height} |
    | Violation rate | {df_meta['violations'].mean() * 100:.1f}% |
    """)
    return df_meta,


@app.cell
def __(df_meta, mo):
    """Cohort distribution."""
    import polars as _pl
    _cohort = df_meta.group_by("split").agg([
        _pl.col("n_nodes").mean().round(2).alias("avg_nodes"),
        _pl.col("n_edges").mean().round(2).alias("avg_edges"),
        _pl.col("n_construct").mean().round(2).alias("avg_constructs"),
        _pl.col("n_value").mean().round(2).alias("avg_values"),
        _pl.col("n_stance").mean().round(2).alias("avg_stances"),
        _pl.col("n_csm").mean().round(2).alias("avg_csms"),
        _pl.col("violations").sum().alias("total_violations"),
        _pl.len().alias("count"),
    ]).sort("split")
    _cohort
    return


@app.cell
def __(mo):
    """Interactive graph inspection section."""
    mo.md("## Interactive Graph Inspection")
    mo.md("Select a transcript to inspect its extracted concept graph:")
    return


@app.cell
def __(mo, graphs):
    """Transcript selector."""
    _options = {_g["transcript_id"]: _g["transcript_id"] for _g in graphs[:200]}
    _default = list(_options.keys())[0] if _options else ""
    dropdown = mo.ui.dropdown(options=_options, label="Transcript ID", value=_default)
    dropdown
    return dropdown,


@app.cell
def __(dropdown, json, glob, mo, Path):
    """Load selected graph."""
    _repo_root = Path(__file__).parent.parent
    _selected_id = dropdown.value
    selected_graph = None
    for _p in glob.glob(str(_repo_root / "data/graphs/free_text/*.json")):
        with open(_p) as _f:
            _g = json.load(_f)
        if _g["transcript_id"] == _selected_id:
            selected_graph = _g
            break

    if selected_graph:
        mo.md(f"**Selected:** {_selected_id} ({selected_graph['split']})")
        mo.md(f"Nodes: {len(selected_graph['nodes'])}, Edges: {len(selected_graph['edges'])}")
        _v = selected_graph.get("validation_violations", [])
        if _v:
            mo.md(f"⚠️ **Violations:** {len(_v)}")
            for _vi in _v:
                mo.md(f"- {_vi}")
        else:
            mo.md("✅ No validation violations")
    else:
        mo.md(f"Graph {_selected_id} not found")
    return selected_graph,


@app.cell
def __(selected_graph, mo):
    """Node detail table."""
    if selected_graph is not None:
        import polars as _pl

        _nodes = selected_graph.get("nodes", [])
        if _nodes:
            _df_nodes = _pl.DataFrame(_nodes)
            mo.md("### Nodes")
            _cols = ["id", "type", "label"]
            if "valence" in _df_nodes.columns:
                _cols.append("valence")
            if "bipolarity_complete" in _df_nodes.columns:
                _cols.append("bipolarity_complete")
            _df_nodes.select(_cols)
        else:
            mo.md("No nodes in this graph")
    return


@app.cell
def __(selected_graph, mo):
    """Edge detail table."""
    if selected_graph is not None:
        import polars as _pl

        _edges = selected_graph.get("edges", [])
        if _edges:
            mo.md("### Edges")
            _pl.DataFrame(_edges)
        else:
            mo.md("No edges in this graph")
    return


@app.cell
def __(selected_graph, mo):
    """Grounding spans."""
    if selected_graph is not None:
        mo.md("### Grounding Spans")
        _spans = []
        for _n in selected_graph.get("nodes", []):
            _span = _n.get("grounding_span", "")
            if _span:
                _spans.append({
                    "id": _n.get("id", ""),
                    "type": _n.get("type", ""),
                    "label": _n.get("label", "")[:40],
                    "grounding": _span[:100],
                })

        if _spans:
            import polars as _pl
            _pl.DataFrame(_spans)
        else:
            mo.md("No grounding spans available")
    return


@app.cell
def __(mo):
    """Footer."""
    mo.md("---")
    mo.md("*Notebook 01 — Extraction Review. Data: free-text concept graphs (n=1,250).*")
    return


if __name__ == "__main__":
    app.run()
