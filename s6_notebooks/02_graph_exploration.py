# ruff: noqa

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    """Import marimo and expose `mo` for all downstream cells.

    Only `mo` is threaded as a cell output — every stdlib or
    third-party module is imported locally in each cell with an
    underscore prefix to avoid Marimo's "redefines variables" error.
    """
    import marimo as mo

    return (mo,)


@app.cell
def _():
    """Load all canonical concept graphs into memory.

    Reads every JSON file from data/graphs/canonical/ (produced by
    canonicalisation/apply_canonical.py).  Each file is one transcript's
    concept graph with canonicalised node labels.  The `graphs` list
    is returned so downstream cells can aggregate statistics, build
    DataFrames, and render interactive views.
    """
    import json as _json

    from pathlib import Path as _Path

    _repo_root = _Path(__file__).parent.parent
    _paths = sorted(_repo_root.glob("s1_data/graphs/canonical/*.json"))
    graphs = []
    for _p in _paths:
        with open(_p) as _f:
            graphs.append(_json.load(_f))
    len(graphs)
    return (graphs,)


@app.cell
def _(graphs):
    """Build a flat per-graph metadata DataFrame.

    Iterates over all loaded graphs and computes:
    - Node counts by type (Construct, Value, Stance, CognitiveStyleMarker)
    - Bipolarity score: fraction of Construct nodes with both poles
      defined (bipolarity_complete=True counts 1.0, missing counts 0.5)
    - Total edges and validation violations

    Returns both the DataFrame (`df_meta`) and the `pl` module so
    downstream cells can use Polars without re-importing.
    """
    import polars as pl

    rows = []
    for _g in graphs:
        _nodes = _g.get("nodes", [])
        _edges = _g.get("edges", [])
        _type_counts = {"Construct": 0, "Value": 0, "Stance": 0, "CognitiveStyleMarker": 0}
        for _n in _nodes:
            _t = _n.get("type", "")
            if _t in _type_counts:
                _type_counts[_t] += 1
        _constructs = [_n for _n in _nodes if _n.get("type") == "Construct"]
        _bipolarity = (
            sum(1.0 if _c.get("bipolarity_complete") else 0.5 for _c in _constructs)
            / len(_constructs)
            if _constructs
            else 0.0
        )
        rows.append(
            {
                "transcript_id": _g["transcript_id"],
                "split": _g["split"],
                "n_nodes": len(_nodes),
                "n_edges": len(_edges),
                "n_construct": _type_counts["Construct"],
                "n_value": _type_counts["Value"],
                "n_stance": _type_counts["Stance"],
                "n_csm": _type_counts["CognitiveStyleMarker"],
                "bipolarity_score": _bipolarity,
                "violations": len(_g.get("validation_violations", [])),
            }
        )

    df_meta = pl.DataFrame(rows)
    df_meta
    return df_meta, pl


@app.cell
def _(df_meta, mo, pl):
    """Render the notebook title and corpus overview table.

    Displays cohort sizes (workforce, creatives, scientists) and
    aggregate graph statistics (mean nodes/edges) as a Markdown table.
    Uses `pl.col("split")` filters to count per-cohort rows.
    """
    mo.md(f"""
    # Notebook 02 — Graph Exploration

    ## Corpus Overview

    | Metric | Value |
    |--------|-------|
    | Total graphs | {df_meta.height} |
    | Workforce | {df_meta.filter(pl.col("split") == "workforce").height} |
    | Creatives | {df_meta.filter(pl.col("split") == "creatives").height} |
    | Scientists | {df_meta.filter(pl.col("split") == "scientists").height} |
    | Mean nodes | {df_meta["n_nodes"].mean():.1f} |
    | Mean edges | {df_meta["n_edges"].mean():.1f} |
    """)
    return


@app.cell
def _(df_meta, mo, pl):
    """Section 1: Per-cohort summary statistics table.

    Groups df_meta by `split` (cohort) and computes means for all
    numeric columns (nodes, edges, type counts, bipolarity score,
    violations).  The Polars `group_by().agg()` chain produces one
    row per cohort.  Displayed via mo.vstack with a heading and the
    DataFrame (Marimo renders DataFrames as interactive HTML tables).
    """
    summary = (
        df_meta.group_by("split")
        .agg(
            [
                pl.col("n_nodes").mean().round(2).alias("avg_nodes"),
                pl.col("n_edges").mean().round(2).alias("avg_edges"),
                pl.col("n_construct").mean().round(2).alias("avg_constructs"),
                pl.col("n_value").mean().round(2).alias("avg_values"),
                pl.col("n_stance").mean().round(2).alias("avg_stances"),
                pl.col("n_csm").mean().round(2).alias("avg_csms"),
                pl.col("bipolarity_score").mean().round(3).alias("avg_bipolarity"),
                pl.col("violations").mean().round(3).alias("avg_violations"),
                pl.len().alias("count"),
            ]
        )
        .sort("split")
    )

    mo.vstack(
        [
            mo.md("## 1. Cohort Summary Statistics"),
            mo.md("Per-cohort means across all graph metrics:"),
            summary,
        ]
    )
    return


@app.cell
def _(df_meta, pl):
    """Section 1b: Grouped bar chart of node type counts by cohort.

    Uses matplotlib to create a grouped bar chart.  For each of the
    four node types (Construct, Value, Stance, CSM), filters df_meta
    by cohort and computes the mean count.  Each type gets its own
    bar offset within each cohort group.  Marimo auto-captures
    matplotlib figures via plt.show().
    """
    import matplotlib.pyplot as _plt
    import numpy as _np

    _fig, _ax = _plt.subplots(figsize=(10, 5))
    _splits = ["workforce", "creatives", "scientists"]
    _x = _np.arange(len(_splits))
    _width = 0.2

    for _i, _col in enumerate(["n_construct", "n_value", "n_stance", "n_csm"]):
        _means = [df_meta.filter(pl.col("split") == _s)[_col].mean() for _s in _splits]
        _label = _col.replace("n_", "").replace("csm", "CSM").title()
        _ax.bar(_x + _i * _width, _means, _width, label=_label)

    _ax.set_ylabel("Mean count per graph")
    _ax.set_title("Node Type Counts by Cohort")
    _ax.set_xticks(_x + _width * 1.5)
    _ax.set_xticklabels(_splits)
    _ax.legend()
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(graphs, mo, pl):
    """Section 2: Stance valence analysis — extraction and summary table.

    Flattens all Stance nodes across all graphs into rows of
    (split, valence).  Then groups by (split, valence) to count
    occurrences and computes the proportion within each cohort using
    Polars' `.over("split")` window function.  Returns df_valence
    for the downstream bar chart cell.
    """
    _valence_rows = []
    for _g in graphs:
        for _n in _g.get("nodes", []):
            if _n.get("type") == "Stance":
                _valence_rows.append(
                    {
                        "split": _g["split"],
                        "valence": _n.get("valence", "unknown"),
                    }
                )

    df_valence = pl.DataFrame(_valence_rows)
    valence_summary = (
        df_valence.group_by(["split", "valence"])
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over("split")).round(3).alias("proportion")
        )
        .sort(["split", "valence"])
    )

    mo.vstack(
        [
            mo.md("## 2. Stance Valence Analysis"),
            mo.md("Distribution of stance valences across cohorts (preview of H2):"),
            valence_summary,
        ]
    )
    return (df_valence,)


@app.cell
def _(df_valence):
    """Valence proportion bar chart.

    For each valence type (positive, negative, mixed, ambivalent),
    computes the proportion within each cohort by filtering
    df_valence and dividing subset height by total.  Plots grouped
    bars with matplotlib.
    """
    import matplotlib.pyplot as _plt
    import numpy as _np

    _splits = ["workforce", "creatives", "scientists"]
    _valences = ["positive", "negative", "mixed", "ambivalent"]
    _x = _np.arange(len(_splits))
    _width = 0.2

    _fig, _ax = _plt.subplots(figsize=(10, 5))
    for _i, _v in enumerate(_valences):
        _props = []
        for _s in _splits:
            _subset = df_valence.filter((df_valence["split"] == _s) & (df_valence["valence"] == _v))
            _total = df_valence.filter(df_valence["split"] == _s).height
            _props.append(_subset.height / _total if _total > 0 else 0)
        _ax.bar(_x + _i * _width, _props, _width, label=_v.title())

    _ax.set_ylabel("Proportion")
    _ax.set_title("Stance Valence Distribution by Cohort")
    _ax.set_xticks(_x + _width * 1.5)
    _ax.set_xticklabels(_splits)
    _ax.legend()
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(graphs, mo, pl):
    """Section 3: Construct bipolarity completeness — extraction and table.

    For each graph, finds Construct nodes and computes the fraction
    that have bipolarity_complete=True.  A score of 1.0 means every
    construct has both positive and negative poles defined.  Groups
    by cohort to produce mean completeness and mean construct count.
    Returns df_bip for the downstream box plot.
    """
    _bip_rows = []
    for _g in graphs:
        _constructs = [_n for _n in _g.get("nodes", []) if _n.get("type") == "Construct"]
        if _constructs:
            _complete = sum(1 for _c in _constructs if _c.get("bipolarity_complete"))
            _bip_rows.append(
                {
                    "split": _g["split"],
                    "bipolarity_complete_frac": _complete / len(_constructs),
                    "n_constructs": len(_constructs),
                }
            )

    df_bip = pl.DataFrame(_bip_rows)
    bip_summary = (
        df_bip.group_by("split")
        .agg(
            [
                pl.col("bipolarity_complete_frac").mean().round(3).alias("mean_complete"),
                pl.col("n_constructs").mean().round(2).alias("avg_constructs"),
            ]
        )
        .sort("split")
    )

    mo.vstack(
        [
            mo.md("## 3. Construct Bipolarity Completeness"),
            mo.md("Fraction of constructs with both poles defined (preview of H3):"),
            bip_summary,
        ]
    )
    return (df_bip,)


@app.cell
def _(df_bip):
    """Bipolarity completeness box plot by cohort.

    Uses matplotlib's boxplot to show the distribution of
    bipolarity_complete_frac per cohort.  This reveals the spread
    (not just the mean) — whether most graphs are fully bipolar or
    only partially.
    """
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(8, 5))
    _data = [
        df_bip.filter(df_bip["split"] == _s)["bipolarity_complete_frac"].to_list()
        for _s in ["workforce", "creatives", "scientists"]
    ]
    _bp = _ax.boxplot(
        _data, tick_labels=["workforce", "creatives", "scientists"], patch_artist=True
    )
    for _patch in _bp["boxes"]:
        _patch.set_facecolor("lightblue")
    _ax.set_ylabel("Bipolarity Completeness Fraction")
    _ax.set_title("Construct Bipolarity by Cohort")
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(mo):
    """Section 4 heading — warns that network metric computation is slow.

    Building NetworkX DiGraphs for all 1,250 transcripts and computing
    betweenness centrality takes several seconds.
    """
    mo.md(
        "## 4. Degree and Centrality Analysis\n\nComputing network metrics for all graphs (this may take a moment)..."
    )
    return


@app.cell
def _(graphs):
    """Compute per-graph degree and centrality metrics using NetworkX.

    For each graph:
    1. Builds a NetworkX DiGraph from nodes and edges
    2. Computes mean/max degree and Value-node degree (hub-and-spoke
       proxy — Value nodes with high degree indicate constructs
       clustering around terminal values)
    3. Computes betweenness centrality (mean and max) with a try/except
       fallback for any graph topology errors

    Returns df_degree for the downstream histogram and summary cells.
    """
    import networkx as _nx
    import numpy as _np
    import polars as _pl

    _degree_rows = []
    for _g in graphs:
        _G = _nx.DiGraph()
        for _n in _g.get("nodes", []):
            _G.add_node(_n["id"], **_n)
        for _e in _g.get("edges", []):
            _G.add_edge(_e["source"], _e["target"])

        _degrees = [_d for _, _d in _G.degree()]
        _value_nodes = [_n["id"] for _n in _g.get("nodes", []) if _n.get("type") == "Value"]
        _value_degrees = [_G.degree(_v) for _v in _value_nodes] if _value_nodes else [0]

        try:
            _btw = _nx.betweenness_centrality(_G)
            _max_btw = max(_btw.values()) if _btw else 0
            _mean_btw = _np.mean(list(_btw.values())) if _btw else 0
        except Exception:
            _max_btw = _mean_btw = 0

        _degree_rows.append(
            {
                "split": _g["split"],
                "mean_degree": _np.mean(_degrees) if _degrees else 0,
                "max_degree": max(_degrees) if _degrees else 0,
                "mean_value_degree": _np.mean(_value_degrees) if _value_degrees else 0,
                "max_betweenness": _max_btw,
                "mean_betweenness": _mean_btw,
            }
        )

    df_degree = _pl.DataFrame(_degree_rows)
    df_degree
    return (df_degree,)


@app.cell
def _(df_degree):
    """Degree distribution histograms — one subplot per cohort.

    Filters df_degree by split and plots a 20-bin histogram of
    mean_degree for each cohort side by side.  The subplot title
    includes the cohort's overall mean for quick comparison.
    """
    import matplotlib.pyplot as _plt

    _fig, _axes = _plt.subplots(1, 3, figsize=(15, 4))
    _splits = ["workforce", "creatives", "scientists"]
    for _ax, _s in zip(_axes, _splits, strict=False):
        _data = df_degree.filter(df_degree["split"] == _s)["mean_degree"].to_list()
        _ax.hist(_data, bins=20, edgecolor="black", alpha=0.7)
        _ax.set_title(f"{_s}\nmean={sum(_data) / len(_data):.1f}")
        _ax.set_xlabel("Mean degree")
        _ax.set_ylabel("Count")
    _plt.suptitle("Degree Distribution by Cohort", y=1.02)
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(df_degree, mo):
    """Value node degree summary — hub-and-spoke hypothesis preview (H1).

    Groups by cohort and averages the mean_value_degree and
    mean_betweenness.  High value degree suggests the graph has a
    hub-and-spoke topology where many constructs connect to a few
    terminal values (a prediction of H1 from CHARTER.md).
    """
    import polars as _pl

    vdeg_summary = (
        df_degree.group_by("split")
        .agg(
            [
                _pl.col("mean_value_degree").mean().round(2).alias("mean_value_degree"),
                _pl.col("mean_betweenness").mean().round(4).alias("mean_betweenness"),
            ]
        )
        .sort("split")
    )

    mo.vstack(
        [
            mo.md("### Value Node Degree (Hub-and-Spoke Preview)"),
            mo.md("Higher value degree suggests constructs cluster around terminal values:"),
            vdeg_summary,
        ]
    )
    return


@app.cell
def _(mo):
    """Section 5 heading — relation type analysis."""
    mo.md("## 5. Relation Type Analysis")
    return


@app.cell
def _(graphs, mo, pl):
    """Compute per-graph relation type proportions.

    Counts edges of each relation type (SERVES, EXPRESSED_VIA,
    MODULATED_BY, CONFLICTS_WITH) per graph, then converts to
    proportions (count / total_edges).  Aggregates by cohort to
    show mean proportions and conflict prevalence (fraction of
    graphs that contain at least one CONFLICTS_WITH edge).
    """
    _rel_rows = []
    for _g in graphs:
        _edges = _g.get("edges", [])
        _rel_counts = {"SERVES": 0, "EXPRESSED_VIA": 0, "MODULATED_BY": 0, "CONFLICTS_WITH": 0}
        for _e in _edges:
            _r = _e.get("relation", "")
            if _r in _rel_counts:
                _rel_counts[_r] += 1
        _total = len(_edges)
        _rel_rows.append(
            {
                "split": _g["split"],
                **{_k: _v / _total if _total > 0 else 0 for _k, _v in _rel_counts.items()},
                "total_edges": _total,
                "has_conflicts": _rel_counts["CONFLICTS_WITH"] > 0,
            }
        )

    df_rel = pl.DataFrame(_rel_rows)
    rel_summary = (
        df_rel.group_by("split")
        .agg(
            [
                pl.col("SERVES").mean().round(3).alias("SERVES"),
                pl.col("EXPRESSED_VIA").mean().round(3).alias("EXPRESSED_VIA"),
                pl.col("MODULATED_BY").mean().round(3).alias("MODULATED_BY"),
                pl.col("CONFLICTS_WITH").mean().round(3).alias("CONFLICTS_WITH"),
                pl.col("has_conflicts").mean().round(3).alias("conflict_prevalence"),
            ]
        )
        .sort("split")
    )

    mo.vstack(
        [
            mo.md("Proportion of each relation type by cohort:"),
            rel_summary,
        ]
    )
    return (rel_summary,)


@app.cell
def _(rel_summary):
    """Relation type grouped bar chart.

    Reads mean proportions from rel_summary and plots a grouped bar
    for each of the four relation types across the three cohorts.
    """
    import matplotlib.pyplot as _plt
    import numpy as _np

    _fig, _ax = _plt.subplots(figsize=(10, 5))
    _splits = ["workforce", "creatives", "scientists"]
    _rel_types = ["SERVES", "EXPRESSED_VIA", "MODULATED_BY", "CONFLICTS_WITH"]
    _x = _np.arange(len(_splits))
    _width = 0.2

    for _i, _rt in enumerate(_rel_types):
        _vals = [rel_summary.filter(rel_summary["split"] == _s)[_rt].item() for _s in _splits]
        _ax.bar(_x + _i * _width, _vals, _width, label=_rt)

    _ax.set_ylabel("Proportion of edges")
    _ax.set_title("Relation Type Distribution by Cohort")
    _ax.set_xticks(_x + _width * 1.5)
    _ax.set_xticklabels(_splits)
    _ax.legend()
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(mo):
    """Section 6 heading — interactive graph viewer with dropdown.

    Users can select a transcript and see its concept graph rendered
    as a NetworkX spring layout with node/edge detail tables below.
    """
    mo.md("## 6. Interactive Graph Viewer\n\nSelect a transcript to visualize its concept graph:")
    return


@app.cell
def _(graphs, mo):
    """Build the transcript selector dropdown.

    Creates a mo.ui.dropdown with the first 200 transcript IDs.
    The selected value drives the downstream "load selected graph"
    and "render graph" cells reactively.
    """
    _options = {_g["transcript_id"]: _g["transcript_id"] for _g in graphs[:200]}
    _default = next(iter(_options.keys())) if _options else ""
    dropdown = mo.ui.dropdown(options=_options, label="Transcript ID", value=_default)
    dropdown
    return (dropdown,)


@app.cell
def _(dropdown, mo):
    """Find and display the selected transcript's graph metadata.

    Linear-scans all canonical JSON files for a matching transcript_id.
    When found, displays the cohort label and node/edge counts via
    mo.vstack.  Returns `selected_graph` so downstream cells can
    render the graph visualization and detail tables.
    """
    import json as _json

    from pathlib import Path as _Path

    _repo_root = _Path(__file__).parent.parent
    _selected_id = dropdown.value
    selected_graph = None
    for _p in _repo_root.glob("s1_data/graphs/canonical/*.json"):
        with open(_p) as _f:
            _g = _json.load(_f)
        if _g["transcript_id"] == _selected_id:
            selected_graph = _g
            break

    if selected_graph:
        mo.vstack(
            [
                mo.md(f"**Selected:** {_selected_id} ({selected_graph['split']})"),
                mo.md(
                    f"Nodes: {len(selected_graph['nodes'])}, Edges: {len(selected_graph['edges'])}"
                ),
            ]
        )
    else:
        mo.md(f"Graph {_selected_id} not found")
    return (selected_graph,)


@app.cell
def _(mo, selected_graph):
    """Render the selected graph as a NetworkX spring-layout visualization.

    Builds a DiGraph, assigns colors by node type (Construct=skyblue,
    Value=lightgreen, Stance=salmon, CSM=plum), truncates labels to
    20 chars for readability, and draws with spring_layout (k=2 for
    spacing, seed=42 for reproducibility).  Includes a color legend.
    """
    import networkx as _nx
    import matplotlib.pyplot as _plt

    if selected_graph is None:
        mo.md("No graph selected")
    else:
        _G = _nx.DiGraph()
        _color_map = {
            "Construct": "skyblue",
            "Value": "lightgreen",
            "Stance": "salmon",
            "CognitiveStyleMarker": "plum",
        }
        _node_colors = []
        _labels = {}

        for _n in selected_graph.get("nodes", []):
            _G.add_node(_n["id"])
            _node_colors.append(_color_map.get(_n.get("type", ""), "gray"))
            _labels[_n["id"]] = _n.get("label", _n["id"])[:20]

        for _e in selected_graph.get("edges", []):
            _G.add_edge(_e["source"], _e["target"], relation=_e.get("relation", ""))

        _fig, _ax = _plt.subplots(figsize=(12, 10))
        _pos = _nx.spring_layout(_G, k=2, iterations=50, seed=42)
        _nx.draw_networkx_nodes(_G, _pos, node_color=_node_colors, node_size=800, ax=_ax)
        _nx.draw_networkx_labels(_G, _pos, _labels, font_size=8, ax=_ax)
        _nx.draw_networkx_edges(_G, _pos, edge_color="gray", arrows=True, arrowsize=15, ax=_ax)

        from matplotlib.patches import Patch

        _legend_elements = [Patch(facecolor=_c, label=_t) for _t, _c in _color_map.items()]
        _ax.legend(handles=_legend_elements, loc="upper right")
        _ax.set_title(f"Concept Graph: {selected_graph['transcript_id']}")
        _ax.axis("off")
        _plt.tight_layout()
        _plt.show()
    return


@app.cell
def _(mo, selected_graph):
    """Display node and edge detail tables for the selected graph.

    Builds two Polars DataFrames from the graph's nodes and edges
    arrays and stacks them vertically with mo.vstack.  Nodes table
    shows id, type, label; edges table shows all edge attributes
    (source, target, relation).
    """
    import polars as _pl

    if selected_graph is None:
        mo.md("No graph selected")
    else:
        nodes_df = _pl.DataFrame(selected_graph.get("nodes", []))
        edges_df = _pl.DataFrame(selected_graph.get("edges", []))
        mo.vstack(
            [
                mo.md("### Nodes"),
                nodes_df.select(["id", "type", "label"]),
                mo.md("### Edges"),
                edges_df,
            ]
        )
    return


@app.cell
def _(mo):
    """Footer — notebook identity and data source."""
    mo.md("---\n\n*Notebook 02 — Graph Exploration. Data: canonical concept graphs (n=1,250).*")
    return


if __name__ == "__main__":
    app.run()
