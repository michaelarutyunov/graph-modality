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
def __(json, glob, Path):
    """Load all canonical graphs into memory."""
    _repo_root = Path(__file__).parent.parent
    _paths = sorted(glob.glob(str(_repo_root / "data/graphs/canonical/*.json")))
    graphs = []
    for _p in _paths:
        with open(_p) as _f:
            graphs.append(json.load(_f))
    len(graphs)
    return graphs,


@app.cell
def __(graphs):
    """Build a flat DataFrame of per-graph metadata."""
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
            if _constructs else 0.0
        )
        rows.append({
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
        })

    df_meta = pl.DataFrame(rows)
    df_meta
    return df_meta, pl


@app.cell
def __(df_meta, mo, pl):
    """Corpus overview header."""
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
def __(df_meta, mo, pl):
    """Section 1: Cohort summary statistics."""
    summary = df_meta.group_by("split").agg([
        pl.col("n_nodes").mean().round(2).alias("avg_nodes"),
        pl.col("n_edges").mean().round(2).alias("avg_edges"),
        pl.col("n_construct").mean().round(2).alias("avg_constructs"),
        pl.col("n_value").mean().round(2).alias("avg_values"),
        pl.col("n_stance").mean().round(2).alias("avg_stances"),
        pl.col("n_csm").mean().round(2).alias("avg_csms"),
        pl.col("bipolarity_score").mean().round(3).alias("avg_bipolarity"),
        pl.col("violations").mean().round(3).alias("avg_violations"),
        pl.len().alias("count"),
    ]).sort("split")

    mo.md("## 1. Cohort Summary Statistics")
    mo.md("Per-cohort means across all graph metrics:")
    summary
    return summary,


@app.cell
def __(df_meta, mo, pl):
    """Section 1b: Node type proportion bar chart."""
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
def __(graphs, mo, pl):
    """Section 2: Stance valence analysis."""
    _valence_rows = []
    for _g in graphs:
        for _n in _g.get("nodes", []):
            if _n.get("type") == "Stance":
                _valence_rows.append({
                    "split": _g["split"],
                    "valence": _n.get("valence", "unknown"),
                })

    df_valence = pl.DataFrame(_valence_rows)
    valence_summary = df_valence.group_by(["split", "valence"]).agg(
        pl.len().alias("count")
    ).with_columns(
        (pl.col("count") / pl.col("count").sum().over("split")).round(3).alias("proportion")
    ).sort(["split", "valence"])

    mo.md("## 2. Stance Valence Analysis")
    mo.md("Distribution of stance valences across cohorts (preview of H2):")
    valence_summary
    return df_valence, valence_summary


@app.cell
def __(df_valence, mo):
    """Valence proportion bar chart."""
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
            _subset = df_valence.filter(
                (df_valence["split"] == _s) & (df_valence["valence"] == _v)
            )
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
def __(graphs, mo, pl):
    """Section 3: Construct bipolarity completeness."""
    _bip_rows = []
    for _g in graphs:
        _constructs = [_n for _n in _g.get("nodes", []) if _n.get("type") == "Construct"]
        if _constructs:
            _complete = sum(1 for _c in _constructs if _c.get("bipolarity_complete"))
            _bip_rows.append({
                "split": _g["split"],
                "bipolarity_complete_frac": _complete / len(_constructs),
                "n_constructs": len(_constructs),
            })

    df_bip = pl.DataFrame(_bip_rows)
    bip_summary = df_bip.group_by("split").agg([
        pl.col("bipolarity_complete_frac").mean().round(3).alias("mean_complete"),
        pl.col("n_constructs").mean().round(2).alias("avg_constructs"),
    ]).sort("split")

    mo.md("## 3. Construct Bipolarity Completeness")
    mo.md("Fraction of constructs with both poles defined (preview of H3):")
    bip_summary
    return bip_summary, df_bip


@app.cell
def __(df_bip, mo):
    """Bipolarity box plot."""
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(8, 5))
    _data = [
        df_bip.filter(df_bip["split"] == _s)["bipolarity_complete_frac"].to_list()
        for _s in ["workforce", "creatives", "scientists"]
    ]
    _bp = _ax.boxplot(_data, tick_labels=["workforce", "creatives", "scientists"], patch_artist=True)
    for _patch in _bp["boxes"]:
        _patch.set_facecolor("lightblue")
    _ax.set_ylabel("Bipolarity Completeness Fraction")
    _ax.set_title("Construct Bipolarity by Cohort")
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def __(mo):
    """Section 4: Degree and centrality analysis."""
    mo.md("## 4. Degree and Centrality Analysis")
    mo.md("Computing network metrics for all graphs (this may take a moment)...")
    return


@app.cell
def __(graphs, mo):
    """Compute degree and centrality metrics."""
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

        _degree_rows.append({
            "split": _g["split"],
            "mean_degree": _np.mean(_degrees) if _degrees else 0,
            "max_degree": max(_degrees) if _degrees else 0,
            "mean_value_degree": _np.mean(_value_degrees) if _value_degrees else 0,
            "max_betweenness": _max_btw,
            "mean_betweenness": _mean_btw,
        })

    df_degree = _pl.DataFrame(_degree_rows)
    df_degree
    return df_degree,


@app.cell
def __(df_degree, mo):
    """Degree distribution histograms."""
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
def __(df_degree, mo):
    """Value node degree (hub-and-spoke preview of H1)."""
    import polars as _pl
    vdeg_summary = df_degree.group_by("split").agg([
        _pl.col("mean_value_degree").mean().round(2).alias("mean_value_degree"),
        _pl.col("mean_betweenness").mean().round(4).alias("mean_betweenness"),
    ]).sort("split")

    mo.md("### Value Node Degree (Hub-and-Spoke Preview)")
    mo.md("Higher value degree suggests constructs cluster around terminal values:")
    vdeg_summary
    return vdeg_summary,


@app.cell
def __(mo):
    """Section 5: Relation type analysis."""
    mo.md("## 5. Relation Type Analysis")
    return


@app.cell
def __(graphs, mo, pl):
    """Compute relation type proportions."""
    _rel_rows = []
    for _g in graphs:
        _edges = _g.get("edges", [])
        _rel_counts = {"SERVES": 0, "EXPRESSED_VIA": 0, "MODULATED_BY": 0, "CONFLICTS_WITH": 0}
        for _e in _edges:
            _r = _e.get("relation", "")
            if _r in _rel_counts:
                _rel_counts[_r] += 1
        _total = len(_edges)
        _rel_rows.append({
            "split": _g["split"],
            **{_k: _v / _total if _total > 0 else 0 for _k, _v in _rel_counts.items()},
            "total_edges": _total,
            "has_conflicts": _rel_counts["CONFLICTS_WITH"] > 0,
        })

    df_rel = pl.DataFrame(_rel_rows)
    rel_summary = df_rel.group_by("split").agg([
        pl.col("SERVES").mean().round(3).alias("SERVES"),
        pl.col("EXPRESSED_VIA").mean().round(3).alias("EXPRESSED_VIA"),
        pl.col("MODULATED_BY").mean().round(3).alias("MODULATED_BY"),
        pl.col("CONFLICTS_WITH").mean().round(3).alias("CONFLICTS_WITH"),
        pl.col("has_conflicts").mean().round(3).alias("conflict_prevalence"),
    ]).sort("split")

    mo.md("Proportion of each relation type by cohort:")
    rel_summary
    return df_rel, rel_summary


@app.cell
def __(rel_summary, mo, pl):
    """Relation type stacked bar chart."""
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
def __(mo):
    """Section 6: Interactive graph viewer."""
    mo.md("## 6. Interactive Graph Viewer")
    mo.md("Select a transcript to visualize its concept graph:")
    return


@app.cell
def __(mo, graphs):
    """Transcript selector dropdown."""
    _options = {_g["transcript_id"]: _g["transcript_id"] for _g in graphs[:200]}
    _default = next(iter(_options.keys())) if _options else ""
    dropdown = mo.ui.dropdown(options=_options, label="Transcript ID", value=_default)
    dropdown
    return dropdown,


@app.cell
def __(dropdown, json, glob, mo, Path):
    """Load selected graph."""
    _repo_root = Path(__file__).parent.parent
    _selected_id = dropdown.value
    selected_graph = None
    for _p in glob.glob(str(_repo_root / "data/graphs/canonical/*.json")):
        with open(_p) as _f:
            _g = json.load(_f)
        if _g["transcript_id"] == _selected_id:
            selected_graph = _g
            break

    if selected_graph:
        mo.md(f"**Selected:** {_selected_id} ({selected_graph['split']})")
        mo.md(f"Nodes: {len(selected_graph['nodes'])}, Edges: {len(selected_graph['edges'])}")
    else:
        mo.md(f"Graph {_selected_id} not found")
    return selected_graph,


@app.cell
def __(selected_graph, mo):
    """Render selected graph as NetworkX spring layout."""
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


@app.cell
def __(selected_graph, mo):
    """Show node/edge tables for selected graph."""
    import polars as _pl

    if selected_graph is None:
        edges_df = _pl.DataFrame()
    else:
        nodes_df = _pl.DataFrame(selected_graph.get("nodes", []))
        edges_df = _pl.DataFrame(selected_graph.get("edges", []))
        mo.md("### Nodes")
        nodes_df.select(["id", "type", "label"])

    return edges_df,


@app.cell
def __(edges_df, mo):
    """Show edge table."""
    mo.md("### Edges")
    edges_df
    return


@app.cell
def __(mo):
    """Footer."""
    mo.md("---")
    mo.md("*Notebook 02 — Graph Exploration. Data: canonical concept graphs (n=1,250).*")
    return


if __name__ == "__main__":
    app.run()
