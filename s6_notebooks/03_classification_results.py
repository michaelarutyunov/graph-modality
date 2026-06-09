# ruff: noqa

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    """Import marimo and expose `mo` for all downstream cells.

    Only `mo` is threaded as a cell output.  All stdlib modules are
    imported locally with underscore prefixes to avoid Marimo's
    "redefines variables" constraint.
    """
    import marimo as mo

    return (mo,)


@app.cell
def _():
    """Load all classification experiment results from the results/ directory.

    Reads three categories of JSON files:
    1. comparison_5routes_*.json — the latest 5-route comparison with
       test macro-F1 for each route
    2. route*_*.json — individual route results (multiple timestamps;
       later files overwrite earlier ones for the same route name, so
       the latest timestamp wins)
    3. demographic_classification_*.json — demographic sub-classification
       results (AI adoption, career stage)

    Returns `comparison`, `demo_results`, and `routes` for downstream
    cells to render tables and charts.
    """
    import json as _json

    from pathlib import Path as _Path
    _repo_root = _Path(__file__).parent.parent
    result_dir = _repo_root / "results"

    # Find the latest comparison files
    _comp_files = sorted(result_dir.glob("comparison_5routes_*.json"))
    _route_files = sorted(result_dir.glob("route*_*.json"))
    _demo_files = sorted(result_dir.glob("demographic_classification_*.json"))

    comparison = _json.loads(_comp_files[-1].read_text()) if _comp_files else {}
    demo_results = _json.loads(_demo_files[-1].read_text()) if _demo_files else {}

    # Load individual route results (latest timestamp)
    routes = {}
    for _p in _route_files:
        _data = _json.loads(_p.read_text())
        _name = _data.get("route", _p.stem)
        routes[_name] = _data

    len(routes), len(demo_results)
    return comparison, demo_results, routes


@app.cell
def _(mo):
    """Notebook title and description.

    Static header identifying this as the classification results
    dashboard for all experimental routes.
    """
    mo.md("""
    # Notebook 03 — Classification Results

    Interactive dashboard for all classification experiments.
    """)
    return


@app.cell
def _(comparison, mo):
    """Section 1: Route comparison table from the 5-route comparison file.

    Reads comparison["routes"] (a dict of {route_name: test_macro_f1})
    from the latest comparison JSON.  Builds a Polars DataFrame and
    displays it stacked with a section heading.  Falls back to a
    "no data" message if the comparison file is missing or empty.
    """
    import polars as _pl

    if comparison.get("routes"):
        _rows = []
        for _name, _f1 in comparison["routes"].items():
            _rows.append({"route": _name, "test_macro_f1": round(_f1, 4)})
        _df_comp = _pl.DataFrame(_rows)
        mo.vstack([mo.md("## 1. Route Comparison"), _df_comp])
    else:
        mo.md("## 1. Route Comparison\n\nNo comparison data found.")
    return


@app.cell
def _(mo, routes):
    """Section 2: Per-route detail table sorted by macro-F1.

    Extracts macro_f1 and n_samples from each route's result dict.
    Builds a Polars DataFrame sorted descending by macro-F1 so the
    best-performing route appears first.  Displayed via mo.vstack
    with a section heading.
    """
    import polars as _pl

    _detail_rows = []
    for _name, _data in routes.items():
        _detail_rows.append({
            "route": _name,
            "macro_f1": round(_data.get("macro_f1", 0), 4),
            "n_samples": _data.get("n_samples", "—"),
        })

    if _detail_rows:
        mo.vstack([
            mo.md("## 2. Per-Route Results"),
            _pl.DataFrame(_detail_rows).sort("macro_f1", descending=True),
        ])
    else:
        mo.md("## 2. Per-Route Results\n\nNo route data found.")
    return


@app.cell
def _(mo, routes):
    """Section 3: Per-class F1 grouped bar chart.

    Flattens per_class_f1 from each route into (route, class, f1)
    rows.  Plots a grouped bar chart with one bar per route for
    each of the three classes (workforce, creatives, scientists).
    Uses matplotlib — Marimo auto-captures the figure.  Falls back
    to a text message if no per-class data is available.
    """
    import matplotlib.pyplot as _plt
    import polars as _pl

    _class_rows = []
    for _name, _data in routes.items():
        _per_class = _data.get("per_class_f1", {})
        for _cls, _f1 in _per_class.items():
            _class_rows.append({
                "route": _name,
                "class": _cls,
                "f1": _f1,
            })

    if _class_rows:
        df_class = _pl.DataFrame(_class_rows)

        # Bar chart
        _fig, _ax = _plt.subplots(figsize=(12, 6))
        _classes = ["workforce", "creatives", "scientists"]
        _x = range(len(_classes))
        _width = 0.15

        # Get unique routes, limit to main ones for readability
        _route_names = sorted(set(df_class["route"].to_list()))
        for _i, _r in enumerate(_route_names):
            _f1s = [df_class.filter((df_class["route"] == _r) & (df_class["class"] == _c))["f1"].item() for _c in _classes]
            _ax.bar([_xi + _i * _width for _xi in _x], _f1s, _width, label=_r)

        _ax.set_ylabel("F1 Score")
        _ax.set_title("Per-Class F1 by Route")
        _ax.set_xticks([_xi + _width * (len(_route_names) - 1) / 2 for _xi in _x])
        _ax.set_xticklabels(_classes)
        _ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        _ax.set_ylim(0, 1)
        _plt.tight_layout()
        _plt.show()
    else:
        mo.md("## 3. Per-Class F1 Breakdown\n\nNo per-class data found.")
    return


@app.cell
def _(mo):
    """Section 4 heading — confusion matrices.

    Confusion matrix PNGs are saved by the classification scripts
    to results/confusion_matrices_*.png.
    """
    mo.md("## 4. Confusion Matrices\n\nConfusion matrix PNGs from the results directory:")
    return


@app.cell
def _(mo):
    """Display confusion matrix images from disk.

    Globs for confusion_matrices_*.png in the results directory and
    stacks all found images vertically via mo.vstack.  If no PNGs
    exist (e.g., confusion matrices weren't generated), shows a
    fallback message.
    """
    from pathlib import Path as _Path
    _repo_root = _Path(__file__).parent.parent
    _pngs = sorted((_repo_root / "results").glob("confusion_matrices_*.png"))
    if _pngs:
        mo.vstack([mo.image(str(_p)) for _p in _pngs])
    else:
        mo.md("No confusion matrix images found.")
    return


@app.cell
def _(mo):
    """Section 5 heading — feature importance for Route 2.

    Route 2 (text + graph stats) uses logistic regression, so
    permutation importance can identify which graph-stat features
    contribute most to classification.
    """
    mo.md("## 5. Feature Importance (Route 2)\n\nTop graph statistics features from permutation importance:")
    return


@app.cell
def _():
    """Feature importance table — pinned from results-log.md.

    These four features had the highest permutation importance in
    Route 2's logistic regression.  The values are pinned here
    (rather than read from disk) because permutation importance
    results were not persisted as a separate JSON file.
    """
    import polars as _pl

    _features = [
        {"feature": "diameter_norm", "importance": 0.0092},
        {"feature": "component_ratio", "importance": 0.0036},
        {"feature": "mixed_stance_frac", "importance": 0.0034},
        {"feature": "max_degree_norm", "importance": 0.0023},
    ]
    _pl.DataFrame(_features)
    return


@app.cell
def _(mo):
    """Section 6 heading — demographic sub-classification results.

    The demographic classification runs a 2-class task (tool_user vs
    integrated) for AI adoption, using the same route structure but
    on the subset of transcripts that have demographic metadata.
    """
    mo.md("## 6. Demographic Classification Results\n\nAI adoption (tool_user vs integrated) — n=1,224:")
    return


@app.cell
def _(demo_results, mo):
    """Demographic results table — AI adoption macro-F1 per route.

    Reads demo_results["ai_adoption"], a dict keyed by route name
    with macro_f1 values.  Builds a Polars DataFrame.  Falls back
    to a text message if no demographic results are available.
    """
    import polars as _pl

    if demo_results:
        _demo_rows = []
        for _route, _data in demo_results.get("ai_adoption", {}).items():
            _demo_rows.append({
                "route": _route,
                "macro_f1": round(_data.get("macro_f1", 0), 4),
            })

        if _demo_rows:
            _pl.DataFrame(_demo_rows)
        else:
            mo.md("No AI adoption data found.")
    else:
        mo.md("No demographic results found.")
    return


@app.cell
def _(mo):
    """Section 7 heading — GNN training curves.

    The training loss and validation F1 curves for the GIN model
    (Route 3) are saved as cache/gnn_curves.png during training.
    """
    mo.md("## 7. GNN Training Curves")
    return


@app.cell
def _(mo):
    """Display the GNN training curve image.

    Loads cache/gnn_curves.png (generated by Phase 3 encoding/gnn/train.py, now archived)
    and displays it inline.  Uses a ternary expression so the image
    or fallback message is unambiguously the cell's last expression
    (required by Marimo's display model).
    """
    from pathlib import Path as _Path
    _repo_root = _Path(__file__).parent.parent
    _curve_path = _repo_root / "cache/gnn_curves.png"
    (
        mo.image(str(_curve_path))
        if _curve_path.exists()
        else mo.md("Training curve image not found at cache/gnn_curves.png")
    )
    return


@app.cell
def _(mo):
    """Footer — notebook identity and data source."""
    mo.md("---\n\n*Notebook 03 — Classification Results. Data from results/ directory.*")
    return


if __name__ == "__main__":
    app.run()
