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
    """Load all result JSON files."""
    _repo_root = Path(__file__).parent.parent
    result_dir = _repo_root / "results"

    # Find the latest comparison files
    _comp_files = sorted(result_dir.glob("comparison_5routes_*.json"))
    _route_files = sorted(result_dir.glob("route*_*.json"))
    _demo_files = sorted(result_dir.glob("demographic_classification_*.json"))

    comparison = json.loads(_comp_files[-1].read_text()) if _comp_files else {}
    demo_results = json.loads(_demo_files[-1].read_text()) if _demo_files else {}

    # Load individual route results (latest timestamp)
    routes = {}
    for _p in _route_files:
        _data = json.loads(_p.read_text())
        _name = _data.get("route", _p.stem)
        routes[_name] = _data

    len(routes), len(demo_results)
    return comparison, demo_results, routes


@app.cell
def __(mo):
    """Header."""
    mo.md("""
    # Notebook 03 — Classification Results

    Interactive dashboard for all classification experiments.
    """)
    return


@app.cell
def __(comparison, mo):
    """Section 1: Route comparison table."""
    import polars as _pl

    mo.md("## 1. Route Comparison")

    if comparison.get("routes"):
        _rows = []
        for _name, _f1 in comparison["routes"].items():
            _rows.append({"route": _name, "test_macro_f1": round(_f1, 4)})
        _df_comp = _pl.DataFrame(_rows)
        _df_comp
    else:
        mo.md("No comparison data found.")
    return


@app.cell
def __(routes, mo):
    """Section 2: Per-route detail tables."""
    import polars as _pl

    mo.md("## 2. Per-Route Results")

    _detail_rows = []
    for _name, _data in routes.items():
        _detail_rows.append({
            "route": _name,
            "macro_f1": round(_data.get("macro_f1", 0), 4),
            "n_samples": _data.get("n_samples", "—"),
        })

    if _detail_rows:
        _pl.DataFrame(_detail_rows).sort("macro_f1", descending=True)
    else:
        mo.md("No route data found.")
    return


@app.cell
def __(routes, mo):
    """Section 3: Per-class F1 breakdown."""
    import matplotlib.pyplot as _plt
    import polars as _pl

    mo.md("## 3. Per-Class F1 Breakdown")

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
        mo.md("No per-class data found.")
    return


@app.cell
def __(mo):
    """Section 4: Confusion matrices."""
    mo.md("## 4. Confusion Matrices")
    mo.md("Confusion matrix PNGs from the results directory:")
    return


@app.cell
def __(mo, Path):
    """Display confusion matrix images."""
    _repo_root = Path(__file__).parent.parent
    _pngs = sorted((_repo_root / "results").glob("confusion_matrices_*.png"))
    if _pngs:
        for _p in _pngs:
            mo.image(str(_p))
    else:
        mo.md("No confusion matrix images found.")
    return


@app.cell
def __(mo):
    """Section 5: Feature importance."""
    mo.md("## 5. Feature Importance (Route 2)")
    mo.md("Top graph statistics features from permutation importance:")
    return


@app.cell
def __(mo):
    """Feature importance table — pinned from results-log.md."""
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
def __(mo):
    """Section 6: Demographic classification."""
    mo.md("## 6. Demographic Classification Results")
    mo.md("AI adoption (tool_user vs integrated) — n=1,224:")
    return


@app.cell
def __(demo_results, mo):
    """Demographic results table."""
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
def __(mo):
    """Section 7: Training curves."""
    mo.md("## 7. GNN Training Curves")
    return


@app.cell
def __(mo, Path):
    """Display training curve image."""
    _repo_root = Path(__file__).parent.parent
    _curve_path = _repo_root / "cache/gnn_curves.png"
    if _curve_path.exists():
        mo.image(str(_curve_path))
    else:
        mo.md("Training curve image not found at cache/gnn_curves.png")
    return


@app.cell
def __(mo):
    """Footer."""
    mo.md("---")
    mo.md("*Notebook 03 — Classification Results. Data from results/ directory.*")
    return


if __name__ == "__main__":
    app.run()
