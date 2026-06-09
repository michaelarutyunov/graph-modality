# ruff: noqa

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    """Imports."""
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy import stats as sp_stats

    return Path, json, mo, np, pl, sp_stats


@app.cell
def _(Path):
    """Resolve repo root."""
    repo_root = Path(__file__).parent.parent
    return (repo_root,)


@app.cell
def _(json, repo_root):
    """Load all canonical graphs."""
    _paths = sorted(repo_root.glob("data/graphs/canonical/*.json"))
    graphs = []
    for _p in _paths:
        with open(_p) as _f:
            graphs.append(json.load(_f))
    n_graphs = len(graphs)
    n_graphs
    return graphs, n_graphs


@app.cell
def _(json, repo_root):
    """Load demographic labels for AI adoption analysis."""
    _demo_path = repo_root / "cache" / "demographics.jsonl"
    demos = {}
    if _demo_path.exists():
        with open(_demo_path) as _f:
            for _line in _f:
                _d = json.loads(_line)
                _tid = _d["transcript_id"]
                _ai = _d.get("ai_adoption", {})
                _cs = _d.get("career_stage", {})
                demos[_tid] = {
                    "ai_adoption": _ai.get("label", "unknown")
                    if isinstance(_ai, dict)
                    else "unknown",
                    "career_stage": _cs.get("label", "unknown")
                    if isinstance(_cs, dict)
                    else "unknown",
                }
    n_demos = len(demos)
    n_demos
    return demos, n_demos


@app.cell
def _(demos, graphs, np, pl):
    """Build unified DataFrame: graph metrics + demographics + split."""
    _rows = []
    for _g in graphs:
        _nodes = _g.get("nodes", [])
        _edges = _g.get("edges", [])
        _tid = _g["transcript_id"]
        _split = _g.get("split", "unknown")

        # --- node type counts ---
        _type_counts = {"Construct": 0, "Value": 0, "Stance": 0, "CognitiveStyleMarker": 0}
        for _n in _nodes:
            _t = _n.get("type", "")
            if _t in _type_counts:
                _type_counts[_t] += 1

        _n_total = len(_nodes)
        _n_edges = len(_edges)
        _n_construct = _type_counts["Construct"]
        _n_value = _type_counts["Value"]
        _n_stance = _type_counts["Stance"]
        _n_csm = _type_counts["CognitiveStyleMarker"]

        # --- H1 metrics: Construct:Value ratio ---
        _cv_ratio = _n_construct / max(_n_value, 1)

        # --- H2 metrics: negative valence fraction ---
        _stances = [_n for _n in _nodes if _n.get("type") == "Stance"]
        _n_negative = sum(1 for _s in _stances if _s.get("valence") == "negative")
        _neg_frac = _n_negative / max(len(_stances), 1)

        # Valence distribution
        _valence_counts = {"positive": 0, "negative": 0, "mixed": 0, "ambivalent": 0}
        for _s in _stances:
            _v = _s.get("valence", "ambivalent")
            if _v in _valence_counts:
                _valence_counts[_v] += 1

        # --- H3 metrics: bipolarity completeness ---
        _constructs = [_n for _n in _nodes if _n.get("type") == "Construct"]
        _bipolarity_score = (
            np.mean([1.0 if _c.get("bipolarity_complete") else 0.5 for _c in _constructs])
            if _constructs
            else 0.0
        )
        _bipolarity_complete_frac = (
            sum(1 for _c in _constructs if _c.get("bipolarity_complete")) / len(_constructs)
            if _constructs
            else 0.0
        )

        # --- H4 metrics: CSM prevalence + verification orientation ---
        _csms = [_n for _n in _nodes if _n.get("type") == "CognitiveStyleMarker"]
        _csm_count = len(_csms)
        _csm_present = float(_csm_count > 0)
        _verify_keywords = [
            "verif",
            "systematic",
            "evidence",
            "empirical",
            "analytic",
            "rigour",
            "rigor",
            "precision",
            "cross-check",
            "cross-referenc",
            "source check",
        ]
        _csm_verify = sum(
            1 for _c in _csms if any(_kw in _c.get("label", "").lower() for _kw in _verify_keywords)
        )
        _csm_verify_frac = _csm_verify / max(_csm_count, 1)

        # --- Structural: edge types ---
        _rel_counts = {"SERVES": 0, "EXPRESSED_VIA": 0, "MODULATED_BY": 0, "CONFLICTS_WITH": 0}
        for _e in _edges:
            _r = _e.get("relation", "")
            if _r in _rel_counts:
                _rel_counts[_r] += 1
        _has_conflict = float(_rel_counts["CONFLICTS_WITH"] > 0)

        # --- Demographics ---
        _demo = demos.get(_tid, {})
        _ai_adoption = _demo.get("ai_adoption", "unknown")
        _career_stage = _demo.get("career_stage", "unknown")

        _rows.append(
            {
                "transcript_id": _tid,
                "split": _split,
                "n_total": _n_total,
                "n_edges": _n_edges,
                "n_construct": _n_construct,
                "n_value": _n_value,
                "n_stance": _n_stance,
                "n_csm": _n_csm,
                "cv_ratio": _cv_ratio,
                "neg_frac": _neg_frac,
                "valence_positive": _valence_counts["positive"],
                "valence_negative": _valence_counts["negative"],
                "valence_mixed": _valence_counts["mixed"],
                "valence_ambivalent": _valence_counts["ambivalent"],
                "bipolarity_score": _bipolarity_score,
                "bipolarity_complete_frac": _bipolarity_complete_frac,
                "csm_count": _csm_count,
                "csm_present": _csm_present,
                "csm_verify_frac": _csm_verify_frac,
                "has_conflict": _has_conflict,
                "ai_adoption": _ai_adoption,
                "career_stage": _career_stage,
            }
        )

    df = pl.DataFrame(_rows)
    df
    return (df,)


@app.cell
def _(df, mo, pl):
    """Header + corpus overview."""
    _n = df.height
    _splits = df.group_by("split").len().sort("split")
    mo.vstack(
        [
            mo.md(f"""
        # Notebook 04 — Structural Analysis (RQ2)

        **Phase 4:** Testing pre-registered hypotheses H1–H4 about cohort differences in concept graph topology.
        All tests use Kruskal-Wallis (non-parametric omnibus) + pairwise Mann-Whitney U with Bonferroni correction.
        Effect sizes reported as Cliff's delta.

        **Corpus:** {_n} canonical graphs across 3 cohorts.
        """),
            _splits,
        ]
    )
    return


@app.cell
def _(mo):
    """Divider."""
    mo.md("---")
    return


# ═══════════════════════════════════════════════════════════════════
# H1 — Scientist Hub-and-Spoke
# ═══════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## H1 — Scientist Hub-and-Spoke

    **Prediction:** Scientists exhibit a higher Construct:Value ratio than other cohorts, with Value nodes
    serving as high-degree hubs. Interpretation: scientists have more constructs serving a smaller set of
    terminal values (epistemic rigour, data integrity), producing a structurally different graph.
    """)
    return


@app.cell
def _(df, mo, pl, sp_stats):
    """H1: Construct:Value ratio — statistical test + summary."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _groups = [df.filter(pl.col("split") == _c)["cv_ratio"].to_list() for _c in _cohorts]

    # Kruskal-Wallis omnibus
    _h, _p_kw = sp_stats.kruskal(*_groups)

    # Summary stats
    _summary = (
        df.group_by("split")
        .agg(
            [
                pl.col("cv_ratio").mean().round(3).alias("mean_cv_ratio"),
                pl.col("cv_ratio").median().round(3).alias("median_cv_ratio"),
                pl.col("cv_ratio").std().round(3).alias("std_cv_ratio"),
                pl.col("n_value").mean().round(2).alias("mean_n_values"),
                pl.col("n_construct").mean().round(2).alias("mean_n_constructs"),
            ]
        )
        .sort("split")
    )

    mo.vstack(
        [
            mo.md(f"""
        **Kruskal-Wallis:** H = {_h:.3f}, p = {_p_kw:.6f}
        {"**Significant!**" if _p_kw < 0.05 else "Not significant"}
        """),
            _summary,
        ]
    )
    return


@app.cell
def _(df, mo, np, pl, sp_stats):
    """H1: Pairwise post-hoc with Bonferroni correction."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _n_comparisons = 3  # 3 pairs
    _alpha_corrected = 0.05 / _n_comparisons

    _rows = []
    for _i, _c1 in enumerate(_cohorts):
        for _c2 in _cohorts[_i + 1 :]:
            _g1 = df.filter(pl.col("split") == _c1)["cv_ratio"].to_list()
            _g2 = df.filter(pl.col("split") == _c2)["cv_ratio"].to_list()
            _u, _p = sp_stats.mannwhitneyu(_g1, _g2, alternative="two-sided")
            _p_corrected = min(_p * _n_comparisons, 1.0)

            # Cliff's delta
            _n1, _n2 = len(_g1), len(_g2)
            _greater = sum(1 for _a in _g1 for _b in _g2 if _a > _b)
            _less = sum(1 for _a in _g1 for _b in _g2 if _a < _b)
            _delta = (_greater - _less) / (_n1 * _n2)

            _rows.append(
                {
                    "comparison": f"{_c1} vs {_c2}",
                    "U": _u,
                    "p_raw": round(_p, 6),
                    "p_bonferroni": round(_p_corrected, 6),
                    "significant": _p_corrected < 0.05,
                    "cliffs_delta": round(_delta, 4),
                }
            )

    _posthoc = pl.DataFrame(_rows)
    mo.vstack(
        [
            mo.md(
                f"**Pairwise Mann-Whitney U** (Bonferroni-corrected α = {_alpha_corrected:.4f}):"
            ),
            _posthoc,
        ]
    )
    return


@app.cell
def _(df, mo, np):
    """H1: Violin plot — Construct:Value ratio by cohort."""
    import matplotlib.pyplot as _plt

    _cohorts = ["workforce", "creatives", "scientists"]
    _fig, _ax = _plt.subplots(figsize=(10, 5))
    _data = [df.filter(df["split"] == _c)["cv_ratio"].to_list() for _c in _cohorts]
    _vp = _ax.violinplot(_data, showmeans=True, showmedians=True)
    for _pc in _vp["bodies"]:
        _pc.set_facecolor("lightblue")
        _pc.set_alpha(0.7)
    _ax.set_xticks([1, 2, 3])
    _ax.set_xticklabels(_cohorts)
    _ax.set_ylabel("Construct:Value Ratio")
    _ax.set_title("H1: Construct:Value Ratio by Cohort (Hub-and-Spoke)")
    _ax.grid(axis="y", alpha=0.3)
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(mo):
    mo.md("---")
    return


# ═══════════════════════════════════════════════════════════════════
# H2 — Creative Negative Valence
# ═══════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## H2 — Creative Negative Valence

    **Prediction:** Creatives exhibit a higher proportion of negative-valence Stances than other cohorts.
    The dual satisfaction/anxiety pattern (97% productivity gains alongside pervasive identity anxiety)
    should leave a structural fingerprint in stance valence distributions.
    """)
    return


@app.cell
def _(df, mo, pl, sp_stats):
    """H2: Negative valence fraction — statistical test."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _groups = [df.filter(pl.col("split") == _c)["neg_frac"].to_list() for _c in _cohorts]

    _h, _p_kw = sp_stats.kruskal(*_groups)

    _summary = (
        df.group_by("split")
        .agg(
            [
                pl.col("neg_frac").mean().round(4).alias("mean_neg_frac"),
                pl.col("neg_frac").median().round(4).alias("median_neg_frac"),
                pl.col("valence_negative").mean().round(2).alias("mean_n_negative"),
                pl.col("n_stance").mean().round(2).alias("mean_n_stances"),
            ]
        )
        .sort("split")
    )

    mo.vstack(
        [
            mo.md(f"""
        **Kruskal-Wallis:** H = {_h:.3f}, p = {_p_kw:.6f}
        {"**Significant!**" if _p_kw < 0.05 else "Not significant"}
        """),
            _summary,
        ]
    )
    return


@app.cell
def _(df, mo, np, pl, sp_stats):
    """H2: Pairwise post-hoc."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _n_comparisons = 3

    _rows = []
    for _i, _c1 in enumerate(_cohorts):
        for _c2 in _cohorts[_i + 1 :]:
            _g1 = df.filter(pl.col("split") == _c1)["neg_frac"].to_list()
            _g2 = df.filter(pl.col("split") == _c2)["neg_frac"].to_list()
            _u, _p = sp_stats.mannwhitneyu(_g1, _g2, alternative="two-sided")
            _p_corrected = min(_p * _n_comparisons, 1.0)

            _n1, _n2 = len(_g1), len(_g2)
            _greater = sum(1 for _a in _g1 for _b in _g2 if _a > _b)
            _less = sum(1 for _a in _g1 for _b in _g2 if _a < _b)
            _delta = (_greater - _less) / (_n1 * _n2)

            _rows.append(
                {
                    "comparison": f"{_c1} vs {_c2}",
                    "p_bonferroni": round(_p_corrected, 6),
                    "significant": _p_corrected < 0.05,
                    "cliffs_delta": round(_delta, 4),
                }
            )

    mo.vstack(
        [
            mo.md("**Pairwise Mann-Whitney U** (Bonferroni-corrected):"),
            pl.DataFrame(_rows),
        ]
    )
    return


@app.cell
def _(df, mo):
    """H2: Violin plot — negative valence fraction by cohort."""
    import matplotlib.pyplot as _plt

    _cohorts = ["workforce", "creatives", "scientists"]
    _fig, _ax = _plt.subplots(figsize=(10, 5))
    _data = [df.filter(df["split"] == _c)["neg_frac"].to_list() for _c in _cohorts]
    _vp = _ax.violinplot(_data, showmeans=True, showmedians=True)
    for _pc in _vp["bodies"]:
        _pc.set_facecolor("salmon")
        _pc.set_alpha(0.7)
    _ax.set_xticks([1, 2, 3])
    _ax.set_xticklabels(_cohorts)
    _ax.set_ylabel("Fraction of Negative-Valence Stances")
    _ax.set_title("H2: Negative Stance Fraction by Cohort")
    _ax.grid(axis="y", alpha=0.3)
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(df, mo, np, pl):
    """H2: Valence distribution stacked bar chart."""
    import matplotlib.pyplot as _plt

    _cohorts = ["workforce", "creatives", "scientists"]
    _valences = ["positive", "negative", "mixed", "ambivalent"]
    _colors = ["#2ca02c", "#d62728", "#ff7f0e", "#7f7f7f"]

    _fig, _axes = _plt.subplots(1, 3, figsize=(15, 5))
    for _ax, _c in zip(_axes, _cohorts, strict=False):
        _subset = df.filter(df["split"] == _c)
        _totals = [_subset[f"valence_{_v}"].sum() for _v in _valences]
        _total = sum(_totals) if sum(_totals) > 0 else 1
        _props = [_t / _total for _t in _totals]
        _ax.pie(_props, labels=_valences, colors=_colors, autopct="%1.1f%%")
        _ax.set_title(f"{_c}\n(n={_subset.height})")

    _plt.suptitle("H2: Stance Valence Distribution by Cohort", fontsize=14)
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(mo):
    mo.md("---")
    return


# ═══════════════════════════════════════════════════════════════════
# H3 — Workforce Bipolarity
# ═══════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## H3 — Workforce Bipolarity

    **Prediction:** Workforce respondents exhibit higher bipolarity completeness scores than creatives.
    Workforce respondents articulate clearer trade-offs (human value vs automatable work), whereas
    creatives express more ambivalent or unresolved constructs.
    """)
    return


@app.cell
def _(df, mo, pl, sp_stats):
    """H3: Bipolarity completeness — statistical test."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _groups = [
        df.filter(pl.col("split") == _c)["bipolarity_complete_frac"].to_list() for _c in _cohorts
    ]

    _h, _p_kw = sp_stats.kruskal(*_groups)

    _summary = (
        df.group_by("split")
        .agg(
            [
                pl.col("bipolarity_score").mean().round(4).alias("mean_bipolarity_score"),
                pl.col("bipolarity_complete_frac").mean().round(4).alias("mean_complete_frac"),
                pl.col("bipolarity_complete_frac").median().round(4).alias("median_complete_frac"),
            ]
        )
        .sort("split")
    )

    mo.vstack(
        [
            mo.md(f"""
        **Kruskal-Wallis:** H = {_h:.3f}, p = {_p_kw:.6f}
        {"**Significant!**" if _p_kw < 0.05 else "Not significant"}
        """),
            _summary,
        ]
    )
    return


@app.cell
def _(df, mo, pl, sp_stats):
    """H3: Pairwise post-hoc."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _n_comparisons = 3

    _rows = []
    for _i, _c1 in enumerate(_cohorts):
        for _c2 in _cohorts[_i + 1 :]:
            _g1 = df.filter(pl.col("split") == _c1)["bipolarity_complete_frac"].to_list()
            _g2 = df.filter(pl.col("split") == _c2)["bipolarity_complete_frac"].to_list()
            _u, _p = sp_stats.mannwhitneyu(_g1, _g2, alternative="two-sided")
            _p_corrected = min(_p * _n_comparisons, 1.0)

            _n1, _n2 = len(_g1), len(_g2)
            _greater = sum(1 for _a in _g1 for _b in _g2 if _a > _b)
            _less = sum(1 for _a in _g1 for _b in _g2 if _a < _b)
            _delta = (_greater - _less) / (_n1 * _n2)

            _rows.append(
                {
                    "comparison": f"{_c1} vs {_c2}",
                    "p_bonferroni": round(_p_corrected, 6),
                    "significant": _p_corrected < 0.05,
                    "cliffs_delta": round(_delta, 4),
                }
            )

    mo.vstack(
        [
            mo.md("**Pairwise Mann-Whitney U** (Bonferroni-corrected):"),
            pl.DataFrame(_rows),
        ]
    )
    return


@app.cell
def _(df, mo):
    """H3: Violin plot — bipolarity completeness by cohort."""
    import matplotlib.pyplot as _plt

    _cohorts = ["workforce", "creatives", "scientists"]
    _fig, _ax = _plt.subplots(figsize=(10, 5))
    _data = [df.filter(df["split"] == _c)["bipolarity_complete_frac"].to_list() for _c in _cohorts]
    _vp = _ax.violinplot(_data, showmeans=True, showmedians=True)
    for _pc in _vp["bodies"]:
        _pc.set_facecolor("lightgreen")
        _pc.set_alpha(0.7)
    _ax.set_xticks([1, 2, 3])
    _ax.set_xticklabels(_cohorts)
    _ax.set_ylabel("Bipolarity Completeness Fraction")
    _ax.set_title("H3: Construct Bipolarity Completeness by Cohort")
    _ax.grid(axis="y", alpha=0.3)
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(mo):
    mo.md("---")
    return


# ═══════════════════════════════════════════════════════════════════
# H4 — Scientist Cognitive Style
# ═══════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## H4 — Scientist Cognitive Style

    **Prediction:** Cognitive Style Markers are more prevalent in scientist transcripts and more likely
    to be verification-oriented (maximiser, evidence-anchored, systematic verification) than in workforce transcripts.
    """)
    return


@app.cell
def _(df, mo, pl, sp_stats):
    """H4: CSM count — statistical test."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _groups = [df.filter(pl.col("split") == _c)["csm_count"].to_list() for _c in _cohorts]

    _h, _p_kw = sp_stats.kruskal(*_groups)

    _summary = (
        df.group_by("split")
        .agg(
            [
                pl.col("csm_count").mean().round(3).alias("mean_csm_count"),
                pl.col("csm_count").median().round(1).alias("median_csm_count"),
                pl.col("csm_present").mean().round(4).alias("csm_prevalence"),
                pl.col("csm_verify_frac").mean().round(4).alias("mean_verify_fraction"),
            ]
        )
        .sort("split")
    )

    mo.vstack(
        [
            mo.md(f"""
        **Kruskal-Wallis (CSM count):** H = {_h:.3f}, p = {_p_kw:.6f}
        {"**Significant!**" if _p_kw < 0.05 else "Not significant"}
        """),
            _summary,
        ]
    )
    return


@app.cell
def _(df, mo, pl, sp_stats):
    """H4: Pairwise post-hoc for CSM count."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _n_comparisons = 3

    _rows = []
    for _i, _c1 in enumerate(_cohorts):
        for _c2 in _cohorts[_i + 1 :]:
            _g1 = df.filter(pl.col("split") == _c1)["csm_count"].to_list()
            _g2 = df.filter(pl.col("split") == _c2)["csm_count"].to_list()
            _u, _p = sp_stats.mannwhitneyu(_g1, _g2, alternative="two-sided")
            _p_corrected = min(_p * _n_comparisons, 1.0)

            _n1, _n2 = len(_g1), len(_g2)
            _greater = sum(1 for _a in _g1 for _b in _g2 if _a > _b)
            _less = sum(1 for _a in _g1 for _b in _g2 if _a < _b)
            _delta = (_greater - _less) / (_n1 * _n2)

            _rows.append(
                {
                    "comparison": f"{_c1} vs {_c2}",
                    "p_bonferroni": round(_p_corrected, 6),
                    "significant": _p_corrected < 0.05,
                    "cliffs_delta": round(_delta, 4),
                }
            )

    mo.vstack(
        [
            mo.md("**Pairwise Mann-Whitney U** (Bonferroni-corrected):"),
            pl.DataFrame(_rows),
        ]
    )
    return


@app.cell
def _(df, mo, pl, sp_stats):
    """H4: Verification-oriented CSM fraction — only graphs that have CSMs."""
    _df_csm = df.filter(pl.col("csm_count") > 0)
    _cohorts = ["workforce", "creatives", "scientists"]
    _groups = [
        _df_csm.filter(pl.col("split") == _c)["csm_verify_frac"].to_list() for _c in _cohorts
    ]

    # Only run if all groups have data
    if all(len(_g) > 0 for _g in _groups):
        _h, _p_kw = sp_stats.kruskal(*_groups)
        mo.md(f"""
        **Verification-orientation (graphs with ≥1 CSM, n={_df_csm.height}):**
        Kruskal-Wallis H = {_h:.3f}, p = {_p_kw:.6f}
        {"**Significant!**" if _p_kw < 0.05 else "Not significant"}
        """)
    else:
        mo.md("Insufficient data for verification-orientation test.")
    return


@app.cell
def _(df, mo, np):
    """H4: CSM prevalence + verification fraction bar chart."""
    import matplotlib.pyplot as _plt

    _cohorts = ["workforce", "creatives", "scientists"]
    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(14, 5))

    # CSM prevalence
    _prevalence = [df.filter(df["split"] == _c)["csm_present"].mean() for _c in _cohorts]
    _ax1.bar(_cohorts, _prevalence, color=["steelblue", "coral", "seagreen"])
    _ax1.set_ylabel("Proportion with ≥1 CSM")
    _ax1.set_title("H4: CSM Prevalence by Cohort")

    # Verification fraction
    _verify = [df.filter(df["split"] == _c)["csm_verify_frac"].mean() for _c in _cohorts]
    _ax2.bar(_cohorts, _verify, color=["steelblue", "coral", "seagreen"])
    _ax2.set_ylabel("Mean Verification-Oriented Fraction")
    _ax2.set_title("H4: Verification-Oriented CSM Fraction")

    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(mo):
    mo.md("---")
    return


# ═══════════════════════════════════════════════════════════════════
# H1-H4 Summary Table
# ═══════════════════════════════════════════════════════════════════


@app.cell
def _(df, mo, np, pl, sp_stats):
    """Build H1-H4 summary table with all test statistics."""
    _cohorts = ["workforce", "creatives", "scientists"]

    _hypotheses = []
    for _h_name, _metric, _description in [
        ("H1", "cv_ratio", "Construct:Value ratio"),
        ("H2", "neg_frac", "Negative stance fraction"),
        ("H3", "bipolarity_complete_frac", "Bipolarity completeness"),
        ("H4", "csm_count", "CSM count"),
    ]:
        _groups = [df.filter(pl.col("split") == _c)[_metric].to_list() for _c in _cohorts]
        _h_stat, _p_val = sp_stats.kruskal(*_groups)

        # Compute eta-squared effect size
        _all_vals = np.concatenate([np.array(_g) for _g in _groups])
        _grand_mean = np.mean(_all_vals)
        _ss_between = sum(len(_g) * (np.mean(_g) - _grand_mean) ** 2 for _g in _groups)
        _ss_total = sum((_v - _grand_mean) ** 2 for _v in _all_vals)
        _eta_sq = _ss_between / _ss_total if _ss_total > 0 else 0.0

        _means = [round(np.mean(_g), 4) for _g in _groups]

        _hypotheses.append(
            {
                "hypothesis": _h_name,
                "metric": _description,
                "Kruskal-Wallis H": round(_h_stat, 3),
                "p_value": round(_p_val, 6),
                "significant (α=0.05)": _p_val < 0.05,
                "eta_squared": round(_eta_sq, 6),
                f"mean_{_cohorts[0]}": _means[0],
                f"mean_{_cohorts[1]}": _means[1],
                f"mean_{_cohorts[2]}": _means[2],
            }
        )

    _df_summary = pl.DataFrame(_hypotheses)
    mo.vstack(
        [
            mo.md("## H1–H4 Summary"),
            mo.md("Omnibus test results with eta-squared effect sizes:"),
            _df_summary,
            mo.md("""
        **Effect size interpretation (eta-squared):**
        - Small: ~0.01 | Medium: ~0.06 | Large: ~0.14
        """),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("---")
    return


# ═══════════════════════════════════════════════════════════════════
# Exploratory: AI Adoption Structural Analysis
# ═══════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Exploratory: Structural Analysis by AI Adoption

    **No pre-registered hypotheses.** Same metrics as H1–H4, regrouped by `tool_user` vs `integrated` (n=1,224;
    novice=21 and power_user=5 excluded). Mann-Whitney U with Cliff's delta.
    """)
    return


@app.cell
def _(df, mo, pl):
    """Filter to AI adoption groups."""
    _df_ai = df.filter(pl.col("ai_adoption").is_in(["tool_user", "integrated"]))
    _n_tool = _df_ai.filter(pl.col("ai_adoption") == "tool_user").height
    _n_integrated = _df_ai.filter(pl.col("ai_adoption") == "integrated").height

    mo.md(f"""
    **Sample:** tool_user = {_n_tool}, integrated = {_n_integrated}, total = {_df_ai.height}
    """)
    return


@app.cell
def _(df, mo, np, pl, sp_stats):
    """AI adoption: compute all metric comparisons."""
    _df_ai = df.filter(pl.col("ai_adoption").is_in(["tool_user", "integrated"]))

    _metrics = [
        ("cv_ratio", "Construct:Value ratio", "H1"),
        ("neg_frac", "Negative stance fraction", "H2"),
        ("bipolarity_complete_frac", "Bipolarity completeness", "H3"),
        ("csm_count", "CSM count", "H4"),
        ("csm_verify_frac", "Verification-oriented CSM fraction", "H4"),
        ("n_total", "Graph size (nodes)", "structural"),
        ("n_edges", "Edge count", "structural"),
        ("has_conflict", "Conflict prevalence", "structural"),
    ]

    _rows = []
    for _metric, _desc, _source in _metrics:
        _g_tool = _df_ai.filter(pl.col("ai_adoption") == "tool_user")[_metric].to_list()
        _g_int = _df_ai.filter(pl.col("ai_adoption") == "integrated")[_metric].to_list()

        _u, _p = sp_stats.mannwhitneyu(_g_tool, _g_int, alternative="two-sided")

        # Cliff's delta
        _n1, _n2 = len(_g_tool), len(_g_int)
        _greater = sum(1 for _a in _g_tool for _b in _g_int if _a > _b)
        _less = sum(1 for _a in _g_tool for _b in _g_int if _a < _b)
        _delta = (_greater - _less) / (_n1 * _n2)

        _mean_tool = np.mean(_g_tool)
        _mean_int = np.mean(_g_int)

        _rows.append(
            {
                "metric": _desc,
                "source_hypothesis": _source,
                "mean_tool_user": round(_mean_tool, 4),
                "mean_integrated": round(_mean_int, 4),
                "cliffs_delta": round(_delta, 4),
                "p_value": round(_p, 6),
                "significant (α=0.05)": _p < 0.05,
                "abs_delta": abs(round(_delta, 4)),
            }
        )

    _df_ai_results = pl.DataFrame(_rows).sort("abs_delta", descending=True)
    mo.vstack(
        [
            mo.md("### AI Adoption — All Metrics"),
            _df_ai_results,
        ]
    )
    return


@app.cell
def _(df, mo, np):
    """AI adoption: violin plots for key metrics."""
    import matplotlib.pyplot as _plt

    _df_ai = df.filter(df["ai_adoption"].is_in(["tool_user", "integrated"]))

    _fig, _axes = _plt.subplots(2, 3, figsize=(16, 10))
    _plot_specs = [
        ("cv_ratio", "Construct:Value Ratio", 0, 0),
        ("neg_frac", "Negative Stance Fraction", 0, 1),
        ("bipolarity_complete_frac", "Bipolarity Completeness", 0, 2),
        ("csm_count", "CSM Count", 1, 0),
        ("n_total", "Graph Size (nodes)", 1, 1),
        ("has_conflict", "Conflict Prevalence", 1, 2),
    ]

    for _metric, _title, _row, _col in _plot_specs:
        _ax = _axes[_row][_col]
        _data = [
            _df_ai.filter(_df_ai["ai_adoption"] == "tool_user")[_metric].to_list(),
            _df_ai.filter(_df_ai["ai_adoption"] == "integrated")[_metric].to_list(),
        ]
        _vp = _ax.violinplot(_data, showmeans=True, showmedians=True)
        _colors = ["steelblue", "coral"]
        for _i, _pc in enumerate(_vp["bodies"]):
            _pc.set_facecolor(_colors[_i])
            _pc.set_alpha(0.7)
        _ax.set_xticks([1, 2])
        _ax.set_xticklabels(["tool_user", "integrated"])
        _ax.set_ylabel(_title)
        _ax.set_title(_title)
        _ax.grid(axis="y", alpha=0.3)

    _plt.suptitle("Structural Metrics by AI Adoption", fontsize=14, y=1.01)
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(mo):
    mo.md("---")
    return


# ═══════════════════════════════════════════════════════════════════
# Effect Size Comparison: Cohort vs AI Adoption
# ═══════════════════════════════════════════════════════════════════


@app.cell
def _(df, mo, np, pl, sp_stats):
    """Compare structural differentiation by cohort vs by AI adoption."""
    _cohorts = ["workforce", "creatives", "scientists"]
    _df_ai = df.filter(pl.col("ai_adoption").is_in(["tool_user", "integrated"]))

    _metrics = [
        ("cv_ratio", "Construct:Value ratio"),
        ("neg_frac", "Negative stance fraction"),
        ("bipolarity_complete_frac", "Bipolarity completeness"),
        ("csm_count", "CSM count"),
    ]

    _rows = []
    for _metric, _desc in _metrics:
        # Cohort: eta-squared from Kruskal-Wallis
        _groups = [df.filter(pl.col("split") == _c)[_metric].to_list() for _c in _cohorts]
        _h_stat, _p_cohort = sp_stats.kruskal(*_groups)
        _all_vals = np.concatenate([np.array(_g) for _g in _groups])
        _grand_mean = np.mean(_all_vals)
        _ss_between = sum(len(_g) * (np.mean(_g) - _grand_mean) ** 2 for _g in _groups)
        _ss_total = sum((_v - _grand_mean) ** 2 for _v in _all_vals)
        _eta_sq = _ss_between / _ss_total if _ss_total > 0 else 0.0

        # AI adoption: absolute Cliff's delta
        _g_tool = _df_ai.filter(pl.col("ai_adoption") == "tool_user")[_metric].to_list()
        _g_int = _df_ai.filter(pl.col("ai_adoption") == "integrated")[_metric].to_list()
        _u, _p_ai = sp_stats.mannwhitneyu(_g_tool, _g_int, alternative="two-sided")
        _n1, _n2 = len(_g_tool), len(_g_int)
        _greater = sum(1 for _a in _g_tool for _b in _g_int if _a > _b)
        _less = sum(1 for _a in _g_tool for _b in _g_int if _a < _b)
        _abs_delta = abs((_greater - _less) / (_n1 * _n2))

        _rows.append(
            {
                "metric": _desc,
                "cohort_eta_sq": round(_eta_sq, 6),
                "cohort_p": round(_p_cohort, 6),
                "ai_abs_cliffs_delta": round(_abs_delta, 4),
                "ai_p": round(_p_ai, 6),
            }
        )

    _df_comp = pl.DataFrame(_rows)
    mo.vstack(
        [
            mo.md("""
        ## Effect Size Comparison: Cohort vs AI Adoption

        Which target shows stronger structural differentiation? Comparing eta-squared (cohort, 3-group)
        against absolute Cliff's delta (AI adoption, binary).
        """),
            _df_comp,
            mo.md("""
        **Interpretation:**
        - **Cohort (eta²):** variance explained by cohort membership
        - **AI adoption (|δ|):** separation between tool_user and integrated
        - Neither effect size is directly comparable across targets (different statistics),
          but magnitudes relative to their baselines indicate which target carries stronger
          structural signal.
        """),
        ]
    )
    return


@app.cell
def _(df, mo, np, pl, sp_stats):
    """Additional exploratory: within-cohort AI adoption effects."""
    _df_ai = df.filter(pl.col("ai_adoption").is_in(["tool_user", "integrated"]))
    _cohorts = ["workforce", "creatives", "scientists"]

    _rows = []
    for _c in _cohorts:
        _df_c = _df_ai.filter(pl.col("split") == _c)
        if _df_c.height < 10:
            continue
        _g_tool = _df_c.filter(pl.col("ai_adoption") == "tool_user")["cv_ratio"].to_list()
        _g_int = _df_c.filter(pl.col("ai_adoption") == "integrated")["cv_ratio"].to_list()
        if len(_g_tool) < 2 or len(_g_int) < 2:
            continue
        _u, _p = sp_stats.mannwhitneyu(_g_tool, _g_int, alternative="two-sided")
        _n1, _n2 = len(_g_tool), len(_g_int)
        _greater = sum(1 for _a in _g_tool for _b in _g_int if _a > _b)
        _less = sum(1 for _a in _g_tool for _b in _g_int if _a < _b)
        _delta = (_greater - _less) / (_n1 * _n2)
        _rows.append(
            {
                "cohort": _c,
                "n_tool": _n1,
                "n_integrated": _n2,
                "mean_cv_tool": round(np.mean(_g_tool), 3),
                "mean_cv_integrated": round(np.mean(_g_int), 3),
                "cliffs_delta": round(_delta, 4),
                "p_value": round(_p, 6),
            }
        )

    if _rows:
        mo.vstack(
            [
                mo.md("### Within-Cohort AI Adoption Effects (Construct:Value ratio)"),
                mo.md(
                    "Does AI adoption associate with graph structure within the same professional cohort?"
                ),
                pl.DataFrame(_rows),
            ]
        )
    else:
        mo.md("Insufficient data for within-cohort AI adoption analysis.")
    return


@app.cell
def _(mo):
    mo.md("---")
    return


# ═══════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ---

    *Notebook 04 — Structural Analysis (RQ2). Phase 4. Data: canonical concept graphs (n=1,250)
    + demographic labels from `cache/demographics.jsonl`.*
    """)
    return


if __name__ == "__main__":
    app.run()
