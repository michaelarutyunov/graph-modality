"""Tests for H_edge edge-axis ablation (s5_classification/h_edge.py)."""

from __future__ import annotations

import numpy as np

from s5_classification.h_edge import _ci, _verdict


def test_verdict_pass_when_ci_excludes_zero_and_mean_meets_threshold():
    deltas = np.array([0.05, 0.04, 0.06, 0.05, 0.05, 0.04, 0.06, 0.05, 0.05, 0.05])
    mean, lo, _hi, verdict = _verdict(deltas)
    assert lo > 0 and mean >= 0.01
    assert verdict == "PASS"


def test_verdict_fail_when_ci_spans_zero():
    deltas = np.array([0.02, -0.03, 0.01, -0.02, 0.00, 0.03, -0.04, 0.02, -0.01, 0.01])
    _mean, lo, hi, verdict = _verdict(deltas)
    assert lo < 0 < hi
    assert verdict == "FAIL"


def test_verdict_fail_when_effect_below_threshold():
    # Tight CI excluding 0 but mean < 0.01 → still FAIL (effect-size floor).
    deltas = np.full(10, 0.005)
    deltas[0] += 1e-6  # avoid zero variance
    mean, _lo, _hi, verdict = _verdict(deltas)
    assert mean < 0.01
    assert verdict == "FAIL"


def test_ci_matches_mean():
    vals = np.array([0.4, 0.45, 0.5, 0.42, 0.48])
    mean, lo, hi = _ci(vals)
    assert lo < mean < hi
    assert abs(mean - float(np.mean(vals))) < 1e-9


def test_ablation_probe_supports_new_target():
    """ablation_probe._load_labels must recognise stance_ambivalence (feature axis)."""
    from s5_classification import ablation_probe

    # The label loader dispatch must include the new target (no ValueError branch).
    src = ablation_probe._load_labels.__code__.co_consts
    assert any(c == "stance_ambivalence" for c in src if isinstance(c, str))
