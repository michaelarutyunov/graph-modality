"""Unit tests for the RSA/Mantel geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from s5_classification.analysis_geometry_rsa import mantel_test, partial_mantel_test


def _random_distance_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 4))
    return pdist(points, metric="euclidean")


def test_mantel_identical_matrices():
    """Two identical distance matrices have Spearman r = 1.0."""
    rng = np.random.default_rng(7)
    points = rng.standard_normal((12, 5))
    d = pdist(points, metric="euclidean")

    result = mantel_test(d, d, n_perm=999, seed=42)

    assert result["r"] == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= result["p"] <= 1.0
    assert result["n"] == 12


def test_mantel_independent_random_matrices():
    """Independent random matrices have near-zero r and non-significant p."""
    d1 = _random_distance_matrix(16, seed=1)
    d2 = _random_distance_matrix(16, seed=2)

    result = mantel_test(d1, d2, n_perm=999, seed=42)

    assert abs(result["r"]) < 0.5
    assert result["p"] > 0.01


def test_partial_mantel_identical_target_and_control():
    """Partial Mantel of x with itself, controlling an unrelated matrix, is ~1."""
    rng = np.random.default_rng(3)
    n = 14
    points = rng.standard_normal((n, 4))
    x = pdist(points, metric="euclidean")
    z = _random_distance_matrix(n, seed=4)

    result = partial_mantel_test(x, x, z, n_perm=999, seed=42)

    assert result["r"] == pytest.approx(1.0, abs=1e-6)


def test_mantel_rejects_non_condensed_vectors():
    """The helper expects condensed vectors, not square matrices."""
    rng = np.random.default_rng(5)
    square = squareform(rng.uniform(size=45))

    with pytest.raises(ValueError, match="condensed distance vector"):
        mantel_test(square, square)
