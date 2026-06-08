"""Tests for Route 1 -- text-only baseline classifier."""

from __future__ import annotations

import os
import tempfile

import pytest
from sklearn.linear_model import LogisticRegression

# Ensure beads tests use a temp DB, never the production one
os.environ.setdefault("BEADS_DB", tempfile.mkdtemp())

from classification.baseline import (
    MODEL_PATH,
    VAL_PREDS_PATH,
    load_data,
    run_baseline,
)


@pytest.fixture(scope="module")
def baseline_results():
    """Run the full baseline pipeline once and cache results."""
    results = run_baseline()
    return results


@pytest.fixture(scope="module")
def raw_data():
    """Load data once for shape/leakage tests."""
    return load_data()


class TestTraining:
    def test_model_trains(self, baseline_results):
        """Model trains without error on real data."""
        assert baseline_results is not None
        assert "macro_f1" in baseline_results
        assert "confusion_matrix" in baseline_results


class TestPredictions:
    def test_predictions_shape(self, baseline_results, raw_data):
        """Predictions have correct length (= val set size)."""
        _X_train, _y_train, _X_val, y_val, _val_ids, _label_names = raw_data
        preds = baseline_results["predictions"]
        assert len(preds) == len(y_val)

    def test_macro_f1_computed(self, baseline_results):
        """Macro-F1 is a float in [0, 1]."""
        f1 = baseline_results["macro_f1"]
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_per_class_f1(self, baseline_results):
        """Per-class F1 >= 0 for all 3 classes."""
        per_class = baseline_results["per_class_f1"]
        assert set(per_class.keys()) == {"workforce", "creatives", "scientists"}
        for cls, score in per_class.items():
            assert 0.0 <= score <= 1.0, f"{cls} F1={score} out of range"

    def test_no_train_leakage(self, baseline_results):
        """Val predictions are on the val set, not the train set."""
        # Reload data fresh to get train/val labels
        _X_train, y_train, _X_val, y_val, _val_ids, _label_names = load_data()
        val_preds = baseline_results["predictions"]
        # Val predictions length must equal val set length (not train)
        assert len(val_preds) == len(y_val)
        assert len(val_preds) != len(y_train)


class TestArtifacts:
    def test_model_saved(self):
        """Model file exists after training."""
        assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"

    def test_val_preds_saved(self):
        """Predictions file exists after training."""
        assert VAL_PREDS_PATH.exists(), f"Predictions not found at {VAL_PREDS_PATH}"

    def test_model_is_logistic_regression(self):
        """Saved model is a LogisticRegression instance."""
        import joblib

        model = joblib.load(MODEL_PATH)
        assert isinstance(model, LogisticRegression)
