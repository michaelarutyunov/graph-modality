"""Sklearn classifier wrapper — matches Phase 5 frozen-embedding interface.

Wraps any sklearn classifier behind the same ``.fit(embeddings_dict, labels)`` /
``.predict(embeddings_dict)`` contract used by the PyTorch models. Adding a new
sklearn classifier is a one-line registration in ``SKLEARN_CLASSES``.

Supports both single-modality (stats-only, text-only) and multi-modality
(text+stats, text+graph, etc.) — multi-modality simply concatenates features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

SKLEARN_CLASSES: dict[str, type] = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boost": GradientBoostingClassifier,
    "svm": SVC,
}


class SklearnClassifier:
    """Wraps an sklearn classifier to consume frozen modality embeddings.

    Handles single or multi-modality input by concatenating selected modalities.
    The ``.fit()`` / ``.predict()`` contract mirrors the PyTorch models so
    the same ``run.py`` can dispatch to either backend.
    """

    def __init__(
        self,
        architecture: str,
        modality_names: list[str],
        n_classes: int,
        seed: int = 42,
    ):
        if architecture not in SKLEARN_CLASSES:
            raise ValueError(
                f"Unknown sklearn architecture: {architecture}. "
                f"Choose from: {list(SKLEARN_CLASSES.keys())}"
            )

        self.architecture = architecture
        self.modality_names = sorted(modality_names)
        self.n_classes = n_classes

        cls = SKLEARN_CLASSES[architecture]
        if architecture == "gradient_boost":
            self.model: ClassifierMixin = cls(random_state=seed)
        else:
            self.model = cls(class_weight="balanced", random_state=seed)

    def _extract(self, embeddings: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate selected modalities into a feature matrix.

        Args:
            embeddings: Dict mapping modality name → (N, dim) array.

        Returns:
            Feature matrix of shape (N, total_dim).
        """
        ordered = [embeddings[m] for m in self.modality_names]
        if len(ordered) == 1:
            return ordered[0]
        return np.concatenate(ordered, axis=1)

    def fit(
        self,
        embeddings: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> SklearnClassifier:
        """Train the sklearn classifier.

        Args:
            embeddings: Dict mapping modality name → (N, dim) array.
            labels: Integer labels of shape (N,).

        Returns:
            self (for chaining).
        """
        X = self._extract(embeddings)
        self.model.fit(X, labels)
        return self

    def predict(self, embeddings: dict[str, np.ndarray]) -> np.ndarray:
        """Predict class labels.

        Args:
            embeddings: Dict mapping modality name → (N, dim) array.

        Returns:
            Predicted integer labels of shape (N,).
        """
        X = self._extract(embeddings)
        return self.model.predict(X)

    def predict_proba(self, embeddings: dict[str, np.ndarray]) -> np.ndarray:
        """Predict class probabilities (if supported by the underlying model).

        Args:
            embeddings: Dict mapping modality name → (N, dim) array.

        Returns:
            Probability matrix of shape (N, n_classes).
        """
        X = self._extract(embeddings)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # Fall back to one-hot predictions
        preds = self.model.predict(X)
        proba = np.zeros((len(preds), self.n_classes))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba
