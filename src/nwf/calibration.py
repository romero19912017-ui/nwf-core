# -*- coding: utf-8 -*-
"""Calibrators for confidence estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression


class Calibrator(ABC):
    """Abstract calibrator."""

    @abstractmethod
    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> "Calibrator":
        """Fit on validation data. labels: 0/1 correct/incorrect."""
        pass

    @abstractmethod
    def predict(self, confidences: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        pass


class AgreementRatio:
    """Agreement ratio: fraction of k nearest neighbors with same predicted label.

    No training required. Use with search results: pass neighbor_labels and
    predicted_label to compute confidence.
    """

    def predict(
        self,
        neighbor_labels: Union[List[Any], np.ndarray],
        predicted_label: Any,
    ) -> float:
        """Compute agreement ratio.

        Args:
            neighbor_labels: labels of k nearest neighbors
            predicted_label: the predicted (winning) label

        Returns:
            Fraction of neighbors that agree with prediction.
        """
        if isinstance(neighbor_labels, np.ndarray):
            neighbor_labels = neighbor_labels.flatten().tolist()
        n = len(neighbor_labels)
        if n == 0:
            return 0.0
        agree = sum(1 for lab in neighbor_labels if lab == predicted_label)
        return agree / n

    def predict_batch(
        self,
        neighbor_labels_batch: List[List[Any]],
        predicted_labels: List[Any],
    ) -> np.ndarray:
        """Batch version."""
        return np.array(
            [
                self.predict(nl, pl)
                for nl, pl in zip(neighbor_labels_batch, predicted_labels)
            ]
        )


class PlattScaler(Calibrator):
    """Platt scaling: logistic regression on confidence -> probability."""

    def __init__(self) -> None:
        self._model: Optional[LogisticRegression] = None

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        """Fit. labels: 1 = correct, 0 = incorrect."""
        confidences = np.asarray(confidences).reshape(-1, 1)
        labels = np.asarray(labels).ravel()
        self._model = LogisticRegression(C=1e10, max_iter=1000)
        self._model.fit(confidences, labels)
        return self

    def predict(self, confidences: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        if self._model is None:
            raise RuntimeError("PlattScaler not fitted")
        c = np.asarray(confidences).reshape(-1, 1)
        return self._model.predict_proba(c)[:, 1]
