"""Tests for evaluation metrics and honest comparison."""

from __future__ import annotations

import numpy as np

from automl.evaluation import classification_metrics, percent_change, regression_metrics


def test_regression_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.8, 3.2, 3.9])
    metrics = regression_metrics(y, pred)
    assert metrics["rmse"] > 0
    assert 0 <= metrics["r2"] <= 1


def test_classification_binary_and_multiclass() -> None:
    y = np.array([0, 1, 0, 1, 1])
    pred = np.array([0, 1, 0, 0, 1])
    proba = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9]])
    metrics = classification_metrics(y, pred, proba)
    assert "f1_weighted" in metrics
    assert "roc_auc" in metrics

    y_m = np.array([0, 1, 2, 1, 2, 0])
    pred_m = np.array([0, 1, 1, 1, 2, 0])
    multi = classification_metrics(y_m, pred_m)
    assert multi["f1_macro"] >= 0


def test_percent_change_is_honest() -> None:
    assert percent_change(1.0, 1.1, True) > 0
    # Raw relative change: RMSE rising from 2.0 to 2.4 is +20%, not hidden.
    assert abs(percent_change(2.0, 2.4, False) - 20.0) < 1e-6
    assert np.isnan(percent_change(0.0, 1.0, True))
