"""Model diagnostics for regression and classification."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from automl.evaluation import classification_metrics, classification_report_frame, regression_metrics


def regression_diagnostics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    metrics = regression_metrics(y_true, y_pred)
    notes = []
    if abs(float(np.mean(residuals))) > 0.25 * (np.std(residuals) + 1e-9):
        notes.append("Residuals appear biased (mean is away from zero).")
    if float(pd.Series(residuals).skew()) > 1.0:
        notes.append("Residual distribution is right-skewed.")
    elif float(pd.Series(residuals).skew()) < -1.0:
        notes.append("Residual distribution is left-skewed.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].scatter(y_pred, y_true, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[0, 0].set_title("Actual vs Predicted")
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(0, color="r", linestyle="--")
    axes[0, 1].set_title("Residual vs Predicted")
    axes[1, 0].hist(residuals, bins=20, color="#667eea")
    axes[1, 0].set_title("Residual Distribution")
    axes[1, 1].hist(np.abs(residuals), bins=20, color="#764ba2")
    axes[1, 1].set_title("Absolute Error Distribution")
    fig.tight_layout()
    return {"metrics": metrics, "figure": fig, "notes": notes, "residuals": residuals}


def classification_diagnostics(
    y_true: Any,
    y_pred: Any,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = classification_metrics(y_true, y_pred, y_proba)
    report = classification_report_frame(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")

    roc_fig = None
    pr_fig = None
    cal_fig = None
    n_classes = len(np.unique(y_true))
    if y_proba is not None and n_classes == 2:
        scores = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] >= 2 else y_proba
        roc_fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_true, scores, ax=ax)
        ax.set_title("ROC Curve")
        pr_fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax)
        ax.set_title("Precision-Recall Curve")
        try:
            frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=8)
            cal_fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(mean_pred, frac_pos, marker="o")
            ax.plot([0, 1], [0, 1], "--")
            ax.set_title("Calibration Curve")
        except Exception:
            cal_fig = None

    proba_fig = None
    if y_proba is not None:
        proba_fig, ax = plt.subplots(figsize=(5, 4))
        if y_proba.ndim == 2:
            ax.hist(y_proba.max(axis=1), bins=15, color="#667eea")
            ax.set_title("Max Class Probability")
        else:
            ax.hist(y_proba, bins=15, color="#667eea")
            ax.set_title("Predicted Probability")

    notes = []
    if metrics.get("accuracy", 0) - metrics.get("balanced_accuracy", 0) > 0.1:
        notes.append("Accuracy is inflated relative to balanced accuracy; prefer F1 / PR-AUC.")
    return {
        "metrics": metrics,
        "report": report,
        "confusion_figure": fig_cm,
        "roc_figure": roc_fig,
        "pr_figure": pr_fig,
        "calibration_figure": cal_fig,
        "proba_figure": proba_fig,
        "notes": notes,
    }
