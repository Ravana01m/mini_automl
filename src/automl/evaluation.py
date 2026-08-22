"""Centralized evaluation metrics for classification and regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_CLASSIFICATION = "f1_weighted"
PRIMARY_REGRESSION = "neg_root_mean_squared_error"


def _as_np(values: Any) -> np.ndarray:
    return np.asarray(values)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true = _as_np(y_true).astype(float)
    y_pred = _as_np(y_pred).astype(float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }
    if np.all(np.abs(y_true) > 1e-8):
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    else:
        metrics["mape"] = float("nan")
    return metrics


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(roc_auc_score(y_true, proba))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba
            return float(average_precision_score(y_true, scores))
        return float(average_precision_score(y_true, proba, average="weighted"))
    except Exception:
        return float("nan")


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = _as_np(y_true)
    y_pred = _as_np(y_pred)
    n_classes = int(len(np.unique(y_true)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = _safe_roc_auc(y_true, y_proba, n_classes)
        metrics["pr_auc"] = _safe_pr_auc(y_true, y_proba, n_classes)
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            metrics["log_loss"] = float("nan")
    return metrics


def evaluate_model(
    estimator: Any,
    X: Any,
    y: Any,
    task_type: str,
) -> dict[str, float]:
    y_pred = estimator.predict(X)
    if task_type == "classification":
        proba = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
        return classification_metrics(y, y_pred, proba)
    return regression_metrics(y, y_pred)


def summarize_cv(scores: list[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci)}


def classification_report_frame(y_true: Any, y_pred: Any) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T


def percent_change(baseline: float, advanced: float, higher_is_better: bool = True) -> float:
    if baseline is None or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return float("nan")
    change = (advanced - baseline) / abs(baseline) * 100
    return float(change)


def scoring_for_task(task_type: str) -> dict[str, str]:
    if task_type == "classification":
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_weighted": "f1_weighted",
            "f1_macro": "f1_macro",
        }
    return {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }
