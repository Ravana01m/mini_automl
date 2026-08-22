"""Task-type detection and imbalance diagnostics."""

from __future__ import annotations

import pandas as pd

MAX_UNIQUE_FOR_CLASSIFICATION = 20


def detect_task_type(target_series: pd.Series) -> str:
    """Detect classification vs regression from the target series."""
    if target_series.empty or target_series.isnull().all():
        raise ValueError("Target series is empty or contains all nulls.")

    if (
        pd.api.types.is_object_dtype(target_series)
        or pd.api.types.is_string_dtype(target_series)
        or isinstance(target_series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(target_series)
    ):
        return "classification"

    if pd.api.types.is_numeric_dtype(target_series):
        if target_series.nunique(dropna=True) <= MAX_UNIQUE_FOR_CLASSIFICATION:
            return "classification"
        return "regression"

    return "regression"


def classify_problem(y: pd.Series) -> str:
    """Return binary / multiclass / continuous."""
    if detect_task_type(y) == "regression":
        return "continuous"
    return "binary" if y.nunique(dropna=True) == 2 else "multiclass"


def imbalance_report(y: pd.Series) -> dict[str, object]:
    counts = y.value_counts(dropna=True)
    total = float(counts.sum()) or 1.0
    pct = {str(k): float(v / total * 100) for k, v in counts.items()}
    ratio = None
    if len(counts) >= 2:
        ratio = float(counts.max() / max(counts.min(), 1))
    return {
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_pct": pct,
        "imbalance_ratio": ratio,
        "is_imbalanced": bool(ratio is not None and ratio >= 3.0),
    }
