"""Validation helpers, column typing, and logging."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Custom exception for validation errors."""


MAX_ROWS = 1_000_000
MAX_COLS = 500


def validate_csv(df: pd.DataFrame, filename: str = "dataset") -> None:
    """Validate the structure of the input dataset."""
    if df.empty:
        raise ValidationError(f"Dataset in {filename} is empty.")
    if len(df.columns) < 2:
        raise ValidationError(f"Dataset in {filename} must have at least 2 columns.")
    if len(df) > MAX_ROWS:
        raise ValidationError(
            f"Dataset in {filename} exceeds maximum allowed rows ({MAX_ROWS})."
        )
    if len(df.columns) > MAX_COLS:
        raise ValidationError(
            f"Dataset in {filename} exceeds maximum allowed columns ({MAX_COLS})."
        )
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValidationError(f"Duplicate column names are not allowed: {dupes}")


def validate_target_column(df: pd.DataFrame, target_col: str) -> None:
    """Validate the target column for modeling."""
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found in dataset.")
    if df[target_col].isnull().all():
        raise ValidationError(f"Target column '{target_col}' contains only null values.")
    if df[target_col].nunique(dropna=True) < 2:
        raise ValidationError(
            f"Target column '{target_col}' must have at least 2 unique values."
        )
    missing_pct = float(df[target_col].isnull().mean() * 100)
    if missing_pct > 40:
        raise ValidationError(
            f"Target column '{target_col}' has {missing_pct:.1f}% missing values."
        )


def get_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separate columns into numeric and categorical, ignoring all-null columns."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if df[col].isnull().all():
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(
            df[col]
        ):
            numeric_cols.append(col)
        elif (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[col])
        ):
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype(str).nunique(dropna=True))


def is_probably_datetime(series: pd.Series, sample_size: int = 40) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(sample_size)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= 0.8


def is_id_like(name: str, series: pd.Series) -> bool:
    lowered = name.lower()
    id_tokens = ("id", "uuid", "guid", "index", "pk", "row_id", "record_id")
    if any(token == lowered or lowered.endswith(f"_{token}") or lowered.startswith(f"{token}_") for token in id_tokens):
        if safe_nunique(series) >= max(20, int(0.9 * series.dropna().shape[0])):
            return True
    n_unique = safe_nunique(series)
    n_non_null = int(series.notna().sum())
    if n_non_null > 20 and n_unique == n_non_null:
        if series.dtype == object or pd.api.types.is_integer_dtype(series):
            return True
    return False


def is_text_like(series: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(80)
    if sample.empty:
        return False
    mean_len = float(sample.str.len().mean())
    return mean_len >= 40 or bool((sample.str.contains(r"\s")).mean() > 0.6 and mean_len > 12)


def setup_logging(level: int = logging.INFO) -> None:
    """Set up standard logging for the automl package."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def flatten_params(estimator: Any) -> int | None:
    try:
        params = estimator.get_params(deep=False)
        return len(params)
    except Exception:
        return None
