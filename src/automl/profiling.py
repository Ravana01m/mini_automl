"""Automated data profiling engine."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from automl.detector import detect_task_type, imbalance_report
from automl.types import ColumnKind, ColumnProfile, DataProfile
from automl.utils import is_id_like, is_probably_datetime, is_text_like, safe_nunique

logger = logging.getLogger(__name__)

NEAR_CONSTANT_RATIO = 0.95
HIGH_MISSING_PCT = 40.0
HIGH_CARDINALITY = 50
LEAKAGE_CORR = 0.98
ID_CORR_HINT = 0.999


def _iqr_outlier_count(series: pd.Series) -> int:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.shape[0] < 8:
        return 0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def _classify_column(name: str, series: pd.Series) -> ColumnKind:
    if series.dropna().empty:
        return ColumnKind.ALL_NULL
    if is_probably_datetime(series):
        return ColumnKind.DATETIME
    if pd.api.types.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if is_text_like(series):
        return ColumnKind.TEXT
    if is_id_like(name, series):
        return ColumnKind.ID_LIKE
    n_unique = safe_nunique(series)
    if n_unique <= 1:
        return ColumnKind.CONSTANT
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return ColumnKind.NUMERICAL
    return ColumnKind.CATEGORICAL


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    desc = numeric.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    outliers = _iqr_outlier_count(numeric)
    n = max(int(numeric.notna().sum()), 1)
    return {
        "mean": _safe_float(desc.get("mean")),
        "median": _safe_float(desc.get("50%")),
        "std": _safe_float(desc.get("std")),
        "min": _safe_float(desc.get("min")),
        "max": _safe_float(desc.get("max")),
        "q01": _safe_float(desc.get("1%")),
        "q25": _safe_float(desc.get("25%")),
        "q50": _safe_float(desc.get("50%")),
        "q75": _safe_float(desc.get("75%")),
        "q99": _safe_float(desc.get("99%")),
        "skewness": _safe_float(numeric.skew()),
        "outlier_count": outliers,
        "outlier_pct": float(outliers / n * 100),
        "has_inf": bool(np.isinf(numeric.fillna(0)).any()),
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def profile_dataframe(df: pd.DataFrame, target_col: str | None = None) -> DataProfile:
    """Build a structured DataProfile. Never silently drops columns."""
    n_rows, n_cols = df.shape
    memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
    missing_cells = int(df.isna().sum().sum())
    columns: dict[str, ColumnProfile] = {}

    buckets: dict[str, list[str]] = {
        "numerical": [],
        "categorical": [],
        "boolean": [],
        "datetime": [],
        "text_like": [],
        "id_like": [],
        "constant": [],
        "near_constant": [],
        "all_null": [],
        "high_missing": [],
        "high_cardinality": [],
        "infinite": [],
        "skewed": [],
        "potential_leakage": [],
    }

    for name in df.columns:
        series = df[name]
        kind = _classify_column(name, series)
        n_unique = safe_nunique(series)
        missing_count = int(series.isna().sum())
        missing_pct = float(missing_count / max(n_rows, 1) * 100)
        notes: list[str] = []

        top_freq = 0.0
        if n_rows and n_unique:
            top_freq = float(series.value_counts(dropna=True).iloc[0] / n_rows) if n_unique else 0.0
        is_near_constant = n_unique > 1 and top_freq >= NEAR_CONSTANT_RATIO
        is_high_missing = missing_pct >= HIGH_MISSING_PCT
        is_high_card = kind in {ColumnKind.CATEGORICAL, ColumnKind.TEXT} and n_unique >= HIGH_CARDINALITY
        id_like = kind == ColumnKind.ID_LIKE or is_id_like(name, series)
        text_like = kind == ColumnKind.TEXT

        stats = _numeric_stats(series) if kind == ColumnKind.NUMERICAL else {}
        if kind == ColumnKind.NUMERICAL and stats.get("has_inf"):
            notes.append("Contains infinite values")
            buckets["infinite"].append(name)
        if kind == ColumnKind.NUMERICAL and stats.get("skewness") is not None:
            if abs(stats["skewness"]) >= 1.0:
                buckets["skewed"].append(name)
                notes.append(f"Skewed ({stats['skewness']:.2f})")

        if is_near_constant:
            notes.append("Near-constant majority class/value")
            buckets["near_constant"].append(name)
        if is_high_missing:
            notes.append(f"High missingness ({missing_pct:.1f}%)")
            buckets["high_missing"].append(name)
        if is_high_card:
            notes.append(f"High cardinality ({n_unique})")
            buckets["high_cardinality"].append(name)
        if id_like:
            notes.append("Looks like an identifier; review before dropping")
            buckets["id_like"].append(name)
        if text_like:
            notes.append("Text-like column; not used as a free-text model feature")
            buckets["text_like"].append(name)
        if kind == ColumnKind.CONSTANT:
            notes.append("Constant column")
            buckets["constant"].append(name)
        if kind == ColumnKind.ALL_NULL:
            notes.append("All values are null")
            buckets["all_null"].append(name)

        kind_bucket = {
            ColumnKind.NUMERICAL: "numerical",
            ColumnKind.CATEGORICAL: "categorical",
            ColumnKind.BOOLEAN: "boolean",
            ColumnKind.DATETIME: "datetime",
        }.get(kind)
        if kind_bucket:
            buckets[kind_bucket].append(name)

        columns[name] = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            kind=kind,
            n_unique=n_unique,
            cardinality=n_unique,
            missing_count=missing_count,
            missing_pct=missing_pct,
            is_constant=kind == ColumnKind.CONSTANT,
            is_near_constant=is_near_constant,
            is_high_cardinality=is_high_card,
            is_high_missing=is_high_missing,
            is_id_like=id_like,
            is_text_like=text_like,
            notes=notes,
            **stats,
        )

    recommendations: list[str] = []
    if buckets["id_like"]:
        recommendations.append(
            f"Review identifier-like columns before modeling: {buckets['id_like']}"
        )
    if buckets["high_missing"]:
        recommendations.append(
            "High-missing columns will receive missing indicators; consider dropping only after review."
        )
    if buckets["constant"] or buckets["all_null"]:
        recommendations.append(
            "Constant / all-null columns can be dropped safely because they carry no information."
        )

    class_counts: dict[str, int] = {}
    class_pct: dict[str, float] = {}
    imbalance_ratio = None
    task_type = None
    if target_col and target_col in df.columns:
        try:
            task_type = detect_task_type(df[target_col])
        except ValueError:
            task_type = None
        if task_type == "classification":
            imb = imbalance_report(df[target_col])
            class_counts = imb["class_counts"]  # type: ignore[assignment]
            class_pct = imb["class_pct"]  # type: ignore[assignment]
            imbalance_ratio = imb["imbalance_ratio"]  # type: ignore[assignment]
            if imb["is_imbalanced"]:
                recommendations.append(
                    f"Target is imbalanced (ratio={imbalance_ratio:.2f}). Prefer balanced accuracy, macro F1, and PR-AUC."
                )
        buckets["potential_leakage"] = _find_leakage_columns(df, target_col)
        if buckets["potential_leakage"]:
            recommendations.append(
                "Potential leakage columns detected (very high association with the target). Review before training."
            )

    profile = DataProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        memory_mb=round(memory_mb, 3),
        duplicate_rows=int(df.duplicated().sum()),
        missing_cells=missing_cells,
        missing_pct=float(missing_cells / max(n_rows * n_cols, 1) * 100),
        columns=columns,
        target=target_col,
        task_type=task_type,
        class_counts=class_counts,
        class_pct=class_pct,
        imbalance_ratio=imbalance_ratio,
        recommendations=recommendations,
        **buckets,
    )
    logger.info(
        "Profiled dataset %s rows x %s cols (missing=%.2f%%, duplicates=%s)",
        n_rows,
        n_cols,
        profile.missing_pct,
        profile.duplicate_rows,
    )
    return profile


def _find_leakage_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    suspects: list[str] = []
    target = df[target_col]
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        try:
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
                corr = float(pd.Series(series).corr(pd.Series(target), method="pearson"))
                if np.isfinite(corr) and abs(corr) >= LEAKAGE_CORR:
                    suspects.append(col)
            elif series.nunique(dropna=True) == target.nunique(dropna=True):
                if series.astype(str).fillna("__NA__").equals(target.astype(str).fillna("__NA__")):
                    suspects.append(col)
        except Exception:
            continue
    return suspects
