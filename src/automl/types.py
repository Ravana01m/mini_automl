"""Shared dataclasses and type aliases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

import pandas as pd

TaskName = Literal["classification", "regression"]
ModelFamily = Literal["linear", "tree", "boosting", "svm", "neural", "ensemble"]


class ColumnKind(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ID_LIKE = "id_like"
    CONSTANT = "constant"
    ALL_NULL = "all_null"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: ColumnKind
    n_unique: int
    cardinality: int
    missing_count: int
    missing_pct: float
    is_constant: bool = False
    is_near_constant: bool = False
    is_high_cardinality: bool = False
    is_high_missing: bool = False
    is_id_like: bool = False
    is_text_like: bool = False
    has_inf: bool = False
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q01: float | None = None
    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    q99: float | None = None
    skewness: float | None = None
    outlier_count: int = 0
    outlier_pct: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    memory_mb: float
    duplicate_rows: int
    missing_cells: int
    missing_pct: float
    columns: dict[str, ColumnProfile]
    numerical: list[str]
    categorical: list[str]
    boolean: list[str]
    datetime: list[str]
    text_like: list[str]
    id_like: list[str]
    constant: list[str]
    near_constant: list[str]
    all_null: list[str]
    high_missing: list[str]
    high_cardinality: list[str]
    infinite: list[str]
    skewed: list[str]
    potential_leakage: list[str]
    class_counts: dict[str, int] = field(default_factory=dict)
    class_pct: dict[str, float] = field(default_factory=dict)
    imbalance_ratio: float | None = None
    target: str | None = None
    task_type: str | None = None
    recommendations: list[str] = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for col in self.columns.values():
            rows.append(
                {
                    "column": col.name,
                    "kind": col.kind.value,
                    "dtype": col.dtype,
                    "n_unique": col.n_unique,
                    "missing_pct": round(col.missing_pct, 2),
                    "cardinality": col.cardinality,
                    "skewness": col.skewness,
                    "outlier_count": col.outlier_count,
                    "is_id_like": col.is_id_like,
                    "is_constant": col.is_constant,
                    "notes": "; ".join(col.notes),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class OutlierReport:
    feature: str
    method: str
    strategy: str
    outlier_count: int
    outlier_pct: float
    lower_bound: float | None
    upper_bound: float | None


@dataclass
class FeatureSelectionReport:
    original_features: list[str]
    engineered_features: list[str]
    removed_features: list[str]
    selected_features: list[str]
    method: str
    reasons: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelSpec:
    name: str
    family: ModelFamily
    task: TaskName
    factory: Callable[..., Any]
    param_grid: dict[str, list[Any]]
    optuna_space: dict[str, Any]
    needs_scaling: bool
    has_predict_proba: bool
    primary_metric: str
    stage: int = 1
    supports_class_weight: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeaderboardRow:
    model: str
    family: str
    cv_score: float | None
    cv_std: float | None
    test_score: float | None
    train_time_s: float
    inference_time_s: float | None
    n_params: int | None
    status: str
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
