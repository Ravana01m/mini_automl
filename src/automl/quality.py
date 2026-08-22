"""Leakage-safe data-quality transformers and outlier handling."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from automl.config import ExperimentConfig, OutlierMethod, OutlierStrategy
from automl.types import OutlierReport
from automl.utils import get_column_types

logger = logging.getLogger(__name__)


def _modified_zscore(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers using the IQR method. Thresholds are learned on fit() only."""

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "OutlierClipper":
        if isinstance(X, pd.DataFrame):
            data = X.select_dtypes(include=[np.number])
        else:
            data = pd.DataFrame(X)

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = (q1 - self.factor * iqr).to_dict()
        self.upper_bounds_ = (q3 + self.factor * iqr).to_dict()
        self.feature_names_in_ = list(data.columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        is_df = isinstance(X, pd.DataFrame)
        df = X.copy() if is_df else pd.DataFrame(X, columns=self.feature_names_in_)
        for col in self.feature_names_in_:
            if col in df.columns:
                df[col] = df[col].clip(
                    lower=self.lower_bounds_.get(col),
                    upper=self.upper_bounds_.get(col),
                )
        return df if is_df else df.values

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = input_features if input_features else self.feature_names_in_
        return np.asarray(names, dtype=object)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Configurable outlier handler. Bounds are fit on training data only."""

    def __init__(
        self,
        method: str = "iqr",
        strategy: str = "clip",
        factor: float = 1.5,
        zscore: float = 3.0,
        winsor_limits: tuple[float, float] = (0.01, 0.99),
    ) -> None:
        self.method = method
        self.strategy = strategy
        self.factor = factor
        self.zscore = zscore
        self.winsor_limits = winsor_limits

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "OutlierHandler":
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        numeric_cols, _ = get_column_types(df)
        self.feature_names_in_ = list(df.columns)
        self.numeric_cols_ = numeric_cols
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.reports_: list[OutlierReport] = []

        if self.strategy == OutlierStrategy.NONE.value:
            return self

        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            clean = values[np.isfinite(values)]
            if clean.size < 8:
                continue
            lower, upper = self._learn_bounds(clean)
            mask = (values < lower) | (values > upper)
            count = int(np.nansum(mask))
            self.bounds_[col] = (lower, upper)
            self.reports_.append(
                OutlierReport(
                    feature=col,
                    method=self.method,
                    strategy=self.strategy,
                    outlier_count=count,
                    outlier_pct=float(count / max(len(values), 1) * 100),
                    lower_bound=lower,
                    upper_bound=upper,
                )
            )
        logger.info("OutlierHandler fitted bounds for %d numeric columns", len(self.bounds_))
        return self

    def _learn_bounds(self, values: np.ndarray) -> tuple[float, float]:
        method = self.method
        if method == OutlierMethod.IQR.value:
            q1, q3 = np.quantile(values, [0.25, 0.75])
            iqr = q3 - q1
            return float(q1 - self.factor * iqr), float(q3 + self.factor * iqr)
        if method == OutlierMethod.ZSCORE.value:
            mean, std = float(np.mean(values)), float(np.std(values) or 1.0)
            return mean - self.zscore * std, mean + self.zscore * std
        if method == OutlierMethod.MODIFIED_ZSCORE.value:
            scores = _modified_zscore(values)
            inliers = values[np.abs(scores) <= self.zscore]
            if inliers.size == 0:
                return float(np.min(values)), float(np.max(values))
            return float(np.min(inliers)), float(np.max(inliers))
        low, high = self.winsor_limits
        return float(np.quantile(values, low)), float(np.quantile(values, high))

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        if self.strategy == OutlierStrategy.NONE.value:
            return df
        for col, (lower, upper) in self.bounds_.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if self.strategy in {OutlierStrategy.CLIP.value, OutlierStrategy.WINSORIZE.value}:
                df[col] = series.clip(lower=lower, upper=upper)
            elif self.strategy == OutlierStrategy.REPLACE.value:
                median = float(series.median()) if series.notna().any() else 0.0
                df[col] = series.where((series >= lower) & (series <= upper), median)
        return df

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(names, dtype=object)

    def report_frame(self) -> pd.DataFrame:
        rows = [r.__dict__ for r in getattr(self, "reports_", [])]
        return pd.DataFrame(rows)


class ColumnPruner(BaseEstimator, TransformerMixin):
    """Drop uninformative columns using training-set statistics only."""

    def __init__(
        self,
        drop_constant: bool = True,
        drop_all_null: bool = True,
        drop_id_like: bool = False,
        extra_drop: list[str] | None = None,
    ) -> None:
        self.drop_constant = drop_constant
        self.drop_all_null = drop_all_null
        self.drop_id_like = drop_id_like
        self.extra_drop = extra_drop

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "ColumnPruner":
        from automl.profiling import profile_dataframe

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(df.columns)
        profile = profile_dataframe(df)
        drop: dict[str, str] = {}
        if self.drop_all_null:
            for col in profile.all_null:
                drop[col] = "all-null on training data"
        if self.drop_constant:
            for col in profile.constant:
                drop[col] = "constant on training data"
        if self.drop_id_like:
            for col in profile.id_like:
                drop[col] = "identifier-like on training data"
        for col in self.extra_drop or []:
            if col in df.columns:
                drop[col] = "explicitly excluded"
        # Never drop every column
        keep = [c for c in df.columns if c not in drop]
        if not keep:
            drop = {}
        self.drop_reasons_ = drop
        self.keep_columns_ = [c for c in df.columns if c not in drop]
        logger.info("ColumnPruner dropping %d columns: %s", len(drop), drop)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        existing = [c for c in self.keep_columns_ if c in df.columns]
        return df.loc[:, existing].copy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(self.keep_columns_, dtype=object)


def outlier_handler_from_config(config: ExperimentConfig) -> OutlierHandler:
    return OutlierHandler(
        method=config.outlier_method.value,
        strategy=config.outlier_strategy.value,
        factor=config.outlier_factor,
        zscore=config.outlier_zscore,
        winsor_limits=config.winsor_limits,
    )
