"""Leakage-safe feature engineering transformers."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, PowerTransformer

from automl.config import ExperimentConfig
from automl.utils import is_probably_datetime

logger = logging.getLogger(__name__)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Add log1p columns for highly skewed, strictly positive numeric features."""

    def __init__(self, skew_threshold: float = 1.0) -> None:
        self.skew_threshold = skew_threshold

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "LogTransformer":
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.n_features_in_ = df.shape[1]
        self.feature_names_in_ = (
            list(df.columns)
            if isinstance(X, pd.DataFrame)
            else [f"f{i}" for i in range(df.shape[1])]
        )
        self.skewed_indices_: list[int] = []
        self.skewed_names_: list[str] = []
        for i, col in enumerate(self.feature_names_in_):
            col_data = pd.to_numeric(df.iloc[:, i], errors="coerce").dropna()
            if len(col_data) > 10 and np.issubdtype(col_data.dtype, np.number):
                skew = float(col_data.skew())
                if abs(skew) > self.skew_threshold and bool((col_data > 0).all()):
                    self.skewed_indices_.append(i)
                    self.skewed_names_.append(str(col))
        logger.info("LogTransformer: %d skewed features found", len(self.skewed_indices_))
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if not self.skewed_indices_:
            return arr
        log_features = np.log1p(np.abs(arr[:, self.skewed_indices_]))
        return np.hstack([arr, log_features])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = list(input_features) if input_features else list(self.feature_names_in_)
        for name in self.skewed_names_:
            names.append(f"{name}_log1p")
        return np.asarray(names, dtype=object)


class PowerFeatureTransformer(BaseEstimator, TransformerMixin):
    """Yeo-Johnson / Box-Cox style transform on skewed numeric columns."""

    def __init__(self, skew_threshold: float = 1.0, method: str = "yeo-johnson") -> None:
        self.skew_threshold = skew_threshold
        self.method = method

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "PowerFeatureTransformer":
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(arr.shape[1])]
        )
        skew = pd.DataFrame(arr).skew()
        self.selected_ = [i for i, s in enumerate(skew) if np.isfinite(s) and abs(float(s)) >= self.skew_threshold]
        self.transformer_ = None
        if self.selected_:
            method = self.method
            subset = arr[:, self.selected_]
            if method == "box-cox" and not np.all(subset > 0):
                method = "yeo-johnson"
            self.transformer_ = PowerTransformer(method=method, standardize=False)
            self.transformer_.fit(subset)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if not self.selected_ or self.transformer_ is None:
            return arr
        transformed = self.transformer_.transform(arr[:, self.selected_])
        return np.hstack([arr, transformed])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = list(input_features) if input_features else list(self.feature_names_in_)
        for i in self.selected_:
            names.append(f"{names[i]}_power")
        return np.asarray(names, dtype=object)


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract calendar and cyclical datetime features. Stateless besides column detection."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "DatetimeFeatureExtractor":
        if not isinstance(X, pd.DataFrame):
            self.datetime_cols_ = []
            self.feature_names_in_ = [f"f{i}" for i in range(np.asarray(X).shape[1])]
            return self
        self.feature_names_in_ = list(X.columns)
        self.datetime_cols_ = [col for col in X.columns if is_probably_datetime(X[col])]
        logger.info("DatetimeFeatureExtractor: %d datetime columns found", len(self.datetime_cols_))
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X)
        if not self.datetime_cols_:
            return X
        df = X.copy()
        for col in self.datetime_cols_:
            if col not in df.columns:
                continue
            dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
            df[f"{col}_year"] = dt.dt.year.astype("float64")
            df[f"{col}_quarter"] = dt.dt.quarter.astype("float64")
            df[f"{col}_month"] = dt.dt.month.astype("float64")
            df[f"{col}_week"] = dt.dt.isocalendar().week.astype("float64")
            df[f"{col}_day"] = dt.dt.day.astype("float64")
            df[f"{col}_day_of_week"] = dt.dt.dayofweek.astype("float64")
            df[f"{col}_day_of_year"] = dt.dt.dayofyear.astype("float64")
            df[f"{col}_hour"] = dt.dt.hour.astype("float64")
            df[f"{col}_minute"] = dt.dt.minute.astype("float64")
            df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float64")
            df[f"{col}_is_month_start"] = dt.dt.is_month_start.astype("float64")
            df[f"{col}_is_month_end"] = dt.dt.is_month_end.astype("float64")
            month = dt.dt.month.fillna(0).astype(float)
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * month / 12.0)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * month / 12.0)
            dow = dt.dt.dayofweek.fillna(0).astype(float)
            df[f"{col}_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            df[f"{col}_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
            df = df.drop(columns=[col])
        return df

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        # Names depend on transform; caller should use pandas columns after transform.
        return np.asarray(input_features if input_features else self.feature_names_in_, dtype=object)


class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """Degree-2 polynomials on a small variance-ranked subset to avoid explosions."""

    def __init__(self, top_k: int = 6, degree: int = 2) -> None:
        self.top_k = top_k
        self.degree = degree

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "PolynomialFeatureGenerator":
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(arr.shape[1])]
        )
        n_select = min(self.top_k, arr.shape[1], 8)
        variances = np.nanvar(arr, axis=0)
        self.selected_indices_ = np.argsort(variances)[-n_select:].tolist()
        self.poly_transformer_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=False,
            include_bias=False,
        )
        self.poly_transformer_.fit(np.nan_to_num(arr[:, self.selected_indices_], nan=0.0))
        logger.info(
            "PolynomialFeatureGenerator: %d source features, %d poly outputs",
            n_select,
            self.poly_transformer_.n_output_features_,
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        poly_features = self.poly_transformer_.transform(
            np.nan_to_num(arr[:, self.selected_indices_], nan=0.0)
        )
        n_original = len(self.selected_indices_)
        return np.hstack([arr, poly_features[:, n_original:]])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is None:
            input_features = list(self.feature_names_in_)
        names = list(input_features)
        selected_names = [input_features[i] for i in self.selected_indices_]
        poly_names = self.poly_transformer_.get_feature_names_out(selected_names)
        for pn in poly_names[len(self.selected_indices_) :]:
            names.append(f"poly_{pn}")
        return np.asarray(names, dtype=object)


class InteractionRatioGenerator(BaseEstimator, TransformerMixin):
    """Create a small number of ratio / product features from top-variance columns."""

    def __init__(self, top_k: int = 5, enable_ratios: bool = True, enable_products: bool = True) -> None:
        self.top_k = top_k
        self.enable_ratios = enable_ratios
        self.enable_products = enable_products

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "InteractionRatioGenerator":
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(arr.shape[1])]
        )
        n_select = min(self.top_k, arr.shape[1], 6)
        variances = np.nanvar(arr, axis=0)
        self.selected_indices_ = sorted(np.argsort(variances)[-n_select:].tolist())
        self.pairs_ = list(combinations(self.selected_indices_, 2))[:8]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        extras = []
        for i, j in self.pairs_:
            if self.enable_products:
                extras.append(arr[:, i] * arr[:, j])
            if self.enable_ratios:
                extras.append(arr[:, i] / np.where(np.abs(arr[:, j]) < 1e-9, np.nan, arr[:, j]))
        if not extras:
            return arr
        extra = np.nan_to_num(np.column_stack(extras), nan=0.0, posinf=0.0, neginf=0.0)
        return np.hstack([arr, extra])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = list(input_features) if input_features else list(self.feature_names_in_)
        for i, j in self.pairs_:
            a, b = names[i], names[j]
            if self.enable_products:
                names.append(f"{a}_x_{b}")
            if self.enable_ratios:
                names.append(f"{a}_div_{b}")
        return np.asarray(names, dtype=object)


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Quantile bin a few numeric columns and append the bin ids."""

    def __init__(self, n_bins: int = 5, top_k: int = 4) -> None:
        self.n_bins = n_bins
        self.top_k = top_k

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "BinningTransformer":
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(arr.shape[1])]
        )
        n_select = min(self.top_k, arr.shape[1])
        variances = np.nanvar(arr, axis=0)
        self.selected_indices_ = np.argsort(variances)[-n_select:].tolist()
        self.binner_ = KBinsDiscretizer(
            n_bins=self.n_bins, encode="ordinal", strategy="quantile"
        )
        self.binner_.fit(np.nan_to_num(arr[:, self.selected_indices_], nan=0.0))
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        bins = self.binner_.transform(np.nan_to_num(arr[:, self.selected_indices_], nan=0.0))
        return np.hstack([arr, bins])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = list(input_features) if input_features else list(self.feature_names_in_)
        for i in self.selected_indices_:
            names.append(f"{names[i]}_bin")
        return np.asarray(names, dtype=object)


class FeatureEngineeringEngine(BaseEstimator, TransformerMixin):
    """Composable feature engineering block used inside CV pipelines."""

    def __init__(
        self,
        enable_log: bool = True,
        enable_power: bool = False,
        enable_polynomial: bool = False,
        enable_interactions: bool = False,
        enable_ratios: bool = False,
        enable_binning: bool = False,
        skew_threshold: float = 1.0,
        poly_top_k: int = 6,
        poly_degree: int = 2,
    ) -> None:
        self.enable_log = enable_log
        self.enable_power = enable_power
        self.enable_polynomial = enable_polynomial
        self.enable_interactions = enable_interactions
        self.enable_ratios = enable_ratios
        self.enable_binning = enable_binning
        self.skew_threshold = skew_threshold
        self.poly_top_k = poly_top_k
        self.poly_degree = poly_degree

    def _build(self) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        if self.enable_log:
            steps.append(("log", LogTransformer(self.skew_threshold)))
        if self.enable_power:
            steps.append(("power", PowerFeatureTransformer(self.skew_threshold)))
        if self.enable_polynomial:
            steps.append(("poly", PolynomialFeatureGenerator(self.poly_top_k, self.poly_degree)))
        if self.enable_interactions or self.enable_ratios:
            steps.append(
                (
                    "interact",
                    InteractionRatioGenerator(
                        top_k=5,
                        enable_ratios=self.enable_ratios,
                        enable_products=self.enable_interactions,
                    ),
                )
            )
        if self.enable_binning:
            steps.append(("bin", BinningTransformer()))
        if not steps:
            steps.append(("noop", "passthrough"))
        return Pipeline(steps)

    def fit(self, X: Any, y: Any = None) -> "FeatureEngineeringEngine":
        self.pipeline_ = self._build()
        self.pipeline_.fit(X, y)
        self.n_features_in_ = np.asarray(X).shape[1]
        self.engineered_names_ = list(self.get_feature_names_out())
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(self.pipeline_.transform(X))

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        try:
            return np.asarray(self.pipeline_.get_feature_names_out(input_features), dtype=object)
        except Exception:
            arr = np.asarray(self.pipeline_.transform(np.zeros((1, self.n_features_in_))))
            return np.asarray([f"fe_{i}" for i in range(arr.shape[1])], dtype=object)


class DataFrameLog1p(BaseEstimator, TransformerMixin):
    """Apply log1p to skewed positive numeric columns before encoding/scaling."""

    def __init__(self, skew_threshold: float = 1.0) -> None:
        self.skew_threshold = skew_threshold

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "DataFrameLog1p":
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(df.columns)
        self.cols_: list[str] = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(data) > 10 and bool((data > 0).all()) and abs(float(data.skew())) > self.skew_threshold:
                self.cols_.append(col)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        for col in self.cols_:
            if col in df.columns:
                df[f"{col}_log1p"] = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0))
        return df

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = list(input_features) if input_features else list(self.feature_names_in_)
        return np.asarray(names + [f"{c}_log1p" for c in self.cols_], dtype=object)


def build_feature_engineer(task_type: str, config: ExperimentConfig | None = None) -> Pipeline:
    """Build a feature engineering pipeline. Kept for backward compatibility."""
    config = config or ExperimentConfig()
    if not config.enable_feature_engineering:
        return Pipeline([("noop", "passthrough")])
    engine = FeatureEngineeringEngine(
        enable_log=config.enable_log_transform,
        enable_power=False,
        enable_polynomial=config.enable_polynomial,
        enable_interactions=config.enable_interactions,
        enable_ratios=config.enable_ratios,
        enable_binning=config.enable_binning,
        skew_threshold=config.skew_threshold,
        poly_top_k=config.poly_top_k,
        poly_degree=config.poly_degree,
    )
    return Pipeline([("engine", engine)])
