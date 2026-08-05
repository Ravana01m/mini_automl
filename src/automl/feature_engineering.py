"""Automated feature engineering transformers.

All transformers follow the sklearn BaseEstimator/TransformerMixin interface
so they compose cleanly into the main Pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transform to highly skewed numeric features.

    Adds new log-transformed columns for features with |skewness| > threshold.
    Original columns are preserved.
    """

    def __init__(self, skew_threshold: float = 1.0) -> None:
        self.skew_threshold = skew_threshold

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "LogTransformer":
        """Identify features with high skewness."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.n_features_in_ = df.shape[1]
        self.feature_names_in_ = (
            list(df.columns) if isinstance(X, pd.DataFrame)
            else [f"f{i}" for i in range(df.shape[1])]
        )
        
        # Only consider numeric columns with enough non-null values
        self.skewed_indices_: list[int] = []
        self.skewed_names_: list[str] = []
        for i, col in enumerate(self.feature_names_in_):
            col_data = df.iloc[:, i].dropna()
            if len(col_data) > 10 and np.issubdtype(col_data.dtype, np.number):
                skew = col_data.skew()
                if abs(skew) > self.skew_threshold and (col_data > 0).all():
                    self.skewed_indices_.append(i)
                    self.skewed_names_.append(col)
        
        logger.info("LogTransformer: %d skewed features found", len(self.skewed_indices_))
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Add log1p columns for skewed features."""
        arr = X if isinstance(X, np.ndarray) else X.values
        if not self.skewed_indices_:
            return arr
        
        log_features = np.log1p(np.abs(arr[:, self.skewed_indices_]))
        return np.hstack([arr, log_features])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return feature names including new log features."""
        names = list(input_features) if input_features else list(self.feature_names_in_)
        for name in self.skewed_names_:
            names.append(f"{name}_log1p")
        return names


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from datetime columns.

    Detects datetime columns and extracts: year, month, day, day_of_week,
    is_weekend. Original datetime column is dropped.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DatetimeFeatureExtractor":
        """Detect datetime columns."""
        if not isinstance(X, pd.DataFrame):
            self.datetime_cols_: list[str] = []
            return self
        
        self.datetime_cols_ = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.datetime_cols_.append(col)
            elif X[col].dtype == object:
                try:
                    pd.to_datetime(X[col].dropna().head(50), infer_datetime_format=True)
                    self.datetime_cols_.append(col)
                except (ValueError, TypeError):
                    pass
        
        logger.info("DatetimeFeatureExtractor: %d datetime columns found", len(self.datetime_cols_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime features."""
        if not isinstance(X, pd.DataFrame) or not self.datetime_cols_:
            return X
        
        df = X.copy()
        for col in self.datetime_cols_:
            if col not in df.columns:
                continue
            dt = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = dt.dt.year.astype("float64")
            df[f"{col}_month"] = dt.dt.month.astype("float64")
            df[f"{col}_day"] = dt.dt.day.astype("float64")
            df[f"{col}_dayofweek"] = dt.dt.dayofweek.astype("float64")
            df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float64")
            df = df.drop(columns=[col])
        
        return df


class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate polynomial interaction features for top-K numeric features.

    Uses variance as a simple proxy to select the top-K features,
    then generates degree-2 polynomial features (including interactions).
    """

    def __init__(self, top_k: int = 10, degree: int = 2) -> None:
        self.top_k = top_k
        self.degree = degree

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "PolynomialFeatureGenerator":
        """Select top-K features by variance and fit polynomial transformer."""
        arr = X if isinstance(X, np.ndarray) else X.values
        self.n_features_in_ = arr.shape[1]
        
        # Select top-K by variance (simple, robust proxy)
        n_select = min(self.top_k, arr.shape[1])
        variances = np.nanvar(arr, axis=0)
        self.selected_indices_ = np.argsort(variances)[-n_select:].tolist()
        
        self.poly_transformer_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=False,
            include_bias=False,
        )
        self.poly_transformer_.fit(arr[:, self.selected_indices_])
        
        logger.info(
            "PolynomialFeatureGenerator: %d features selected, %d poly features generated",
            n_select,
            self.poly_transformer_.n_output_features_,
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generate polynomial features and append to original array."""
        arr = X if isinstance(X, np.ndarray) else X.values
        
        poly_features = self.poly_transformer_.transform(arr[:, self.selected_indices_])
        # Remove the original features from poly output (they're already in arr)
        # PolynomialFeatures with include_bias=False starts with original features
        n_original = len(self.selected_indices_)
        new_features = poly_features[:, n_original:]
        
        return np.hstack([arr, new_features])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return feature names including polynomial features."""
        if input_features is None:
            input_features = [f"f{i}" for i in range(self.n_features_in_)]
        
        names = list(input_features)
        selected_names = [input_features[i] for i in self.selected_indices_]
        poly_names = self.poly_transformer_.get_feature_names_out(selected_names)
        # Skip original features (already in names)
        for pn in poly_names[len(self.selected_indices_):]:
            names.append(f"poly_{pn}")
        return names


def build_feature_engineer(task_type: str) -> Pipeline:
    """Build a feature engineering pipeline.

    Args:
        task_type: 'classification' or 'regression'.

    Returns:
        sklearn Pipeline containing feature engineering steps.
    """
    return Pipeline([
        ("log_transform", LogTransformer(skew_threshold=1.0)),
        ("poly_features", PolynomialFeatureGenerator(top_k=10, degree=2)),
    ])
