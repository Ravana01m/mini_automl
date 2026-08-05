"""Dynamic preprocessing pipeline builder.

Builds an sklearn ColumnTransformer that automatically handles:
- Missing value imputation (median for numeric, mode for categorical)
- Outlier clipping (IQR-based custom transformer)
- Categorical encoding (one-hot for low cardinality, target encoding for high)
- Feature scaling (StandardScaler for numeric)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from automl.utils import get_column_types

logger = logging.getLogger(__name__)

CARDINALITY_THRESHOLD = 15


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers using the IQR method.

    For each numeric column, values below Q1 - factor*IQR are clipped to that
    bound, and values above Q3 + factor*IQR are clipped to that bound.
    """

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "OutlierClipper":
        """Compute IQR bounds for each numeric column."""
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
        """Clip values to fitted IQR bounds."""
        is_df = isinstance(X, pd.DataFrame)
        if not is_df:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            df = X.copy()
        
        for col in self.feature_names_in_:
            if col in df.columns:
                df[col] = df[col].clip(
                    lower=self.lower_bounds_.get(col),
                    upper=self.upper_bounds_.get(col),
                )
        
        return df if is_df else df.values

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return feature names."""
        return input_features if input_features else self.feature_names_in_


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a ColumnTransformer tailored to the input data.

    Args:
        df: Feature DataFrame (target column already removed).
        target_col: Name of the target column (for logging context).
        task_type: 'classification' or 'regression'.

    Returns:
        Tuple of (ColumnTransformer, numeric_columns, categorical_columns).
    """
    numeric_cols, categorical_cols = get_column_types(df)
    
    # Split categorical by cardinality
    low_card_cats = [
        col for col in categorical_cols
        if df[col].nunique() <= CARDINALITY_THRESHOLD
    ]
    high_card_cats = [
        col for col in categorical_cols
        if df[col].nunique() > CARDINALITY_THRESHOLD
    ]
    
    logger.info(
        "Columns detected — numeric: %d, low-cardinality cat: %d, "
        "high-cardinality cat: %d",
        len(numeric_cols), len(low_card_cats), len(high_card_cats),
    )
    
    transformers: list[tuple[str, Pipeline | str, list[str]]] = []
    
    # Numeric pipeline: impute → scale
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    
    # Low-cardinality categorical: impute → one-hot
    if low_card_cats:
        low_card_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                max_categories=50,
            )),
        ])
        transformers.append(("cat_low", low_card_pipeline, low_card_cats))
    
    # High-cardinality categorical: impute → target encode
    if high_card_cats:
        high_card_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(
                smooth="auto",
                target_type="auto",
            )),
        ])
        transformers.append(("cat_high", high_card_pipeline, high_card_cats))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    
    return preprocessor, numeric_cols, categorical_cols
