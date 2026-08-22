"""Family-aware, leakage-safe preprocessing.

Every learned transformation lives inside sklearn transformers so
cross-validation can refit them on each training fold.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from automl.config import EncoderType, ExperimentConfig, ScalerType
from automl.quality import OutlierClipper, OutlierHandler, outlier_handler_from_config
from automl.utils import get_column_types

logger = logging.getLogger(__name__)

CARDINALITY_THRESHOLD = 15


def _scaler_from_name(name: str, n_samples: int) -> Any:
    if name == ScalerType.NONE.value:
        return "passthrough"
    if name == ScalerType.ROBUST.value:
        return RobustScaler()
    if name == ScalerType.MINMAX.value:
        return MinMaxScaler()
    if name == ScalerType.POWER.value:
        return PowerTransformer(method="yeo-johnson")
    if name == ScalerType.QUANTILE.value:
        return QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, max(10, n_samples)),
        )
    return StandardScaler()


def recommend_scaler(family: str, n_samples: int, has_outliers: bool) -> str:
    if family in {"tree", "boosting"}:
        return ScalerType.NONE.value
    if family == "neural":
        return ScalerType.STANDARD.value
    if has_outliers and n_samples >= 50:
        return ScalerType.ROBUST.value
    if family == "svm":
        return ScalerType.STANDARD.value
    return ScalerType.STANDARD.value


def _numeric_imputer(strategy: str, n_samples: int, n_features: int) -> Any:
    if strategy == "knn" and n_samples >= 30:
        return KNNImputer(n_neighbors=min(5, max(2, n_samples // 10)))
    if strategy == "iterative" and n_samples >= 40 and n_features <= 40:
        return IterativeImputer(max_iter=8, random_state=42)
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=0)
    return SimpleImputer(strategy="median")


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    family: str = "linear",
    config: ExperimentConfig | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a ColumnTransformer tailored to the input data and model family.

    Args:
        df: Feature DataFrame (target column already removed).
        target_col: Name of the target column (for logging context).
        task_type: 'classification' or 'regression'.
        family: Model family used to choose scaling/encoding defaults.
        config: Optional experiment configuration.
    """
    config = config or ExperimentConfig()
    numeric_cols, categorical_cols = get_column_types(df)
    n_samples = len(df)
    card_threshold = config.cardinality_threshold

    low_card_cats = [c for c in categorical_cols if df[c].nunique(dropna=True) <= card_threshold]
    high_card_cats = [c for c in categorical_cols if df[c].nunique(dropna=True) > card_threshold]

    has_outliers = False
    if numeric_cols:
        sample = df[numeric_cols].select_dtypes(include=[np.number])
        if not sample.empty:
            q1, q3 = sample.quantile(0.25), sample.quantile(0.75)
            iqr = (q3 - q1).replace(0, np.nan)
            has_outliers = bool(((sample < q1 - 1.5 * iqr) | (sample > q3 + 1.5 * iqr)).any().any())

    scaler_name = config.scaler.value
    if scaler_name == ScalerType.AUTO.value:
        scaler_name = recommend_scaler(family, n_samples, has_outliers)

    encoder_name = config.encoder.value
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        num_steps: list[tuple[str, Any]] = [
            ("imputer", _numeric_imputer(config.numeric_imputer, n_samples, len(numeric_cols)))
        ]
        if config.add_missing_indicator and df[numeric_cols].isna().any().any():
            # MissingIndicator is composed separately to keep shapes stable
            pass
        scaler = _scaler_from_name(scaler_name, n_samples)
        if scaler != "passthrough":
            num_steps.append(("scaler", scaler))
        transformers.append(("numeric", Pipeline(num_steps), numeric_cols))
        if config.add_missing_indicator and df[numeric_cols].isna().any().any():
            transformers.append(
                (
                    "num_missing",
                    MissingIndicator(features="missing-only"),
                    numeric_cols,
                )
            )

    def _cat_encoder(high_card: bool) -> Any:
        if encoder_name == EncoderType.ORDINAL.value:
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if encoder_name == EncoderType.TARGET.value or (
            encoder_name == EncoderType.AUTO.value and high_card
        ):
            return TargetEncoder(smooth="auto", target_type="auto")
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=40)

    if low_card_cats:
        cat_imputer = (
            SimpleImputer(strategy="constant", fill_value="Missing")
            if config.categorical_imputer == "constant"
            else SimpleImputer(strategy="most_frequent")
        )
        transformers.append(
            (
                "cat_low",
                Pipeline(
                    [
                        ("imputer", cat_imputer),
                        ("encoder", _cat_encoder(high_card=False)),
                    ]
                ),
                low_card_cats,
            )
        )

    if high_card_cats:
        cat_imputer = (
            SimpleImputer(strategy="constant", fill_value="Missing")
            if config.categorical_imputer == "constant"
            else SimpleImputer(strategy="most_frequent")
        )
        transformers.append(
            (
                "cat_high",
                Pipeline(
                    [
                        ("imputer", cat_imputer),
                        ("encoder", _cat_encoder(high_card=True)),
                    ]
                ),
                high_card_cats,
            )
        )

    logger.info(
        "Preprocessor family=%s scaler=%s numeric=%d low_cat=%d high_cat=%d",
        family,
        scaler_name,
        len(numeric_cols),
        len(low_card_cats),
        len(high_card_cats),
    )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor, numeric_cols, categorical_cols


class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Ensure pipeline steps receive a DataFrame with stable column names."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> "DataFrameConverter":
        if isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
        elif self.columns is not None:
            self.columns_ = list(self.columns)
        else:
            n = np.asarray(X).shape[1]
            self.columns_ = [f"f{i}" for i in range(n)]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X), columns=self.columns_)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)
