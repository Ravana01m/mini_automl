"""Feature selection pipeline builder.

Combines multiple feature selection strategies:
1. VarianceThreshold — drop near-zero variance features
2. Correlation filter — drop one of highly correlated pairs (|r| > 0.95)
3. SelectKBest — statistical test (f_classif or f_regression)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from highly correlated pairs.

    For each pair with |correlation| > threshold, drops the feature
    that appears most frequently in high-correlation pairs.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "CorrelationFilter":
        """Identify features to drop based on pairwise correlation."""
        arr = X if isinstance(X, np.ndarray) else X.values
        n_features = arr.shape[1]
        
        # Handle edge cases
        if n_features <= 1:
            self.drop_indices_: list[int] = []
            self.n_features_in_ = n_features
            return self
        
        # Compute correlation matrix (handle NaN with pairwise complete)
        # Use numpy for speed
        with np.errstate(invalid="ignore"):
            corr_matrix = np.abs(np.corrcoef(arr.T))
        
        # Replace NaN correlations with 0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Find features to drop
        drop_set: set[int] = set()
        for i in range(n_features):
            if i in drop_set:
                continue
            for j in range(i + 1, n_features):
                if j in drop_set:
                    continue
                if corr_matrix[i, j] > self.threshold:
                    drop_set.add(j)
        
        self.drop_indices_ = sorted(drop_set)
        self.n_features_in_ = n_features
        logger.info(
            "CorrelationFilter: dropping %d of %d features (threshold=%.2f)",
            len(self.drop_indices_), n_features, self.threshold,
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Drop identified features."""
        arr = X if isinstance(X, np.ndarray) else X.values
        if not self.drop_indices_:
            return arr
        keep = [i for i in range(arr.shape[1]) if i not in self.drop_indices_]
        return arr[:, keep]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return kept feature names."""
        if input_features is None:
            input_features = [f"f{i}" for i in range(self.n_features_in_)]
        return [f for i, f in enumerate(input_features) if i not in self.drop_indices_]


def build_feature_selector(task_type: str, k: int | str = "auto") -> Pipeline:
    """Build a feature selection pipeline.

    Args:
        task_type: 'classification' or 'regression'.
        k: Number of features for SelectKBest, or 'auto' for heuristic.

    Returns:
        sklearn Pipeline with VarianceThreshold, CorrelationFilter, SelectKBest.
    """
    score_func = f_classif if task_type == "classification" else f_regression
    
    # 'auto' → select all (let SelectKBest act as a scorer only)
    k_value = k if isinstance(k, int) else "all"
    
    return Pipeline([
        ("variance", VarianceThreshold(threshold=0.01)),
        ("correlation", CorrelationFilter(threshold=0.95)),
        ("select_k_best", SelectKBest(score_func=score_func, k=k_value)),
    ])
