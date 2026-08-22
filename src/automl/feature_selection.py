"""Leakage-safe feature selection used inside cross-validation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.pipeline import Pipeline

from automl.config import ExperimentConfig, FeatureSelectionStrategy
from automl.types import FeatureSelectionReport

logger = logging.getLogger(__name__)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from highly correlated pairs. Fit on training folds only."""

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "CorrelationFilter":
        arr = np.asarray(X, dtype=float)
        n_features = arr.shape[1]
        self.n_features_in_ = n_features
        if n_features <= 1:
            self.drop_indices_: list[int] = []
            return self
        with np.errstate(invalid="ignore"):
            corr_matrix = np.abs(np.corrcoef(arr.T))
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
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
        logger.info(
            "CorrelationFilter: dropping %d of %d features (threshold=%.2f)",
            len(self.drop_indices_),
            n_features,
            self.threshold,
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X)
        if not self.drop_indices_:
            return arr
        keep = [i for i in range(arr.shape[1]) if i not in self.drop_indices_]
        return arr[:, keep]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is None:
            input_features = [f"f{i}" for i in range(self.n_features_in_)]
        return np.asarray(
            [f for i, f in enumerate(input_features) if i not in self.drop_indices_],
            dtype=object,
        )


class FeatureSelectionEngine(BaseEstimator, TransformerMixin):
    """Intelligent multi-strategy selector. Never blindly keeps a fixed K."""

    def __init__(
        self,
        task_type: str = "classification",
        strategy: str = "automatic",
        correlation_threshold: float = 0.95,
        family: str = "linear",
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        self.strategy = strategy
        self.correlation_threshold = correlation_threshold
        self.family = family
        self.random_state = random_state

    def _plan(self, n_features: int, n_samples: int) -> list[tuple[str, Any]]:
        strategy = self.strategy
        if strategy == FeatureSelectionStrategy.NONE.value:
            return []
        if strategy == FeatureSelectionStrategy.AUTOMATIC.value:
            if n_features < 30:
                strategy = FeatureSelectionStrategy.LIGHT.value
            elif n_features < 100:
                return [
                    ("variance", VarianceThreshold(threshold=1e-4)),
                    ("correlation", CorrelationFilter(self.correlation_threshold)),
                    ("mi", self._kbest("mi", k=min(n_features, max(12, n_features // 2)))),
                ]
            elif n_features < 500:
                return [
                    ("variance", VarianceThreshold(threshold=1e-3)),
                    ("correlation", CorrelationFilter(min(self.correlation_threshold, 0.95))),
                    ("kbest", self._kbest("f", k=min(80, max(20, n_samples // 5)))),
                ]
            else:
                return [
                    ("variance", VarianceThreshold(threshold=1e-3)),
                    ("correlation", CorrelationFilter(0.90)),
                    ("model", self._model_selector()),
                ]
        if strategy == FeatureSelectionStrategy.LIGHT.value:
            return [
                ("variance", VarianceThreshold(threshold=1e-4)),
                ("correlation", CorrelationFilter(self.correlation_threshold)),
            ]
        # aggressive
        k = min(n_features, max(8, min(40, n_samples // 8)))
        steps = [
            ("variance", VarianceThreshold(threshold=1e-3)),
            ("correlation", CorrelationFilter(min(self.correlation_threshold, 0.90))),
            ("kbest", self._kbest("mi", k=k)),
        ]
        if n_samples >= 80 and n_features <= 80:
            steps.append(("rfe", self._rfe(min(k, 20))))
        return steps

    def _kbest(self, kind: str, k: int | str) -> SelectKBest:
        if self.task_type == "classification":
            score = mutual_info_classif if kind == "mi" else f_classif
        else:
            score = mutual_info_regression if kind == "mi" else f_regression
        return SelectKBest(score_func=score, k=k)

    def _model_selector(self) -> SelectFromModel:
        if self.task_type == "classification":
            if self.family in {"linear", "svm", "neural"}:
                estimator = LogisticRegression(
                    penalty="l1", solver="saga", max_iter=400, random_state=self.random_state
                )
            else:
                estimator = ExtraTreesClassifier(
                    n_estimators=80, random_state=self.random_state, n_jobs=1
                )
        else:
            if self.family in {"linear", "svm", "neural"}:
                estimator = LassoCV(max_iter=400, random_state=self.random_state)
            else:
                estimator = ExtraTreesRegressor(
                    n_estimators=80, random_state=self.random_state, n_jobs=1
                )
        return SelectFromModel(estimator, max_features=80)

    def _rfe(self, n_features: int) -> RFE:
        if self.task_type == "classification":
            est = ExtraTreesClassifier(n_estimators=40, random_state=self.random_state, n_jobs=1)
        else:
            est = ExtraTreesRegressor(n_estimators=40, random_state=self.random_state, n_jobs=1)
        return RFE(est, n_features_to_select=n_features, step=0.25)

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "FeatureSelectionEngine":
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(arr.shape[1])]
        )
        steps = self._plan(arr.shape[1], arr.shape[0])
        if not steps:
            self.pipeline_ = Pipeline([("noop", "passthrough")])
        else:
            self.pipeline_ = Pipeline(steps)
        y_arr = None if y is None else np.asarray(y)
        try:
            self.pipeline_.fit(arr, y_arr)
        except Exception as exc:
            logger.warning("Feature selection failed (%s); passing all features through", exc)
            self.pipeline_ = Pipeline([("noop", "passthrough")])
            self.pipeline_.fit(arr, y_arr)
        self.selected_mask_ = np.ones(arr.shape[1], dtype=bool)
        try:
            Xt = self.pipeline_.transform(arr)
            names = list(self.get_feature_names_out(self.feature_names_in_))
            self.selected_names_ = names
            self.removed_names_ = [n for n in self.feature_names_in_ if n not in names]
        except Exception:
            self.selected_names_ = list(self.feature_names_in_)
            self.removed_names_ = []
        self.report_ = FeatureSelectionReport(
            original_features=list(self.feature_names_in_),
            engineered_features=[],
            removed_features=self.removed_names_,
            selected_features=self.selected_names_,
            method=self.strategy,
        )
        logger.info(
            "FeatureSelectionEngine[%s]: %d -> %d features",
            self.strategy,
            arr.shape[1],
            len(self.selected_names_),
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline_.transform(np.asarray(X, dtype=float)))

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = list(input_features) if input_features is not None else list(self.feature_names_in_)
        try:
            return np.asarray(self.pipeline_.get_feature_names_out(names), dtype=object)
        except Exception:
            return np.asarray(names, dtype=object)


def build_feature_selector(
    task_type: str,
    k: int | str = "auto",
    config: ExperimentConfig | None = None,
    family: str = "linear",
) -> Pipeline:
    """Build a feature selection pipeline. Backward compatible with k='auto'."""
    score_func = f_classif if task_type == "classification" else f_regression
    k_value: int | str = "all" if k in {"auto", "all"} else k
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.01)),
            ("correlation", CorrelationFilter(threshold=0.95)),
            ("select_k_best", SelectKBest(score_func=score_func, k=k_value)),
        ]
    )
