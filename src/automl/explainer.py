"""SHAP explainability. Failures never crash the AutoML pipeline."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ModelExplainer:
    """SHAP-based model explainer with tree / linear / kernel fallbacks."""

    def __init__(self, pipeline: Pipeline | Any, task_type: str) -> None:
        self.pipeline = pipeline
        self.task_type = task_type
        self.shap_values_: np.ndarray | None = None
        self.explainer_: Any | None = None
        self.feature_names_: list[str] | None = None
        self.X_transformed_: np.ndarray | None = None

    def _get_model(self) -> Any:
        if hasattr(self.pipeline, "named_steps") and "model" in self.pipeline.named_steps:
            return self.pipeline.named_steps["model"]
        if hasattr(self.pipeline, "named_steps"):
            return list(self.pipeline.named_steps.values())[-1]
        return self.pipeline

    def _get_preprocessor(self) -> Any:
        if hasattr(self.pipeline, "named_steps") and "model" in self.pipeline.named_steps:
            steps = [(k, v) for k, v in self.pipeline.named_steps.items() if k != "model"]
            return Pipeline(steps) if steps else None
        if hasattr(self.pipeline, "named_steps"):
            steps = list(self.pipeline.named_steps.items())[:-1]
            return Pipeline(steps) if steps else None
        return None

    def compute_shap_values(
        self,
        X: pd.DataFrame | np.ndarray,
        max_samples: int = 80,
    ) -> None:
        if shap is None:
            logger.warning("SHAP is not installed; skipping explanations")
            return
        try:
            self._compute(X, max_samples)
        except Exception as exc:
            logger.warning("SHAP computation failed and was skipped: %s", exc)
            self.shap_values_ = None

    def _compute(self, X: pd.DataFrame | np.ndarray, max_samples: int) -> None:
        model = self._get_model()
        preprocessor = self._get_preprocessor()
        if preprocessor is not None:
            X_transformed = preprocessor.transform(X)
        else:
            X_transformed = X.values if isinstance(X, pd.DataFrame) else X
        if isinstance(X_transformed, pd.DataFrame):
            X_transformed = X_transformed.values
        X_transformed = np.asarray(X_transformed, dtype=float)
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)

        n_samples = min(max_samples, X_transformed.shape[0])
        indices = np.random.RandomState(42).choice(X_transformed.shape[0], n_samples, replace=False)
        X_sample = X_transformed[indices]
        self.X_transformed_ = X_sample

        try:
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                self.feature_names_ = [str(n) for n in preprocessor.get_feature_names_out()]
            else:
                self.feature_names_ = [f"Feature {i}" for i in range(X_sample.shape[1])]
        except Exception:
            self.feature_names_ = [f"Feature {i}" for i in range(X_sample.shape[1])]

        model_type = type(model).__name__
        tree_names = {
            "RandomForestClassifier",
            "RandomForestRegressor",
            "ExtraTreesClassifier",
            "ExtraTreesRegressor",
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "GradientBoostingClassifier",
            "GradientBoostingRegressor",
            "HistGradientBoostingClassifier",
            "HistGradientBoostingRegressor",
        }
        try:
            if model_type in tree_names:
                self.explainer_ = shap.TreeExplainer(model)
                self.shap_values_ = self.explainer_.shap_values(X_sample)
            elif model_type in {"LogisticRegression", "Ridge", "Lasso", "ElasticNet", "LinearRegression"}:
                self.explainer_ = shap.LinearExplainer(model, X_sample)
                self.shap_values_ = self.explainer_.shap_values(X_sample)
            else:
                background = shap.kmeans(X_sample, min(8, n_samples))
                predict_fn = (
                    model.predict_proba
                    if hasattr(model, "predict_proba") and self.task_type == "classification"
                    else model.predict
                )
                self.explainer_ = shap.KernelExplainer(predict_fn, background)
                self.shap_values_ = self.explainer_.shap_values(X_sample[: min(30, n_samples)])
                self.X_transformed_ = X_sample[: min(30, n_samples)]
        except Exception as exc:
            logger.warning("Primary SHAP explainer failed (%s); using KernelExplainer", exc)
            background = shap.kmeans(X_sample, min(8, n_samples))
            self.explainer_ = shap.KernelExplainer(model.predict, background)
            self.shap_values_ = self.explainer_.shap_values(X_sample[: min(20, n_samples)])
            self.X_transformed_ = X_sample[: min(20, n_samples)]

        if isinstance(self.shap_values_, list) and self.shap_values_:
            self.shap_values_ = (
                self.shap_values_[1] if len(self.shap_values_) == 2 else self.shap_values_[0]
            )
        logger.info(
            "SHAP values computed: shape=%s, features=%d",
            getattr(self.shap_values_, "shape", None),
            len(self.feature_names_ or []),
        )

    def summary_plot(self) -> matplotlib.figure.Figure:
        if self.shap_values_ is None:
            raise ValueError("Call compute_shap_values() first.")
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values_,
            self.X_transformed_,
            feature_names=self.feature_names_,
            show=False,
            max_display=20,
        )
        fig = plt.gcf()
        fig.tight_layout()
        return fig

    def bar_plot(self) -> matplotlib.figure.Figure:
        if self.shap_values_ is None:
            raise ValueError("Call compute_shap_values() first.")
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values_,
            self.X_transformed_,
            feature_names=self.feature_names_,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        fig = plt.gcf()
        fig.tight_layout()
        return fig

    def waterfall_plot(self, idx: int = 0) -> matplotlib.figure.Figure:
        if self.shap_values_ is None:
            raise ValueError("Call compute_shap_values() first.")
        if idx >= self.shap_values_.shape[0]:
            idx = 0
        fig, _ = plt.subplots(figsize=(10, 6))
        expected_value = (
            self.explainer_.expected_value
            if not isinstance(self.explainer_.expected_value, (list, np.ndarray))
            else self.explainer_.expected_value[0]
        )
        explanation = shap.Explanation(
            values=self.shap_values_[idx],
            base_values=expected_value,
            data=self.X_transformed_[idx],
            feature_names=self.feature_names_,
        )
        shap.waterfall_plot(explanation, show=False, max_display=15)
        fig = plt.gcf()
        fig.tight_layout()
        return fig

    def local_contributors(self, idx: int = 0, top_n: int = 8) -> pd.DataFrame:
        if self.shap_values_ is None:
            return pd.DataFrame()
        values = self.shap_values_[idx]
        names = self.feature_names_ or [f"f{i}" for i in range(len(values))]
        frame = pd.DataFrame({"feature": names, "shap": values})
        frame["direction"] = np.where(frame["shap"] >= 0, "positive", "negative")
        return frame.reindex(frame["shap"].abs().sort_values(ascending=False).index).head(top_n)
