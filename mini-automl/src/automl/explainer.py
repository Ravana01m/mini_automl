"""Model explainability using SHAP.

Provides global feature importance (summary plot) and per-prediction
explanations (waterfall/force plot) for the best model.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelExplainer:
    """SHAP-based model explainer."""

    def __init__(
        self,
        pipeline: Pipeline,
        task_type: str,
    ) -> None:
        self.pipeline = pipeline
        self.task_type = task_type
        self.shap_values_: np.ndarray | None = None
        self.explainer_: Any | None = None
        self.feature_names_: list[str] | None = None
        self.X_transformed_: np.ndarray | None = None

    def _get_model(self) -> Any:
        """Extract the final model from the pipeline."""
        if hasattr(self.pipeline, "named_steps"):
            # Get the last step
            steps = list(self.pipeline.named_steps.values())
            return steps[-1]
        return self.pipeline

    def _get_preprocessor(self) -> Any:
        """Extract the preprocessing steps from the pipeline."""
        if hasattr(self.pipeline, "named_steps"):
            steps = list(self.pipeline.named_steps.values())
            if len(steps) > 1:
                return Pipeline(list(self.pipeline.named_steps.items())[:-1])
        return None

    def compute_shap_values(
        self,
        X: pd.DataFrame | np.ndarray,
        max_samples: int = 100,
    ) -> None:
        """Compute SHAP values for the given data."""
        model = self._get_model()
        preprocessor = self._get_preprocessor()
        
        # Transform data through preprocessing
        if preprocessor is not None:
            X_transformed = preprocessor.transform(X)
        else:
            X_transformed = X.values if isinstance(X, pd.DataFrame) else X
        
        if isinstance(X_transformed, pd.DataFrame):
            X_transformed = X_transformed.values
        
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        
        # Subsample for speed
        n_samples = min(max_samples, X_transformed.shape[0])
        indices = np.random.RandomState(42).choice(
            X_transformed.shape[0], n_samples, replace=False
        )
        X_sample = X_transformed[indices]
        
        self.X_transformed_ = X_sample
        
        # Get feature names
        try:
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                self.feature_names_ = list(preprocessor.get_feature_names_out())
            else:
                self.feature_names_ = [f"Feature {i}" for i in range(X_sample.shape[1])]
        except Exception:
            self.feature_names_ = [f"Feature {i}" for i in range(X_sample.shape[1])]
        
        # Select appropriate SHAP explainer
        model_type = type(model).__name__
        
        try:
            if model_type in (
                "RandomForestClassifier", "RandomForestRegressor",
                "XGBClassifier", "XGBRegressor",
                "LGBMClassifier", "LGBMRegressor",
            ):
                self.explainer_ = shap.TreeExplainer(model)
                self.shap_values_ = self.explainer_.shap_values(X_sample)
            elif model_type in ("LogisticRegression", "Ridge"):
                self.explainer_ = shap.LinearExplainer(model, X_sample)
                self.shap_values_ = self.explainer_.shap_values(X_sample)
            else:
                # Fallback: KernelExplainer (works for ANN, SVM, etc.)
                background = shap.kmeans(X_sample, min(10, n_samples))
                predict_fn = (
                    model.predict_proba if hasattr(model, "predict_proba") and self.task_type == "classification"
                    else model.predict
                )
                self.explainer_ = shap.KernelExplainer(predict_fn, background)
                self.shap_values_ = self.explainer_.shap_values(
                    X_sample[:min(50, n_samples)]
                )
                self.X_transformed_ = X_sample[:min(50, n_samples)]
        except Exception as e:
            logger.warning("SHAP computation failed: %s. Using fallback.", str(e))
            # Fallback: use KernelExplainer
            try:
                background = shap.kmeans(X_sample, min(10, n_samples))
                self.explainer_ = shap.KernelExplainer(model.predict, background)
                self.shap_values_ = self.explainer_.shap_values(
                    X_sample[:min(30, n_samples)]
                )
                self.X_transformed_ = X_sample[:min(30, n_samples)]
            except Exception as e2:
                logger.error("All SHAP methods failed: %s", str(e2))
                raise
        
        # For multi-class, take the values for the positive/first class
        if isinstance(self.shap_values_, list) and len(self.shap_values_) > 0:
            self.shap_values_ = self.shap_values_[1] if len(self.shap_values_) == 2 else self.shap_values_[0]
        
        logger.info(
            "SHAP values computed: shape=%s, features=%d",
            self.shap_values_.shape, len(self.feature_names_),
        )

    def summary_plot(self) -> matplotlib.figure.Figure:
        """Generate SHAP summary plot (global feature importance)."""
        if self.shap_values_ is None:
            raise ValueError("Call compute_shap_values() first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
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

    def waterfall_plot(self, idx: int = 0) -> matplotlib.figure.Figure:
        """Generate SHAP waterfall plot for a single prediction."""
        if self.shap_values_ is None:
            raise ValueError("Call compute_shap_values() first.")
        
        if idx >= self.shap_values_.shape[0]:
            idx = 0
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create an Explanation object
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
