"""Pipeline orchestrator: assembles the full AutoML pipeline.

Combines preprocessing, feature engineering, feature selection, and the
best model into a single sklearn Pipeline that works end-to-end on raw data.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from automl.detector import detect_task_type
from automl.eda import generate_eda_report
from automl.preprocessing import OutlierClipper, build_preprocessor
from automl.feature_engineering import build_feature_engineer
from automl.feature_selection import build_feature_selector
from automl.model_registry import get_models
from automl.trainer import Trainer
from automl.explainer import ModelExplainer
from automl.utils import validate_csv, validate_target_column

logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """Main orchestrator for the AutoML pipeline."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.task_type_: str | None = None
        self.pipeline_: Pipeline | None = None
        self.preprocessing_pipeline_: Pipeline | None = None
        self.trainer_: Trainer | None = None
        self.explainer_: ModelExplainer | None = None
        self.leaderboard_: pd.DataFrame | None = None
        self.eda_report_: dict[str, Any] | None = None
        self.target_col_: str | None = None
        self.n_classes_: int | None = None

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type_override: str | None = None,
        progress_callback: Any | None = None,
        skip_optuna: bool = False,
    ) -> dict[str, Any]:
        """Execute the full AutoML pipeline.

        Args:
            df: Raw uploaded DataFrame.
            target_col: Name of the target column.
            task_type_override: Optional manual override for task type.
            progress_callback: Optional callable(stage, detail) for UI updates.
            skip_optuna: Skip Optuna tuning for faster results.

        Returns:
            Dict with keys: 'task_type', 'leaderboard', 'best_model_name',
            'pipeline', 'explainer', 'metrics', 'eda_report'.
        """
        self.target_col_ = target_col
        
        # Step 1: Validation
        if progress_callback:
            progress_callback("validation", "Validating input data...")
        validate_csv(df)
        validate_target_column(df, target_col)
        
        # Step 2: Task detection
        if progress_callback:
            progress_callback("detection", "Detecting task type...")
        if task_type_override:
            self.task_type_ = task_type_override
        else:
            self.task_type_ = detect_task_type(df[target_col])
        logger.info("Task type: %s", self.task_type_)
        
        # Step 3: EDA
        if progress_callback:
            progress_callback("eda", "Generating EDA report...")
        try:
            self.eda_report_ = generate_eda_report(df, target_col, self.task_type_)
        except Exception as e:
            logger.warning("EDA generation failed: %s", str(e))
            self.eda_report_ = None
        
        # Step 4: Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()
        
        # Encode target for classification if needed
        if self.task_type_ == "classification" and y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target_col)
            self.n_classes_ = len(le.classes_)
        elif self.task_type_ == "classification":
            self.n_classes_ = int(y.nunique())
        
        # Step 5: Build preprocessing
        if progress_callback:
            progress_callback("preprocessing", "Building preprocessing pipeline...")
        
        # Outlier clipping (before ColumnTransformer)
        outlier_clipper = OutlierClipper(factor=1.5)
        
        # ColumnTransformer
        preprocessor, numeric_cols, categorical_cols = build_preprocessor(
            X, target_col, self.task_type_
        )
        
        # Feature engineering
        feature_engineer = build_feature_engineer(self.task_type_)
        
        # Feature selection
        feature_selector = build_feature_selector(self.task_type_)
        
        # Assemble preprocessing pipeline
        self.preprocessing_pipeline_ = Pipeline([
            ("outlier_clipper", outlier_clipper),
            ("preprocessor", preprocessor),
            ("feature_engineer", feature_engineer),
            ("feature_selector", feature_selector),
        ])
        
        # Fit preprocessing and transform
        if progress_callback:
            progress_callback("preprocessing", "Fitting preprocessing pipeline...")
        
        X_processed = self.preprocessing_pipeline_.fit_transform(X, y)
        logger.info(
            "Preprocessing complete: %s -> %s",
            X.shape, X_processed.shape,
        )
        
        # Step 6: Get models
        if progress_callback:
            progress_callback("training", "Loading model candidates...")
        
        input_dim = X_processed.shape[1]
        models = get_models(
            task_type=self.task_type_,
            input_dim=input_dim,
            n_classes=self.n_classes_,
        )
        
        # Step 7: Train and evaluate
        if progress_callback:
            progress_callback("training", "Training models with cross-validation...")
        
        self.trainer_ = Trainer(
            task_type=self.task_type_,
            cv_folds=5,
            random_state=self.random_state,
        )
        
        self.leaderboard_ = self.trainer_.train_and_evaluate(
            X_processed, y, models, progress_callback=None,
        )
        
        # Step 8: Grid Search
        if progress_callback:
            progress_callback("tuning", "Running GridSearchCV...")
        
        self.leaderboard_ = self.trainer_.grid_search(X_processed, y, models)
        
        # Step 9: Optuna (optional)
        if not skip_optuna:
            if progress_callback:
                progress_callback("tuning", "Running Optuna tuning on top 2 models...")
            self.leaderboard_ = self.trainer_.optuna_tune(
                X_processed, y, top_n=2, n_trials=50,
            )
        
        # Step 10: Build final pipeline
        if progress_callback:
            progress_callback("finalizing", "Building final pipeline...")
        
        self.pipeline_ = self.trainer_.get_best_pipeline(self.preprocessing_pipeline_)
        
        # Step 11: Explainability
        if progress_callback:
            progress_callback("explainability", "Computing SHAP explanations...")
        
        try:
            self.explainer_ = ModelExplainer(self.pipeline_, self.task_type_)
            self.explainer_.compute_shap_values(X)
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", str(e))
            self.explainer_ = None
        
        # Build result dict
        return {
            "task_type": self.task_type_,
            "leaderboard": self.leaderboard_,
            "best_model_name": self.trainer_.best_model_name_,
            "pipeline": self.pipeline_,
            "explainer": self.explainer_,
            "eda_report": self.eda_report_,
            "cv_results": self.trainer_.cv_results_,
            "fitted_models": self.trainer_.fitted_models_,
        }

    def export_pipeline(self, filepath: str) -> str:
        """Export the full fitted pipeline to a joblib file."""
        if self.pipeline_ is None:
            raise ValueError("No pipeline to export. Run the pipeline first.")
        joblib.dump(self.pipeline_, filepath)
        logger.info("Pipeline exported to %s", filepath)
        return filepath

    @staticmethod
    def load_pipeline(filepath: str) -> Pipeline:
        """Load a previously exported pipeline."""
        return joblib.load(filepath)
