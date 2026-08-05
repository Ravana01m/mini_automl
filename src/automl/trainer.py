"""Model training, evaluation, and hyperparameter tuning.

Provides cross-validated training for all candidate models,
GridSearchCV for fast sweeps, and Optuna for deep tuning of top models.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline

from automl.model_registry import get_optuna_search_space

logger = logging.getLogger(__name__)

# Suppress excessive warnings during grid search
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Trainer:
    """Orchestrates model training, comparison, and tuning."""

    def __init__(
        self,
        task_type: str,
        scoring_metric: str | None = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Auto-select scoring metric
        if scoring_metric is None:
            self.scoring_metric = "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"
        else:
            self.scoring_metric = scoring_metric
        
        self.results_: list[dict[str, Any]] = []
        self.cv_results_: dict[str, list[float]] = {}
        self.best_model_: BaseEstimator | None = None
        self.best_model_name_: str | None = None
        self.fitted_models_: dict[str, BaseEstimator] = {}

    def _get_cv(self) -> StratifiedKFold | KFold:
        """Get cross-validation splitter based on task type."""
        if self.task_type == "classification":
            return StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        return KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        models: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]],
        progress_callback: Any | None = None,
    ) -> pd.DataFrame:
        """Train all candidate models with cross-validation."""
        self.results_ = []
        self.cv_results_ = {}
        cv = self._get_cv()
        
        # Convert to numpy for consistency
        X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        # Handle NaN values in y
        X = np.nan_to_num(X, nan=0.0)
        
        total = len(models)
        for idx, (name, (estimator, _)) in enumerate(models.items()):
            logger.info("Training %s (%d/%d)", name, idx + 1, total)
            if progress_callback:
                progress_callback(name, idx + 1, total)
            
            try:
                start_time = time.time()
                model = clone(estimator)
                
                # Define scoring based on task type
                if self.task_type == "classification":
                    scoring = {
                        "accuracy": "accuracy",
                        "f1_weighted": "f1_weighted",
                    }
                else:
                    scoring = {
                        "r2": "r2",
                        "neg_rmse": "neg_root_mean_squared_error",
                    }
                
                cv_result = cross_validate(
                    model, X, y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=1,  # Avoid nested parallelism issues
                    error_score="raise",
                )
                
                train_time = time.time() - start_time
                
                # Build result row
                result: dict[str, Any] = {"model": name, "train_time_s": round(train_time, 2)}
                
                if self.task_type == "classification":
                    acc_scores = cv_result["test_accuracy"]
                    f1_scores = cv_result["test_f1_weighted"]
                    result["accuracy_mean"] = round(float(np.mean(acc_scores)), 4)
                    result["accuracy_std"] = round(float(np.std(acc_scores)), 4)
                    result["f1_weighted_mean"] = round(float(np.mean(f1_scores)), 4)
                    result["f1_weighted_std"] = round(float(np.std(f1_scores)), 4)
                    self.cv_results_[name] = f1_scores.tolist()
                else:
                    r2_scores = cv_result["test_r2"]
                    rmse_scores = -cv_result["test_neg_rmse"]
                    result["r2_mean"] = round(float(np.mean(r2_scores)), 4)
                    result["r2_std"] = round(float(np.std(r2_scores)), 4)
                    result["rmse_mean"] = round(float(np.mean(rmse_scores)), 4)
                    result["rmse_std"] = round(float(np.std(rmse_scores)), 4)
                    self.cv_results_[name] = r2_scores.tolist()
                
                result["status"] = "success"
                self.results_.append(result)
                
                # Fit model on full training data for later use
                model.fit(X, y)
                self.fitted_models_[name] = model
                
                logger.info("%s completed in %.1fs", name, train_time)
                
            except Exception as e:
                logger.warning("Failed to train %s: %s", name, str(e))
                result = {
                    "model": name,
                    "status": "failed",
                    "error": str(e),
                    "train_time_s": 0.0,
                }
                self.results_.append(result)
        
        # Set best model
        leaderboard = self.get_leaderboard()
        if not leaderboard.empty:
            self.best_model_name_ = leaderboard.iloc[0]["model"]
            self.best_model_ = self.fitted_models_.get(self.best_model_name_)
        
        return leaderboard

    def grid_search(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        models: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]],
    ) -> pd.DataFrame:
        """Run GridSearchCV on all models with their param grids."""
        X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y = y_train.values if isinstance(y_train, pd.Series) else y_train
        X = np.nan_to_num(X, nan=0.0)
        cv = self._get_cv()
        
        for name, (estimator, param_grid) in models.items():
            if not param_grid:
                continue
            
            logger.info("GridSearchCV for %s", name)
            try:
                start_time = time.time()
                grid = GridSearchCV(
                    estimator=clone(estimator),
                    param_grid=param_grid,
                    scoring=self.scoring_metric,
                    cv=cv,
                    n_jobs=1,
                    refit=True,
                    error_score=np.nan,
                )
                grid.fit(X, y)
                train_time = time.time() - start_time
                
                # Update results
                best_score = grid.best_score_
                self.fitted_models_[name] = grid.best_estimator_
                
                # Update the result entry
                for res in self.results_:
                    if res["model"] == name:
                        if self.task_type == "classification":
                            res["f1_weighted_mean"] = round(float(best_score), 4)
                        else:
                            res["r2_mean"] = round(float(best_score) if "r2" in self.scoring_metric else float(-best_score), 4)
                        res["train_time_s"] = round(train_time, 2)
                        res["grid_best_params"] = grid.best_params_
                        break
                
                logger.info("%s GridSearchCV done in %.1fs, best score: %.4f", name, train_time, best_score)
                
            except Exception as e:
                logger.warning("GridSearchCV failed for %s: %s", name, str(e))
        
        # Update best model
        leaderboard = self.get_leaderboard()
        if not leaderboard.empty:
            self.best_model_name_ = leaderboard.iloc[0]["model"]
            self.best_model_ = self.fitted_models_.get(self.best_model_name_)
        
        return leaderboard

    def optuna_tune(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        top_n: int = 2,
        n_trials: int = 50,
    ) -> pd.DataFrame:
        """Run Optuna tuning on the top-N models from grid search."""
        X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y = y_train.values if isinstance(y_train, pd.Series) else y_train
        X = np.nan_to_num(X, nan=0.0)
        
        leaderboard = self.get_leaderboard()
        top_models = leaderboard.head(top_n)["model"].tolist()
        
        logger.info("Optuna tuning top %d models: %s", top_n, top_models)
        
        for model_name in top_models:
            search_space = get_optuna_search_space(model_name, self.task_type)
            if not search_space:
                logger.info("No Optuna search space for %s, skipping", model_name)
                continue
            
            # Find the original estimator
            base_estimator = self.fitted_models_.get(model_name)
            if base_estimator is None:
                continue
            
            cv = self._get_cv()
            
            def objective(trial: optuna.Trial) -> float:
                params: dict[str, Any] = {}
                for param_name, spec in search_space.items():
                    if spec["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name, spec["low"], spec["high"],
                            log=spec.get("log", False),
                        )
                    elif spec["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, spec["low"], spec["high"],
                            step=spec.get("step", 1),
                        )
                    elif spec["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, spec["choices"]
                        )
                
                model = clone(base_estimator)
                try:
                    model.set_params(**params)
                except (ValueError, TypeError):
                    pass
                
                scores = cross_validate(
                    model, X, y,
                    cv=cv,
                    scoring=self.scoring_metric,
                    n_jobs=1,
                    error_score=np.nan,
                )
                return float(np.nanmean(scores["test_score"]))
            
            try:
                study = optuna.create_study(
                    direction="maximize",
                    pruner=optuna.pruners.MedianPruner(),
                )
                study.optimize(objective, n_trials=n_trials, timeout=300, show_progress_bar=False)
                
                # Apply best params
                best_model = clone(base_estimator)
                try:
                    best_model.set_params(**study.best_params)
                except (ValueError, TypeError):
                    pass
                best_model.fit(X, y)
                self.fitted_models_[model_name] = best_model
                
                # Update results
                for res in self.results_:
                    if res["model"] == model_name:
                        best_score = study.best_value
                        if self.task_type == "classification":
                            res["f1_weighted_mean"] = round(float(best_score), 4)
                        else:
                            if best_score < 0:  # neg_rmse
                                res["rmse_mean"] = round(float(-best_score), 4)
                            else:
                                res["r2_mean"] = round(float(best_score), 4)
                        res["optuna_best_params"] = study.best_params
                        break
                
                logger.info(
                    "%s Optuna done: best score=%.4f, params=%s",
                    model_name, study.best_value, study.best_params,
                )
                
            except Exception as e:
                logger.warning("Optuna failed for %s: %s", model_name, str(e))
        
        # Update best model
        leaderboard = self.get_leaderboard()
        if not leaderboard.empty:
            self.best_model_name_ = leaderboard.iloc[0]["model"]
            self.best_model_ = self.fitted_models_.get(self.best_model_name_)
        
        return leaderboard

    def get_best_pipeline(
        self, preprocessing_pipeline: Pipeline
    ) -> Pipeline:
        """Return the full pipeline with the best model attached."""
        if self.best_model_ is None:
            raise ValueError("No model trained yet. Run train_and_evaluate first.")
        
        return Pipeline([
            ("preprocessing", preprocessing_pipeline),
            ("model", self.best_model_),
        ])
