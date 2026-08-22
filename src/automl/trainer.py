"""Staged, leakage-safe training, comparison, and hyperparameter search."""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore[assignment]
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from automl.config import ExperimentConfig, TuningMode
from automl.evaluation import scoring_for_task
from automl.model_registry import get_optuna_search_space

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_cv(task_type: str, config: ExperimentConfig):
    if task_type == "classification":
        if config.cv_repeats > 1:
            return RepeatedStratifiedKFold(
                n_splits=config.cv_folds,
                n_repeats=config.cv_repeats,
                random_state=config.random_state,
            )
        return StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=config.shuffle,
            random_state=config.random_state,
        )
    if config.cv_repeats > 1:
        return RepeatedKFold(
            n_splits=config.cv_folds,
            n_repeats=config.cv_repeats,
            random_state=config.random_state,
        )
    return KFold(
        n_splits=config.cv_folds,
        shuffle=config.shuffle,
        random_state=config.random_state,
    )


class Trainer:
    """Orchestrates leakage-safe model training, comparison, and tuning."""

    def __init__(
        self,
        task_type: str,
        scoring_metric: str | None = None,
        cv_folds: int = 5,
        random_state: int = 42,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.config = config or ExperimentConfig(
            cv_folds=cv_folds, random_state=random_state
        )
        if scoring_metric is None:
            self.scoring_metric = (
                "f1_weighted"
                if task_type == "classification"
                else "neg_root_mean_squared_error"
            )
        else:
            self.scoring_metric = scoring_metric
        self.results_: list[dict[str, Any]] = []
        self.cv_results_: dict[str, list[float]] = {}
        self.best_model_: BaseEstimator | None = None
        self.best_model_name_: str | None = None
        self.fitted_models_: dict[str, BaseEstimator] = {}
        self.families_: dict[str, str] = {}

    def _get_cv(self):
        return make_cv(self.task_type, self.config)

    def _to_xy(self, X_train: Any, y_train: Any) -> tuple[Any, Any]:
        # Keep DataFrames so datetime / column-name transformers work.
        y = y_train.values if isinstance(y_train, pd.Series) else y_train
        return X_train, y

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        models: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]],
        progress_callback: Any | None = None,
        families: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Cross-validate each candidate. One failure never aborts the run."""
        self.results_ = []
        self.cv_results_ = {}
        self.families_ = families or {}
        cv = self._get_cv()
        X, y = self._to_xy(X_train, y_train)
        scoring = scoring_for_task(self.task_type)
        total = len(models)

        for idx, (name, (estimator, _)) in enumerate(models.items()):
            logger.info("Training %s (%d/%d)", name, idx + 1, total)
            if progress_callback:
                progress_callback(name, idx + 1, total)
            result: dict[str, Any] = {
                "model": name,
                "family": self.families_.get(name, ""),
                "train_time_s": 0.0,
                "status": "failed",
            }
            try:
                start = time.time()
                model = clone(estimator)
                cv_result = cross_validate(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=1,
                    error_score="raise",
                )
                train_time = time.time() - start
                result.update(self._summarize_cv(cv_result, train_time))
                result["status"] = "success"
                if self.task_type == "classification":
                    self.cv_results_[name] = list(cv_result.get("test_f1_weighted", []))
                else:
                    self.cv_results_[name] = list(cv_result.get("test_r2", []))
                fit_start = time.time()
                model.fit(X, y)
                result["fit_time_s"] = round(time.time() - fit_start, 2)
                self.fitted_models_[name] = model
                logger.info("%s completed in %.1fs", name, train_time)
            except Exception as exc:
                logger.warning("Failed to train %s: %s", name, exc)
                result["error"] = str(exc)
            self.results_.append(result)

        leaderboard = self.get_leaderboard()
        if not leaderboard.empty:
            self.best_model_name_ = str(leaderboard.iloc[0]["model"])
            self.best_model_ = self.fitted_models_.get(self.best_model_name_)
        return leaderboard

    def _summarize_cv(self, cv_result: dict[str, np.ndarray], train_time: float) -> dict[str, Any]:
        out: dict[str, Any] = {"train_time_s": round(train_time, 2)}
        if self.task_type == "classification":
            for key in ("accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"):
                scores = cv_result.get(f"test_{key}")
                if scores is None:
                    continue
                out[f"{key}_mean"] = round(float(np.mean(scores)), 4)
                out[f"{key}_std"] = round(float(np.std(scores)), 4)
            out["cv_score"] = out.get("f1_weighted_mean")
            out["cv_std"] = out.get("f1_weighted_std")
        else:
            r2 = cv_result.get("test_r2", np.array([np.nan]))
            rmse = -cv_result.get("test_neg_rmse", np.array([np.nan]))
            mae = -cv_result.get("test_neg_mae", np.array([np.nan]))
            out["r2_mean"] = round(float(np.nanmean(r2)), 4)
            out["r2_std"] = round(float(np.nanstd(r2)), 4)
            out["rmse_mean"] = round(float(np.nanmean(rmse)), 4)
            out["rmse_std"] = round(float(np.nanstd(rmse)), 4)
            out["mae_mean"] = round(float(np.nanmean(mae)), 4)
            out["cv_score"] = out["r2_mean"]
            out["cv_std"] = out["r2_std"]
        return out

    def get_leaderboard(self) -> pd.DataFrame:
        if not self.results_:
            return pd.DataFrame()
        leaderboard = pd.DataFrame(self.results_)
        metric_col = "f1_weighted_mean" if self.task_type == "classification" else "r2_mean"
        if metric_col in leaderboard.columns:
            rank_key = leaderboard[metric_col].astype(float).fillna(-np.inf)
            if "status" in leaderboard.columns:
                rank_key = rank_key.where(leaderboard["status"] == "success", -np.inf)
            leaderboard = leaderboard.loc[rank_key.sort_values(ascending=False).index].reset_index(drop=True)
        for i, row in leaderboard.iterrows():
            name = row["model"]
            if name in self.cv_results_:
                continue
            if self.task_type == "classification" and "f1_weighted_mean" in row:
                self.cv_results_[name] = [row.get("f1_weighted_mean")]
            elif "r2_mean" in row:
                self.cv_results_[name] = [row.get("r2_mean")]
        return leaderboard

    def _prefix_grid(self, estimator: BaseEstimator, param_grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not param_grid:
            return {}
        if hasattr(estimator, "named_steps") and "model" in getattr(estimator, "named_steps", {}):
            return {f"model__{k}": v for k, v in param_grid.items()}
        return param_grid

    def grid_search(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        models: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]],
        names: list[str] | None = None,
    ) -> pd.DataFrame:
        X, y = self._to_xy(X_train, y_train)
        cv = self._get_cv()
        selected = names or list(models)
        for name in selected:
            if name not in models:
                continue
            estimator, param_grid = models[name]
            grid = self._prefix_grid(estimator, param_grid)
            if not grid:
                continue
            logger.info("GridSearchCV for %s", name)
            try:
                start = time.time()
                search = GridSearchCV(
                    estimator=clone(estimator),
                    param_grid=grid,
                    scoring=self.scoring_metric,
                    cv=cv,
                    n_jobs=1,
                    refit=True,
                    error_score=np.nan,
                )
                search.fit(X, y)
                self.fitted_models_[name] = search.best_estimator_
                self._update_score(name, float(search.best_score_), time.time() - start, search.best_params_)
            except Exception as exc:
                logger.warning("GridSearchCV failed for %s: %s", name, exc)
        return self.get_leaderboard()

    def _update_score(
        self,
        name: str,
        best_score: float,
        train_time: float,
        params: dict[str, Any] | None = None,
        source: str = "grid",
    ) -> None:
        for res in self.results_:
            if res["model"] != name:
                continue
            if self.task_type == "classification":
                res["f1_weighted_mean"] = round(float(best_score), 4)
                res["cv_score"] = res["f1_weighted_mean"]
            elif self.scoring_metric == "r2":
                res["r2_mean"] = round(float(best_score), 4)
                res["cv_score"] = res["r2_mean"]
            else:
                # neg RMSE or r2 depending on metric
                if best_score < 0:
                    res["rmse_mean"] = round(float(-best_score), 4)
                    res["cv_score"] = res.get("r2_mean")
                else:
                    res["r2_mean"] = round(float(best_score), 4)
                    res["cv_score"] = res["r2_mean"]
            res["train_time_s"] = round(train_time, 2)
            res[f"{source}_best_params"] = params
            res["status"] = "success"
            break

    def optuna_tune(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        top_n: int = 2,
        n_trials: int = 50,
        timeout: int | None = None,
    ) -> pd.DataFrame:
        X, y = self._to_xy(X_train, y_train)
        leaderboard = self.get_leaderboard()
        if leaderboard.empty:
            return leaderboard
        top_models = [
            m
            for m in leaderboard.loc[leaderboard["status"] == "success", "model"].tolist()
            if m in self.fitted_models_
        ][:top_n]
        if optuna is None:
            logger.warning("Optuna is not installed; skipping deep tuning")
            return self.get_leaderboard()
        logger.info("Optuna tuning top %d models: %s", top_n, top_models)
        timeout = timeout if timeout is not None else self.config.optuna_timeout

        for model_name in top_models:
            search_space = get_optuna_search_space(model_name, self.task_type)
            if not search_space:
                continue
            base_estimator = self.fitted_models_.get(model_name)
            if base_estimator is None:
                continue
            prefix = "model__" if hasattr(base_estimator, "named_steps") else ""
            cv = self._get_cv()

            def objective(trial: optuna.Trial, space=search_space, est=base_estimator, pfx=prefix) -> float:
                params: dict[str, Any] = {}
                for param_name, spec in space.items():
                    key = f"{pfx}{param_name}" if pfx and not param_name.startswith("model__") else param_name
                    raw = param_name.split("__")[-1] if param_name.startswith("optimizer__") else param_name
                    spec_use = spec
                    if spec_use["type"] == "float":
                        value = trial.suggest_float(
                            raw, spec_use["low"], spec_use["high"], log=spec_use.get("log", False)
                        )
                    elif spec_use["type"] == "int":
                        value = trial.suggest_int(
                            raw, spec_use["low"], spec_use["high"], step=spec_use.get("step", 1)
                        )
                    else:
                        value = trial.suggest_categorical(raw, spec_use["choices"])
                    params[key] = value
                model = clone(est)
                try:
                    model.set_params(**params)
                except (ValueError, TypeError):
                    return float("nan")
                scores = cross_validate(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring=self.scoring_metric,
                    n_jobs=1,
                    error_score=np.nan,
                )
                return float(np.nanmean(scores["test_score"]))

            try:
                sampler = optuna.samplers.TPESampler(seed=self.random_state)
                study = optuna.create_study(
                    direction="maximize",
                    pruner=optuna.pruners.MedianPruner(),
                    sampler=sampler,
                )
                study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
                best_model = clone(base_estimator)
                prefixed = {}
                for k, v in study.best_params.items():
                    key = f"{prefix}{k}" if prefix and not k.startswith("model__") else k
                    prefixed[key] = v
                try:
                    best_model.set_params(**prefixed)
                except (ValueError, TypeError):
                    pass
                best_model.fit(X, y)
                self.fitted_models_[model_name] = best_model
                self._update_score(
                    model_name,
                    float(study.best_value),
                    0.0,
                    study.best_params,
                    source="optuna",
                )
            except Exception as exc:
                logger.warning("Optuna failed for %s: %s", model_name, exc)
        return self.get_leaderboard()

    def staged_search(
        self,
        X_train: Any,
        y_train: Any,
        models: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]],
        families: dict[str, str] | None = None,
        stages: dict[str, int] | None = None,
        progress_callback: Any | None = None,
    ) -> pd.DataFrame:
        """Fast baselines → promising models → tune top models."""
        mode = self.config.tuning_mode
        stage_map = stages or {}
        stage1 = [n for n, s in stage_map.items() if s <= 1] or list(models)
        stage1_models = {k: models[k] for k in stage1 if k in models}
        self.train_and_evaluate(X_train, y_train, stage1_models, progress_callback, families)

        remaining = [n for n in models if n not in stage1_models]
        if mode != TuningMode.FAST and remaining:
            self.train_and_evaluate_append(X_train, y_train, {k: models[k] for k in remaining}, families)

        leaderboard = self.get_leaderboard()
        successful = leaderboard.loc[leaderboard.get("status", "success") == "success", "model"].tolist()
        top_for_grid = successful[: self.config.stage1_top_k]
        if mode == TuningMode.FAST:
            return leaderboard
        if mode in {TuningMode.STANDARD, TuningMode.DEEP} and top_for_grid:
            grid_models = {k: models[k] for k in top_for_grid if k in models}
            # FAST grids only; skip ANN grids in FAST already handled
            if mode == TuningMode.STANDARD:
                self.grid_search(X_train, y_train, grid_models, names=top_for_grid[:3])
            top_for_optuna = self.get_leaderboard()
            top_names = top_for_optuna.loc[
                top_for_optuna["status"] == "success", "model"
            ].tolist()[: self.config.stage2_top_k]
            if mode == TuningMode.DEEP or (mode == TuningMode.STANDARD and self.config.optuna_trials > 0):
                trials = 8 if mode == TuningMode.STANDARD else self.config.optuna_trials
                self.optuna_tune(
                    X_train,
                    y_train,
                    top_n=len(top_names),
                    n_trials=trials,
                    timeout=self.config.optuna_timeout,
                )
        return self.get_leaderboard()

    def train_and_evaluate_append(
        self,
        X_train: Any,
        y_train: Any,
        models: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]],
        families: dict[str, str] | None = None,
    ) -> None:
        existing = list(self.results_)
        fitted = dict(self.fitted_models_)
        cv_results = dict(self.cv_results_)
        extra = Trainer(
            task_type=self.task_type,
            scoring_metric=self.scoring_metric,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            config=self.config,
        )
        extra.train_and_evaluate(X_train, y_train, models, families=families)
        self.results_ = existing + extra.results_
        self.fitted_models_ = {**fitted, **extra.fitted_models_}
        self.cv_results_ = {**cv_results, **extra.cv_results_}

    def get_best_pipeline(self, preprocessing_pipeline: Pipeline | None = None) -> Pipeline:
        if self.best_model_ is None:
            raise ValueError("No model trained yet. Run train_and_evaluate first.")
        if isinstance(self.best_model_, Pipeline) or hasattr(self.best_model_, "named_steps"):
            return self.best_model_
        if preprocessing_pipeline is None:
            return Pipeline([("model", self.best_model_)])
        return Pipeline(
            [
                ("preprocessing", preprocessing_pipeline),
                ("model", self.best_model_),
            ]
        )
