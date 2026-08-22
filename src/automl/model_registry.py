"""Model registry with family metadata and search spaces."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl.ann_builder import get_keras_estimator, scikeras_available, tensorflow_available
from automl.types import ModelSpec

logger = logging.getLogger(__name__)


def _xgb():
    from xgboost import XGBClassifier, XGBRegressor

    return XGBClassifier, XGBRegressor


def _lgbm():
    from lightgbm import LGBMClassifier, LGBMRegressor

    return LGBMClassifier, LGBMRegressor


def _tree_grid() -> dict[str, list[Any]]:
    return {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }


def _boost_grid() -> dict[str, list[Any]]:
    return {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }


def _tree_optuna() -> dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 80, "high": 400, "step": 40},
        "max_depth": {"type": "int", "low": 3, "high": 16},
        "min_samples_split": {"type": "int", "low": 2, "high": 12},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 8},
    }


def _boost_optuna() -> dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 80, "high": 400, "step": 40},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    }


def _classification_specs(random_state: int = 42) -> list[ModelSpec]:
    XGBClassifier, _ = _xgb()
    LGBMClassifier, _ = _lgbm()
    return [
        ModelSpec(
            name="LogisticRegression",
            family="linear",
            task="classification",
            factory=lambda **kw: LogisticRegression(
                max_iter=800, solver="saga", n_jobs=1, random_state=random_state, **kw
            ),
            param_grid={"C": [0.1, 1.0, 10.0], "penalty": ["l2"]},
            optuna_space={
                "C": {"type": "float", "low": 1e-3, "high": 50.0, "log": True},
                "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
            },
            needs_scaling=True,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=1,
            supports_class_weight=True,
        ),
        ModelSpec(
            name="RidgeClassifier",
            family="linear",
            task="classification",
            factory=lambda **kw: RidgeClassifier(random_state=random_state, **kw),
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            optuna_space={"alpha": {"type": "float", "low": 1e-3, "high": 100.0, "log": True}},
            needs_scaling=True,
            has_predict_proba=False,
            primary_metric="f1_weighted",
            stage=1,
            supports_class_weight=True,
        ),
        ModelSpec(
            name="DecisionTree",
            family="tree",
            task="classification",
            factory=lambda **kw: DecisionTreeClassifier(random_state=random_state, **kw),
            param_grid={"max_depth": [4, 8, None], "min_samples_split": [2, 8]},
            optuna_space={
                "max_depth": {"type": "int", "low": 2, "high": 16},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
            },
            needs_scaling=False,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=1,
            supports_class_weight=True,
        ),
        ModelSpec(
            name="RandomForest",
            family="tree",
            task="classification",
            factory=lambda **kw: RandomForestClassifier(
                random_state=random_state, n_jobs=1, **kw
            ),
            param_grid=_tree_grid(),
            optuna_space=_tree_optuna(),
            needs_scaling=False,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=1,
            supports_class_weight=True,
        ),
        ModelSpec(
            name="ExtraTrees",
            family="tree",
            task="classification",
            factory=lambda **kw: ExtraTreesClassifier(random_state=random_state, n_jobs=1, **kw),
            param_grid={"n_estimators": [100, 200], "max_depth": [6, None]},
            optuna_space=_tree_optuna(),
            needs_scaling=False,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=2,
            supports_class_weight=True,
        ),
        ModelSpec(
            name="GradientBoosting",
            family="boosting",
            task="classification",
            factory=lambda **kw: GradientBoostingClassifier(random_state=random_state, **kw),
            param_grid={"n_estimators": [80, 150], "learning_rate": [0.05, 0.1]},
            optuna_space={
                "n_estimators": {"type": "int", "low": 60, "high": 250, "step": 20},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
                "max_depth": {"type": "int", "low": 2, "high": 5},
            },
            needs_scaling=False,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=2,
        ),
        ModelSpec(
            name="HistGradientBoosting",
            family="boosting",
            task="classification",
            factory=lambda **kw: HistGradientBoostingClassifier(random_state=random_state, **kw),
            param_grid={"max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
            optuna_space={
                "max_depth": {"type": "int", "low": 2, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
                "max_iter": {"type": "int", "low": 80, "high": 250, "step": 20},
            },
            needs_scaling=False,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=1,
        ),
        ModelSpec(
            name="SVM",
            family="svm",
            task="classification",
            factory=lambda **kw: SVC(probability=True, random_state=random_state, **kw),
            param_grid={"C": [0.5, 2.0], "kernel": ["rbf"]},
            optuna_space={
                "C": {"type": "float", "low": 1e-2, "high": 30.0, "log": True},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
            },
            needs_scaling=True,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=2,
            supports_class_weight=True,
        ),
        ModelSpec(
            name="XGBoost",
            family="boosting",
            task="classification",
            factory=lambda **kw: XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=1,
                verbosity=0,
                **kw,
            ),
            param_grid=_boost_grid(),
            optuna_space=_boost_optuna(),
            needs_scaling=False,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=2,
        ),
        ModelSpec(
            name="LightGBM",
            family="boosting",
            task="classification",
            factory=lambda **kw: LGBMClassifier(
                random_state=random_state, n_jobs=1, verbose=-1, **kw
            ),
            param_grid=_boost_grid(),
            optuna_space=_boost_optuna() | {
                "num_leaves": {"type": "int", "low": 15, "high": 80}
            },
            needs_scaling=False,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=2,
        ),
        ModelSpec(
            name="NeuralNetwork",
            family="neural",
            task="classification",
            factory=lambda **kw: get_keras_estimator("classification", **kw),
            param_grid={"batch_size": [16, 32]},
            optuna_space={
                "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
                "optimizer__learning_rate": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-2,
                    "log": True,
                },
            },
            needs_scaling=True,
            has_predict_proba=True,
            primary_metric="f1_weighted",
            stage=3,
        ),
    ]


def _regression_specs(random_state: int = 42) -> list[ModelSpec]:
    _, XGBRegressor = _xgb()
    _, LGBMRegressor = _lgbm()
    return [
        ModelSpec(
            name="LinearRegression",
            family="linear",
            task="regression",
            factory=lambda **kw: LinearRegression(**kw),
            param_grid={},
            optuna_space={},
            needs_scaling=True,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=1,
        ),
        ModelSpec(
            name="Ridge",
            family="linear",
            task="regression",
            factory=lambda **kw: Ridge(random_state=random_state, **kw),
            param_grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
            optuna_space={"alpha": {"type": "float", "low": 1e-3, "high": 500.0, "log": True}},
            needs_scaling=True,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=1,
        ),
        ModelSpec(
            name="Lasso",
            family="linear",
            task="regression",
            factory=lambda **kw: Lasso(random_state=random_state, max_iter=4000, **kw),
            param_grid={"alpha": [0.001, 0.01, 0.1]},
            optuna_space={"alpha": {"type": "float", "low": 1e-4, "high": 1.0, "log": True}},
            needs_scaling=True,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=2,
        ),
        ModelSpec(
            name="ElasticNet",
            family="linear",
            task="regression",
            factory=lambda **kw: ElasticNet(random_state=random_state, max_iter=4000, **kw),
            param_grid={"alpha": [0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
            optuna_space={
                "alpha": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
                "l1_ratio": {"type": "float", "low": 0.05, "high": 0.95},
            },
            needs_scaling=True,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=2,
        ),
        ModelSpec(
            name="DecisionTree",
            family="tree",
            task="regression",
            factory=lambda **kw: DecisionTreeRegressor(random_state=random_state, **kw),
            param_grid={"max_depth": [4, 8, None]},
            optuna_space={"max_depth": {"type": "int", "low": 2, "high": 16}},
            needs_scaling=False,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=1,
        ),
        ModelSpec(
            name="RandomForest",
            family="tree",
            task="regression",
            factory=lambda **kw: RandomForestRegressor(random_state=random_state, n_jobs=1, **kw),
            param_grid=_tree_grid(),
            optuna_space=_tree_optuna(),
            needs_scaling=False,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=1,
        ),
        ModelSpec(
            name="ExtraTrees",
            family="tree",
            task="regression",
            factory=lambda **kw: ExtraTreesRegressor(random_state=random_state, n_jobs=1, **kw),
            param_grid={"n_estimators": [100, 200], "max_depth": [6, None]},
            optuna_space=_tree_optuna(),
            needs_scaling=False,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=2,
        ),
        ModelSpec(
            name="GradientBoosting",
            family="boosting",
            task="regression",
            factory=lambda **kw: GradientBoostingRegressor(random_state=random_state, **kw),
            param_grid={"n_estimators": [80, 150], "learning_rate": [0.05, 0.1]},
            optuna_space={
                "n_estimators": {"type": "int", "low": 60, "high": 250, "step": 20},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
                "max_depth": {"type": "int", "low": 2, "high": 5},
            },
            needs_scaling=False,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=2,
        ),
        ModelSpec(
            name="HistGradientBoosting",
            family="boosting",
            task="regression",
            factory=lambda **kw: HistGradientBoostingRegressor(random_state=random_state, **kw),
            param_grid={"max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
            optuna_space={
                "max_depth": {"type": "int", "low": 2, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
            },
            needs_scaling=False,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=1,
        ),
        ModelSpec(
            name="SVR",
            family="svm",
            task="regression",
            factory=lambda **kw: SVR(**kw),
            param_grid={"C": [0.5, 2.0], "kernel": ["rbf"]},
            optuna_space={
                "C": {"type": "float", "low": 1e-2, "high": 30.0, "log": True},
                "epsilon": {"type": "float", "low": 0.01, "high": 0.5},
            },
            needs_scaling=True,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=2,
        ),
        ModelSpec(
            name="XGBoost",
            family="boosting",
            task="regression",
            factory=lambda **kw: XGBRegressor(
                random_state=random_state, n_jobs=1, verbosity=0, **kw
            ),
            param_grid=_boost_grid(),
            optuna_space=_boost_optuna(),
            needs_scaling=False,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=2,
        ),
        ModelSpec(
            name="LightGBM",
            family="boosting",
            task="regression",
            factory=lambda **kw: LGBMRegressor(
                random_state=random_state, n_jobs=1, verbose=-1, **kw
            ),
            param_grid=_boost_grid(),
            optuna_space=_boost_optuna() | {
                "num_leaves": {"type": "int", "low": 15, "high": 80}
            },
            needs_scaling=False,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=2,
        ),
        ModelSpec(
            name="NeuralNetwork",
            family="neural",
            task="regression",
            factory=lambda **kw: get_keras_estimator("regression", **kw),
            param_grid={"batch_size": [16, 32]},
            optuna_space={
                "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
                "optimizer__learning_rate": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-2,
                    "log": True,
                },
            },
            needs_scaling=True,
            has_predict_proba=False,
            primary_metric="neg_root_mean_squared_error",
            stage=3,
        ),
    ]


def list_model_specs(task_type: str, random_state: int = 42) -> list[ModelSpec]:
    specs = _classification_specs(random_state) if task_type == "classification" else _regression_specs(random_state)
    if not tensorflow_available() or not scikeras_available():
        specs = [s for s in specs if s.name != "NeuralNetwork"]
        logger.warning("TensorFlow/SciKeras unavailable; NeuralNetwork removed from registry")
    return specs


def get_model_specs(
    task_type: str,
    selected: list[str] | None = None,
    include_ann: bool = True,
    random_state: int = 42,
) -> list[ModelSpec]:
    specs = list_model_specs(task_type, random_state)
    if not include_ann:
        specs = [s for s in specs if s.family != "neural"]
    if selected:
        selected_l = {n.lower() for n in selected}
        specs = [s for s in specs if s.name.lower() in selected_l]
    return specs


def get_models(
    task_type: str,
    input_dim: int | None = None,
    n_classes: int | None = None,
) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    """Backward-compatible registry used by older tests and callers."""
    models: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]] = {}
    for spec in get_model_specs(task_type, include_ann=input_dim is not None):
        kwargs: dict[str, Any] = {}
        if spec.family == "neural":
            if input_dim is None:
                continue
            kwargs["input_dim"] = input_dim
            if n_classes is not None:
                kwargs["n_classes"] = n_classes
            kwargs["epochs"] = 40
        try:
            estimator = spec.factory(**kwargs)
        except Exception as exc:
            logger.warning("Skipping %s: %s", spec.name, exc)
            continue
        models[spec.name] = (estimator, spec.param_grid)
    logger.info("Model registry: %d models for %s", len(models), task_type)
    return models


def get_optuna_search_space(model_name: str, task_type: str) -> dict[str, Any]:
    for spec in list_model_specs(task_type):
        if spec.name == model_name:
            return spec.optuna_space
    return {}
