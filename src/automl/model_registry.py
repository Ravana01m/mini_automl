"""Model registry: provides candidate model configurations for training.

Returns a dictionary of model_name -> (estimator, param_grid) for both
classification and regression tasks. Includes 6 models each:
Logistic/Ridge, RandomForest, XGBoost, LightGBM, SVM, Keras ANN.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from automl.ann_builder import get_keras_estimator

logger = logging.getLogger(__name__)


def get_models(
    task_type: str,
    input_dim: int | None = None,
    n_classes: int | None = None,
) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    """Return candidate models with their GridSearchCV param grids.

    Args:
        task_type: 'classification' or 'regression'.
        input_dim: Number of input features (needed for Keras ANN).
        n_classes: Number of classes (classification only, needed for Keras ANN).

    Returns:
        Dict mapping model_name -> (estimator_instance, param_grid_dict).
    """
    if task_type == "classification":
        models: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]] = {
            "LogisticRegression": (
                LogisticRegression(
                    max_iter=1000, random_state=42, solver="saga", n_jobs=-1
                ),
                {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                },
            ),
            "RandomForest": (
                RandomForestClassifier(random_state=42, n_jobs=-1),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                },
            ),
            "XGBoost": (
                XGBClassifier(
                    random_state=42, eval_metric="logloss",
                    n_jobs=-1, verbosity=0,
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.01, 0.1],
                },
            ),
            "LightGBM": (
                LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6, -1],
                    "learning_rate": [0.01, 0.1],
                },
            ),
            "SVM": (
                SVC(random_state=42, probability=True),
                {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["rbf"],
                },
            ),
        }
        # Add ANN if input_dim is known
        if input_dim is not None:
            models["NeuralNetwork"] = (
                get_keras_estimator(
                    task_type="classification",
                    input_dim=input_dim,
                    n_classes=n_classes or 2,
                    epochs=100,
                    batch_size=32,
                ),
                {
                    "batch_size": [16, 32],
                },
            )
    else:
        models = {
            "Ridge": (
                Ridge(random_state=42),
                {
                    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                },
            ),
            "RandomForest": (
                RandomForestRegressor(random_state=42, n_jobs=-1),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                },
            ),
            "XGBoost": (
                XGBRegressor(
                    random_state=42, n_jobs=-1, verbosity=0,
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.01, 0.1],
                },
            ),
            "LightGBM": (
                LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6, -1],
                    "learning_rate": [0.01, 0.1],
                },
            ),
            "SVR": (
                SVR(),
                {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["rbf"],
                },
            ),
        }
        if input_dim is not None:
            models["NeuralNetwork"] = (
                get_keras_estimator(
                    task_type="regression",
                    input_dim=input_dim,
                    epochs=100,
                    batch_size=32,
                ),
                {
                    "batch_size": [16, 32],
                },
            )
    
    logger.info("Model registry: %d models for %s", len(models), task_type)
    return models


def get_optuna_search_space(
    model_name: str, task_type: str
) -> dict[str, Any]:
    """Return Optuna hyperparameter search space for a given model.

    Args:
        model_name: Name of the model (must match keys from get_models).
        task_type: 'classification' or 'regression'.

    Returns:
        Dict defining Optuna trial parameter specs.
        Keys are param names, values are dicts with 'type' and 'args'.
    """
    spaces: dict[str, dict[str, Any]] = {
        "LogisticRegression": {
            "C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
            "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
        },
        "Ridge": {
            "alpha": {"type": "float", "low": 1e-3, "high": 1000.0, "log": True},
        },
        "RandomForest": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
        "XGBoost": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        },
        "LightGBM": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 10, "high": 150},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        },
        "SVM": {
            "C": {"type": "float", "low": 1e-2, "high": 100.0, "log": True},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        },
        "SVR": {
            "C": {"type": "float", "low": 1e-2, "high": 100.0, "log": True},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
            "epsilon": {"type": "float", "low": 0.01, "high": 1.0},
        },
        "NeuralNetwork": {
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        },
    }
    
    return spaces.get(model_name, {})
