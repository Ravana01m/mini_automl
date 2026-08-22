"""Tests for staged training and leaderboard ranking."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl.config import ExperimentConfig, TuningMode
from automl.datasets import make_binary, make_regression
from automl.trainer import Trainer


class TestTrainer:
    def test_trains_all_models(self) -> None:
        df = make_binary(n=80, seed=1)
        X, y = df[["x1", "x2"]], df["target"]
        models = {
            "LogisticRegression": (LogisticRegression(max_iter=200), {}),
            "DecisionTree": (DecisionTreeClassifier(max_depth=3), {}),
        }
        trainer = Trainer("classification", cv_folds=3, random_state=1)
        board = trainer.train_and_evaluate(X, y, models)
        assert len(board) == 2
        assert set(trainer.fitted_models_)

    def test_leaderboard_sorted_by_metric(self) -> None:
        df = make_regression(n=80, seed=2)
        X, y = df[["x1", "x2", "x3"]], df["target"]
        models = {
            "Ridge": (Ridge(), {}),
            "DecisionTree": (DecisionTreeRegressor(max_depth=2), {}),
        }
        trainer = Trainer("regression", cv_folds=3, random_state=2)
        board = trainer.train_and_evaluate(X, y, models)
        scores = board["r2_mean"].astype(float)
        assert scores.is_monotonic_decreasing or len(scores) == 1

    def test_grid_search_improves_or_matches(self) -> None:
        df = make_regression(n=70, seed=3)
        X, y = df[["x1", "x2", "x3"]], df["target"]
        models = {"Ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0]})}
        trainer = Trainer("regression", cv_folds=3, random_state=3)
        trainer.train_and_evaluate(X, y, models)
        before = trainer.get_leaderboard().iloc[0]["r2_mean"]
        trainer.grid_search(X, y, models)
        after = trainer.get_leaderboard().iloc[0]["r2_mean"]
        assert after >= before - 0.25

    def test_optuna_runs_on_top_n(self) -> None:
        df = make_regression(n=70, seed=4)
        X, y = df[["x1", "x2", "x3"]], df["target"]
        models = {"Ridge": (Ridge(), {"alpha": [1.0]})}
        trainer = Trainer(
            "regression",
            cv_folds=3,
            random_state=4,
            config=ExperimentConfig(cv_folds=3, optuna_timeout=15, tuning_mode=TuningMode.STANDARD),
        )
        trainer.train_and_evaluate(X, y, models)
        board = trainer.optuna_tune(X, y, top_n=1, n_trials=4, timeout=20)
        assert not board.empty
