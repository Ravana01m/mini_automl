"""End-to-end leakage-safe pipeline tests on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from automl.config import ExperimentConfig, TuningMode
from automl.pipeline_builder import AutoMLPipeline
from automl.utils import ValidationError, validate_csv, validate_target_column


def _fast_config(models: list[str]) -> ExperimentConfig:
    return ExperimentConfig(
        random_state=42,
        cv_folds=3,
        tuning_mode=TuningMode.FAST,
        skip_ann=True,
        enable_ensemble=False,
        baseline_enabled=False,
        skip_eda=True,
        skip_shap=True,
        shap_max_samples=8,
        selected_models=models,
        enable_polynomial=False,
    )


class TestEndToEndClassification:
    def test_iris_pipeline(self, iris_df: pd.DataFrame) -> None:
        automl = AutoMLPipeline(config=_fast_config(["LogisticRegression", "RandomForest"]))
        result = automl.run(iris_df, "target", task_type_override="classification")
        assert result["best_model_name"] is not None
        assert result["pipeline"] is not None
        preds = result["pipeline"].predict(result["X_test"])
        assert len(preds) == len(result["y_test"])

    def test_iris_model_accuracy_above_threshold(self, iris_df: pd.DataFrame) -> None:
        automl = AutoMLPipeline(config=_fast_config(["RandomForest"]))
        result = automl.run(iris_df, "target", task_type_override="classification")
        acc = result["test_metrics"]["accuracy"]
        assert acc >= 0.6


class TestEndToEndRegression:
    def test_housing_pipeline(self, housing_df: pd.DataFrame) -> None:
        automl = AutoMLPipeline(config=_fast_config(["Ridge", "RandomForest"]))
        result = automl.run(housing_df, "target", task_type_override="regression")
        assert result["pipeline"] is not None
        assert "rmse" in result["test_metrics"]

    def test_housing_model_r2_above_threshold(self, housing_df: pd.DataFrame) -> None:
        automl = AutoMLPipeline(config=_fast_config(["RandomForest"]))
        result = automl.run(housing_df, "target", task_type_override="regression")
        assert result["test_metrics"]["r2"] > 0.2


class TestEdgeCases:
    def test_messy_data_no_crash(self, messy_df: pd.DataFrame) -> None:
        automl = AutoMLPipeline(config=_fast_config(["Ridge"]))
        result = automl.run(messy_df, "target", task_type_override="regression")
        assert result["pipeline"] is not None

    def test_invalid_target_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValidationError):
            validate_target_column(df, "missing")

    def test_single_column_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_csv(pd.DataFrame({"a": [1, 2, 3]}))


def test_binary_and_imbalanced(binary_df: pd.DataFrame, imbalanced_df: pd.DataFrame) -> None:
    for frame in (binary_df, imbalanced_df):
        automl = AutoMLPipeline(config=_fast_config(["LogisticRegression"]))
        result = automl.run(frame, "target", task_type_override="classification")
        assert result["pipeline"] is not None


def test_one_model_failure_does_not_stop_run(housing_df: pd.DataFrame) -> None:
    automl = AutoMLPipeline(config=_fast_config(["Ridge", "NeuralNetwork"]))
    # NeuralNetwork is skipped via skip_ann; Ridge should still succeed.
    result = automl.run(housing_df, "target", task_type_override="regression")
    assert result["best_model_name"] is not None
