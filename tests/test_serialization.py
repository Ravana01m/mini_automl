"""Serialization and raw-dataframe inference tests."""

from __future__ import annotations

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from automl.config import ExperimentConfig, TuningMode
from automl.datasets import make_binary
from automl.pipeline_builder import AutoMLPipeline
from automl.serialization import load_model, save_model


def test_save_load_predict_raw_dataframe(tmp_path: Path) -> None:
    df = make_binary(n=120, seed=11)
    config = ExperimentConfig(
        random_state=11,
        cv_folds=3,
        tuning_mode=TuningMode.FAST,
        skip_ann=True,
        enable_ensemble=False,
        baseline_enabled=False,
        skip_eda=True,
        skip_shap=True,
        enable_polynomial=False,
        shap_max_samples=10,
        selected_models=["LogisticRegression", "DecisionTree"],
    )
    automl = AutoMLPipeline(config=config)
    automl.run(df, "target", task_type_override="classification")
    path = tmp_path / "model.joblib"
    automl.export_pipeline(str(path))
    loaded = load_model(path)
    preds = loaded.predict(df.drop(columns=["target"]))
    assert len(preds) == len(df)
    if hasattr(loaded.pipeline, "predict_proba"):
        proba = loaded.predict_proba(df.drop(columns=["target"]))
        assert proba.shape[0] == len(df)


def test_save_model_helper(tmp_path: Path) -> None:
    pipe = Pipeline([("model", LogisticRegression())])
    bundle = {"pipeline": pipe, "task_type": "classification"}
    path = save_model(bundle, tmp_path / "x.joblib")
    assert Path(path).exists()
