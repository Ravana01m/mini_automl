"""Guards against train/test contamination."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from automl.config import ExperimentConfig, TuningMode
from automl.datasets import make_regression
from automl.feature_selection import CorrelationFilter
from automl.pipeline_factory import build_model_pipeline
from automl.model_registry import get_model_specs
from automl.quality import OutlierHandler


def test_outlier_bounds_ignore_test_extremes() -> None:
    train = pd.DataFrame({"x": np.linspace(0, 1, 50)})
    test = pd.DataFrame({"x": np.array([1000.0])})
    handler = OutlierHandler(method="iqr", strategy="clip")
    handler.fit(train)
    assert handler.bounds_["x"][1] < 10
    clipped = handler.transform(test)["x"].iloc[0]
    assert clipped < 10


def test_correlation_filter_uses_training_pairs_only() -> None:
    rng = np.random.RandomState(0)
    a = rng.randn(80)
    train = pd.DataFrame({"a": a, "b": a + 0.0001 * rng.randn(80), "c": rng.randn(80)})
    test = pd.DataFrame({"a": rng.randn(20), "b": rng.randn(20), "c": rng.randn(20)})
    filt = CorrelationFilter(0.95)
    filt.fit(train)
    out = filt.transform(test)
    assert out.shape[1] == 2


def test_full_pipeline_is_cv_safe() -> None:
    df = make_regression(n=90, seed=8)
    X, y = df.drop(columns=["target"]), df["target"]
    spec = [s for s in get_model_specs("regression", include_ann=False) if s.name == "Ridge"][0]
    pipe = build_model_pipeline(
        spec,
        X,
        "regression",
        ExperimentConfig(tuning_mode=TuningMode.FAST, enable_polynomial=False),
        baseline=False,
    )
    scores = cross_val_score(pipe, X, y, cv=KFold(3, shuffle=True, random_state=0), scoring="r2")
    assert np.isfinite(scores).all()
    assert "preprocessor" in pipe.named_steps
    assert "model" in pipe.named_steps
