"""Tests for intelligent feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from automl.feature_selection import FeatureSelectionEngine


def test_light_selection_drops_low_variance() -> None:
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "a": rng.randn(80),
            "b": rng.randn(80),
            "const": np.ones(80),
        }
    )
    y = (X["a"] > 0).astype(int)
    engine = FeatureSelectionEngine(task_type="classification", strategy="light")
    Xt = engine.fit_transform(X, y)
    assert Xt.shape[1] <= 3
    assert Xt.shape[0] == 80


def test_automatic_adapts_to_width() -> None:
    rng = np.random.RandomState(1)
    wide = pd.DataFrame(rng.randn(60, 40), columns=[f"f{i}" for i in range(40)])
    y = (wide["f0"] + wide["f1"] > 0).astype(int)
    engine = FeatureSelectionEngine(task_type="classification", strategy="automatic")
    Xt = engine.fit_transform(wide, y)
    assert 1 <= Xt.shape[1] <= 40
    assert engine.report_ is not None
