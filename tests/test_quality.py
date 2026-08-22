"""Tests for outlier handling and column pruning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from automl.quality import ColumnPruner, OutlierHandler


def test_outlier_bounds_learned_on_fit_only() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    test = pd.DataFrame({"a": [100.0]})
    handler = OutlierHandler(method="iqr", strategy="clip", factor=1.5)
    handler.fit(train)
    transformed = handler.transform(test)
    assert transformed["a"].iloc[0] < 100
    assert handler.reports_


def test_zscore_and_percentile_strategies() -> None:
    df = pd.DataFrame({"a": np.concatenate([np.linspace(0, 1, 40), [50.0]])})
    for method in ("zscore", "modified_zscore", "percentile"):
        handler = OutlierHandler(method=method, strategy="clip")
        out = handler.fit_transform(df)
        assert out["a"].max() < 50


def test_column_pruner_drops_constant_from_train() -> None:
    train = pd.DataFrame({"a": [1, 2, 3], "const": [5, 5, 5]})
    test = pd.DataFrame({"a": [4, 5], "const": [9, 10]})
    pruner = ColumnPruner(drop_constant=True)
    pruner.fit(train)
    out = pruner.transform(test)
    assert "const" not in out.columns
    assert "a" in out.columns
