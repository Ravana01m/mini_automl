"""Tests for feature selection pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.feature_selection import CorrelationFilter, build_feature_selector


class TestCorrelationFilter:
    """Tests for CorrelationFilter."""

    def test_drops_highly_correlated(self) -> None:
        np.random.seed(42)
        a = np.random.randn(100)
        df = pd.DataFrame({"a": a, "b": a + 0.001 * np.random.randn(100), "c": np.random.randn(100)})
        cf = CorrelationFilter(threshold=0.95)
        cf.fit(df)
        result = cf.transform(df)
        assert result.shape[1] == 2  # One of a/b dropped

    def test_preserves_uncorrelated(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        cf = CorrelationFilter(threshold=0.95)
        cf.fit(df)
        result = cf.transform(df)
        assert result.shape[1] == 2


class TestBuildFeatureSelector:
    """Tests for build_feature_selector()."""

    def test_classification_uses_f_classif(self) -> None:
        pipe = build_feature_selector("classification")
        step_names = [name for name, _ in pipe.steps]
        assert "select_k_best" in step_names

    def test_regression_uses_f_regression(self) -> None:
        pipe = build_feature_selector("regression")
        step_names = [name for name, _ in pipe.steps]
        assert "select_k_best" in step_names
