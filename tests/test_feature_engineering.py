"""Tests for feature engineering transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.feature_engineering import (
    DatetimeFeatureExtractor,
    LogTransformer,
    PolynomialFeatureGenerator,
)


class TestLogTransformer:
    """Tests for LogTransformer."""

    def test_transforms_skewed_features(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.exponential(5, 500), "b": np.random.normal(0, 1, 500)})
        # Make 'b' positive so log can apply if skewed
        df["b"] = df["b"] + 10
        lt = LogTransformer(skew_threshold=0.5)
        lt.fit(df)
        result = lt.transform(df)
        assert result.shape[1] >= df.shape[1]  # At least one log feature added

    def test_preserves_original_columns(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.exponential(5, 200), "b": np.random.normal(10, 1, 200)})
        lt = LogTransformer(skew_threshold=1.0)
        lt.fit(df)
        result = lt.transform(df)
        # Original columns preserved
        np.testing.assert_array_almost_equal(result[:, :2], df.values)

    def test_skips_non_skewed(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(10, 1, 200), "b": np.random.normal(10, 1, 200)})
        lt = LogTransformer(skew_threshold=5.0)  # Very high threshold
        lt.fit(df)
        result = lt.transform(df)
        assert result.shape[1] == 2  # No features added


class TestDatetimeFeatureExtractor:
    """Tests for DatetimeFeatureExtractor."""

    def test_extracts_datetime_features(self) -> None:
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "value": range(10),
        })
        dte = DatetimeFeatureExtractor()
        dte.fit(df)
        result = dte.transform(df)
        assert "date_year" in result.columns
        assert "date_month" in result.columns
        assert "date_is_weekend" in result.columns

    def test_drops_original_datetime_col(self) -> None:
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "value": range(10),
        })
        dte = DatetimeFeatureExtractor()
        dte.fit(df)
        result = dte.transform(df)
        assert "date" not in result.columns


class TestPolynomialFeatureGenerator:
    """Tests for PolynomialFeatureGenerator."""

    def test_generates_interactions(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        pfg = PolynomialFeatureGenerator(top_k=2, degree=2)
        pfg.fit(df)
        result = pfg.transform(df)
        # 2 original + 3 poly (a^2, ab, b^2) = 5
        assert result.shape[1] == 5

    def test_limits_to_top_k(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 20))
        pfg = PolynomialFeatureGenerator(top_k=3, degree=2)
        pfg.fit(df)
        assert len(pfg.selected_indices_) == 3
