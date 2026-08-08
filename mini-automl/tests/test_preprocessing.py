"""Tests for preprocessing pipeline builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.preprocessing import OutlierClipper, build_preprocessor


class TestOutlierClipper:
    """Tests for OutlierClipper transformer."""

    def test_clips_high_outliers(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
        clipper = OutlierClipper(factor=1.5)
        clipper.fit(df)
        result = clipper.transform(df)
        assert result["a"].max() < 100

    def test_clips_low_outliers(self) -> None:
        df = pd.DataFrame({"a": [-100, 1, 2, 3, 4, 5]})
        clipper = OutlierClipper(factor=1.5)
        clipper.fit(df)
        result = clipper.transform(df)
        assert result["a"].min() > -100

    def test_preserves_normal_values(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        clipper = OutlierClipper(factor=1.5)
        clipper.fit(df)
        result = clipper.transform(df)
        np.testing.assert_array_equal(result["a"].values, df["a"].values)


class TestBuildPreprocessor:
    """Tests for build_preprocessor()."""

    def test_handles_numeric_columns(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        preprocessor, num_cols, cat_cols = build_preprocessor(df, "target", "regression")
        assert len(num_cols) == 2
        assert len(cat_cols) == 0

    def test_handles_low_cardinality_categorical(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
        preprocessor, num_cols, cat_cols = build_preprocessor(df, "target", "classification")
        assert "b" in cat_cols

    def test_handles_high_cardinality_categorical(self) -> None:
        cats = [f"cat_{i}" for i in range(20)]
        df = pd.DataFrame({"a": range(20), "b": cats})
        preprocessor, num_cols, cat_cols = build_preprocessor(df, "target", "classification")
        assert "b" in cat_cols

    def test_handles_all_null_column(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]})
        preprocessor, num_cols, cat_cols = build_preprocessor(df, "target", "regression")
        # All-null column should be excluded
        assert "b" not in num_cols
        assert "b" not in cat_cols
