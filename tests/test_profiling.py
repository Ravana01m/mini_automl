"""Tests for the data profiling engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from automl.profiling import profile_dataframe


def test_profile_detects_column_kinds() -> None:
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 100.0],
            "cat": ["a", "b", "a", "b", "a"],
            "id": [f"id_{i}" for i in range(5)],
            "const": [1, 1, 1, 1, 1],
            "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "target": [0, 1, 0, 1, 0],
        }
    )
    profile = profile_dataframe(df, "target")
    assert profile.n_rows == 5
    assert "num" in profile.numerical
    assert "cat" in profile.categorical
    assert "const" in profile.constant
    assert "all_null" in profile.all_null
    frame = profile.as_frame()
    assert "column" in frame.columns


def test_profile_reports_imbalance() -> None:
    df = pd.DataFrame({"x": range(20), "target": [0] * 18 + [1, 1]})
    profile = profile_dataframe(df, "target")
    assert profile.imbalance_ratio is not None
    assert profile.imbalance_ratio >= 3


def test_profile_does_not_drop_columns() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan], "target": [1, 0, 1]})
    profile = profile_dataframe(df, "target")
    assert "b" in profile.columns
    assert any("null" in n.lower() or "All values" in n for n in profile.columns["b"].notes)
