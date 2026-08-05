"""End-to-end pipeline tests on real datasets.

These tests prove the pipeline generalizes across different CSVs,
not just one hardcoded dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestEndToEndClassification:
    """End-to-end tests with classification datasets."""

    def test_iris_pipeline(self, iris_df: pd.DataFrame) -> None:
        """Full pipeline runs successfully on Iris dataset."""
        pass  # TODO: implement in Phase 7

    def test_iris_model_accuracy_above_threshold(self, iris_df: pd.DataFrame) -> None:
        """Best model should achieve > 80% accuracy on Iris."""
        pass  # TODO: implement in Phase 7


class TestEndToEndRegression:
    """End-to-end tests with regression datasets."""

    def test_housing_pipeline(self, housing_df: pd.DataFrame) -> None:
        """Full pipeline runs successfully on housing dataset."""
        pass  # TODO: implement in Phase 7

    def test_housing_model_r2_above_threshold(self, housing_df: pd.DataFrame) -> None:
        """Best model should achieve R² > 0.5 on housing data."""
        pass  # TODO: implement in Phase 7


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_messy_data_no_crash(self, messy_df: pd.DataFrame) -> None:
        """Pipeline should handle messy data without crashing."""
        pass  # TODO: implement in Phase 7

    def test_invalid_target_raises(self) -> None:
        """Non-existent target column should raise ValidationError."""
        pass  # TODO: implement in Phase 7

    def test_single_column_raises(self) -> None:
        """DataFrame with only 1 column should raise ValidationError."""
        pass  # TODO: implement in Phase 7
