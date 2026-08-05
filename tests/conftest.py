"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_iris

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def iris_df() -> pd.DataFrame:
    """Load the Iris dataset as a DataFrame (classification)."""
    data = load_iris(as_frame=True)
    df = data.frame  # type: ignore[union-attr]
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    return df


@pytest.fixture
def housing_df() -> pd.DataFrame:
    """Load a simple housing dataset for regression testing."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # type: ignore[union-attr]
    return df


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Create a messy DataFrame with missing values, mixed types, outliers."""
    import numpy as np
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "age": rng.randint(18, 80, n).astype(float),
        "salary": rng.normal(50000, 15000, n),
        "city": rng.choice(["NYC", "LA", "Chicago", "Houston", None], n),
        "department": rng.choice(
            [f"dept_{i}" for i in range(25)], n
        ),  # high cardinality
        "score": rng.uniform(0, 100, n),
        "target": rng.choice([0, 1], n),
    })
    # Inject missing values
    df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "salary"] = np.nan
    # Inject outliers
    df.loc[0, "salary"] = 1_000_000
    df.loc[1, "salary"] = -50_000
    return df
