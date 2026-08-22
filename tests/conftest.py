"""Shared pytest fixtures. All datasets are synthetic — no network required."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from automl.datasets import (
    make_binary,
    make_categorical,
    make_imbalanced,
    make_missing,
    make_multiclass,
    make_outliers,
    make_regression,
)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def iris_df() -> pd.DataFrame:
    return make_multiclass(n=150, seed=0).rename(columns={"f1": "sepal_length", "f2": "sepal_width"})


@pytest.fixture
def housing_df() -> pd.DataFrame:
    return make_regression(n=160, seed=1)


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return make_missing(n=180, seed=2)


@pytest.fixture
def binary_df() -> pd.DataFrame:
    return make_binary(n=160, seed=3)


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    return make_imbalanced(n=200, seed=4)


@pytest.fixture
def outlier_df() -> pd.DataFrame:
    return make_outliers(n=160, seed=5)


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    return make_categorical(n=160, seed=6)
