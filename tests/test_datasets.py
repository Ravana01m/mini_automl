"""Synthetic example datasets used by tests and the demo."""

from __future__ import annotations

from automl.datasets import (
    make_binary,
    make_categorical,
    make_imbalanced,
    make_missing,
    make_multiclass,
    make_outliers,
    make_regression,
    write_examples,
)


def test_synthetic_shapes() -> None:
    assert make_regression().shape[0] > 50
    assert set(make_binary()["target"].unique()) <= {0, 1}
    assert make_multiclass()["target"].nunique() == 3
    assert make_missing().isna().any().any()
    assert make_outliers()["x1"].max() > 10
    assert make_categorical()["city"].nunique() >= 3
    assert make_imbalanced()["target"].value_counts().min() < 40


def test_write_examples(tmp_path) -> None:
    paths = write_examples(tmp_path)
    assert len(paths) == 7
    assert all(p.exists() for p in paths)
