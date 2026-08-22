"""Small synthetic datasets for tests and demos. No external downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _rng(seed: int = 42) -> np.random.RandomState:
    return np.random.RandomState(seed)


def make_regression(n: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = _rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(2, 1.4, n)
    x3 = rng.uniform(0, 5, n)
    noise = rng.normal(0, 0.6, n)
    y = 3.2 * x1 - 1.1 * x2 + 0.4 * x3 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "group": rng.choice(["A", "B", "C"], n), "target": y})


def make_binary(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = _rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logits = 1.4 * x1 - 0.9 * x2 + rng.normal(0, 0.3, n)
    y = (logits > 0).astype(int)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cat": rng.choice(["red", "blue", "green"], n),
            "target": y,
        }
    )


def make_multiclass(n: int = 210, seed: int = 42) -> pd.DataFrame:
    rng = _rng(seed)
    centers = np.array([[0, 0], [3, 3], [-2, 2]])
    X = []
    y = []
    for i, center in enumerate(centers):
        X.append(rng.normal(center, 0.8, size=(n // 3, 2)))
        y.append(np.full(n // 3, i))
    X = np.vstack(X)
    y = np.concatenate(y)
    return pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "color": rng.choice(["p", "q"], len(y)), "target": y})


def make_missing(n: int = 160, seed: int = 42) -> pd.DataFrame:
    df = make_regression(n, seed)
    rng = _rng(seed + 1)
    df.loc[rng.choice(n, 25, replace=False), "x1"] = np.nan
    df.loc[rng.choice(n, 18, replace=False), "group"] = None
    return df


def make_outliers(n: int = 160, seed: int = 42) -> pd.DataFrame:
    df = make_regression(n, seed)
    df.loc[0, "x1"] = 25
    df.loc[1, "x2"] = -30
    df.loc[2, "x3"] = 80
    return df


def make_categorical(n: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = _rng(seed)
    city = rng.choice(["NYC", "LA", "CHI", "HOU", "MIA"], n)
    plan = rng.choice(["basic", "plus", "pro"], n)
    score = rng.normal(50, 10, n) + np.where(plan == "pro", 8, 0)
    y = (score + np.where(city == "NYC", 3, 0) > 52).astype(int)
    return pd.DataFrame({"city": city, "plan": plan, "score": score, "target": y})


def make_imbalanced(n: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = _rng(seed)
    n_pos = max(12, int(n * 0.08))
    n_neg = n - n_pos
    x_neg = rng.normal(0, 1, size=(n_neg, 2))
    x_pos = rng.normal(2.2, 0.9, size=(n_pos, 2))
    X = np.vstack([x_neg, x_pos])
    y = np.array([0] * n_neg + [1] * n_pos)
    idx = rng.permutation(n)
    return pd.DataFrame({"a": X[idx, 0], "b": X[idx, 1], "target": y[idx]})


def write_examples(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mapping = {
        "regression.csv": make_regression(),
        "binary.csv": make_binary(),
        "multiclass.csv": make_multiclass(),
        "missing.csv": make_missing(),
        "outliers.csv": make_outliers(),
        "categorical.csv": make_categorical(),
        "imbalanced.csv": make_imbalanced(),
    }
    paths = []
    for name, frame in mapping.items():
        path = directory / name
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths
