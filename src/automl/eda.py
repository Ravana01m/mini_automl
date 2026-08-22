"""Professional EDA report generation with intelligent sampling."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from automl.correlation import correlation_analysis
from automl.profiling import profile_dataframe
from automl.utils import get_column_types

sns.set_style("whitegrid")


def _maybe_sample(df: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed)


def _plot_target_distribution(target: pd.Series, task_type: str) -> plt.Figure:
    fig = plt.figure(figsize=(8, 5))
    if task_type == "classification":
        sns.countplot(x=target)
        plt.title("Target Distribution (Classification)")
    else:
        sns.histplot(target, kde=True)
        plt.title("Target Distribution (Regression)")
    plt.tight_layout()
    return fig


def _plot_correlation_heatmap(df: pd.DataFrame, target_col: str) -> plt.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    fig = plt.figure(figsize=(10, 8))
    if numeric_df.shape[1] > 0:
        if numeric_df.shape[1] > 25:
            numeric_df = numeric_df.iloc[:, :25]
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        annot = numeric_df.shape[1] <= 15
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            annot=annot,
            fmt=".2f",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
        )
        plt.title("Numeric Feature Correlation Heatmap (EDA only)")
    else:
        plt.text(0.5, 0.5, "No numeric features available", ha="center", va="center")
        plt.title("Correlation Heatmap")
    plt.tight_layout()
    return fig


def _plot_missing_values(df: pd.DataFrame) -> plt.Figure | None:
    if df.isnull().sum().sum() == 0:
        return None
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    return fig


def _univariate_numeric(df: pd.DataFrame, col: str) -> plt.Figure:
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Histogram + KDE: {col}")
    plt.subplot(1, 3, 2)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot: {col}")
    plt.subplot(1, 3, 3)
    sns.violinplot(x=df[col])
    plt.title(f"Violin: {col}")
    plt.tight_layout()
    return fig


def generate_eda_report(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    max_samples: int = 1500,
) -> dict[str, Any]:
    """Generate a comprehensive EDA report. Charts use sampled data for large frames."""
    profile = profile_dataframe(df, target_col)
    viz_df = _maybe_sample(df, max_samples)
    numeric_cols, categorical_cols = get_column_types(df)
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()

    summary = {
        "shape": df.shape,
        "dtypes": {str(k): v for k, v in df.dtypes.value_counts().to_dict().items()},
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "n_duplicates": int(df.duplicated().sum()),
        "missing_pct": missing_pct,
        "profile": profile,
    }

    target_fig = _plot_target_distribution(viz_df[target_col], task_type)
    numeric_figs = []
    for col in numeric_cols[:10]:
        if col == target_col:
            continue
        numeric_figs.append(_univariate_numeric(viz_df, col))

    categorical_figs = []
    for col in categorical_cols[:8]:
        if col == target_col:
            continue
        fig = plt.figure(figsize=(8, 5))
        top_10 = viz_df[col].astype(str).value_counts().nlargest(10)
        sns.barplot(x=top_10.index, y=top_10.values)
        plt.title(f"Top 10 Values of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        categorical_figs.append(fig)

    pairplot_fig = None
    pair_cols = [c for c in numeric_cols if c != target_col][:5]
    if 2 <= len(pair_cols) <= 5 and len(viz_df) <= 800:
        try:
            grid = sns.pairplot(viz_df[pair_cols + ([target_col] if target_col in viz_df else [])].dropna())
            pairplot_fig = grid.fig
        except Exception:
            pairplot_fig = None

    return {
        "summary": summary,
        "profile": profile,
        "target_fig": target_fig,
        "numeric_figs": numeric_figs,
        "categorical_figs": categorical_figs,
        "correlation_fig": _plot_correlation_heatmap(viz_df, target_col),
        "missing_fig": _plot_missing_values(viz_df),
        "pairplot_fig": pairplot_fig,
        "correlation_tables": correlation_analysis(df, target_col),
        "stats": df.describe(include="all").T,
    }
