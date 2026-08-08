from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def _plot_target_distribution(target: pd.Series, task_type: str) -> plt.Figure:
    fig = plt.figure(figsize=(8, 5))
    if task_type == 'classification':
        sns.countplot(x=target)
        plt.title('Target Distribution (Classification)')
    else:
        sns.histplot(target, kde=True)
        plt.title('Target Distribution (Regression)')
    plt.tight_layout()
    return fig

def _plot_correlation_heatmap(df: pd.DataFrame, target_col: str) -> plt.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    fig = plt.figure(figsize=(10, 8))
    if numeric_df.shape[1] > 0:
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        annot = numeric_df.shape[1] <= 15
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=annot, fmt=".2f",
                    vmin=-1, vmax=1, square=True, linewidths=.5)
        plt.title('Numeric Feature Correlation Heatmap')
    else:
        plt.text(0.5, 0.5, "No numeric features available", ha='center', va='center')
        plt.title('Correlation Heatmap')
    plt.tight_layout()
    return fig

def _plot_missing_values(df: pd.DataFrame) -> plt.Figure | None:
    if df.isnull().sum().sum() == 0:
        return None
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    return fig

def generate_eda_report(df: pd.DataFrame, target_col: str, task_type: str) -> dict[str, Any]:
    """Generates a comprehensive Exploratory Data Analysis report."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
    
    summary = {
        'shape': df.shape,
        'dtypes': {str(k): v for k, v in df.dtypes.value_counts().to_dict().items()},
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'n_duplicates': int(df.duplicated().sum()),
        'missing_pct': missing_pct
    }
    
    target_fig = _plot_target_distribution(df[target_col], task_type)
    
    numeric_figs = []
    for col in numeric_cols[:12]:
        if col == target_col:
            continue
        fig = plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        numeric_figs.append(fig)
        
    categorical_figs = []
    for col in categorical_cols[:8]:
        if col == target_col:
            continue
        fig = plt.figure(figsize=(8, 5))
        top_10 = df[col].value_counts().nlargest(10)
        sns.barplot(x=top_10.index, y=top_10.values)
        plt.title(f'Top 10 Values of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        categorical_figs.append(fig)
        
    correlation_fig = _plot_correlation_heatmap(df, target_col)
    missing_fig = _plot_missing_values(df)
    
    stats = df.describe(include='all').T
    
    return {
        'summary': summary,
        'target_fig': target_fig,
        'numeric_figs': numeric_figs,
        'categorical_figs': categorical_figs,
        'correlation_fig': correlation_fig,
        'missing_fig': missing_fig,
        'stats': stats
    }
