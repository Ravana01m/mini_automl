"""Plotly charts for model comparison.

All charts are interactive (hover, zoom, pan) via Plotly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

# Color palette
COLORS = [
    "#667eea", "#764ba2", "#f093fb", "#f5576c",
    "#4facfe", "#00f2fe", "#43e97b", "#fa709a",
]


def metric_bar_chart(
    leaderboard: pd.DataFrame,
    metrics: list[str],
    best_model: str,
) -> go.Figure:
    """Grouped bar chart comparing models across metrics."""
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        if metric not in leaderboard.columns:
            continue
        
        colors = [
            "#FFD700" if m == best_model else COLORS[i % len(COLORS)]
            for m in leaderboard["model"]
        ]
        
        fig.add_trace(go.Bar(
            name=metric.replace("_", " ").title(),
            x=leaderboard["model"],
            y=leaderboard[metric],
            marker_color=colors,
            text=leaderboard[metric].round(4),
            textposition="auto",
        ))
    
    fig.update_layout(
        title="Model Comparison — Metrics",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode="group",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
    )
    return fig


def radar_chart(
    leaderboard: pd.DataFrame,
    metrics: list[str],
    top_n: int = 3,
) -> go.Figure:
    """Radar/spider chart for multi-metric comparison of top models."""
    top = leaderboard.head(top_n)
    available_metrics = [m for m in metrics if m in top.columns]
    
    if not available_metrics:
        return go.Figure()
    
    fig = go.Figure()
    
    for i, (_, row) in enumerate(top.iterrows()):
        values = []
        for m in available_metrics:
            val = row[m]
            # Normalize: higher is always better for radar chart
            col_vals = leaderboard[m].dropna()
            if col_vals.max() != col_vals.min():
                normalized = (val - col_vals.min()) / (col_vals.max() - col_vals.min())
            else:
                normalized = 1.0
            values.append(round(float(normalized), 3))
        
        # Close the polygon
        values.append(values[0])
        labels = [m.replace("_", " ").title() for m in available_metrics]
        labels.append(labels[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name=row["model"],
            line_color=COLORS[i % len(COLORS)],
            opacity=0.7,
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        title="Top Models — Multi-Metric Radar",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        height=450,
    )
    return fig


def cv_box_plot(
    cv_results: dict[str, list[float]],
    metric_name: str,
) -> go.Figure:
    """Box plot of cross-validation fold scores per model."""
    fig = go.Figure()
    
    for i, (model_name, scores) in enumerate(cv_results.items()):
        fig.add_trace(go.Box(
            y=scores,
            name=model_name,
            marker_color=COLORS[i % len(COLORS)],
            boxmean=True,
        ))
    
    fig.update_layout(
        title=f"Cross-Validation Score Distribution — {metric_name}",
        yaxis_title=metric_name.replace("_", " ").title(),
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        height=400,
    )
    return fig


def training_time_chart(leaderboard: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of training times per model."""
    if "train_time_s" not in leaderboard.columns:
        return go.Figure()
    
    sorted_lb = leaderboard.sort_values("train_time_s", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=sorted_lb["train_time_s"],
        y=sorted_lb["model"],
        orientation="h",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(sorted_lb))],
        text=sorted_lb["train_time_s"].round(1).astype(str) + "s",
        textposition="auto",
    ))
    
    fig.update_layout(
        title="Training Time per Model",
        xaxis_title="Time (seconds)",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        height=350,
    )
    return fig


def confusion_matrix_chart(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    labels: list[str] | None = None,
) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    
    # Calculate percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    # Create text annotations
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)")
    
    text = np.array(annotations).reshape(cm.shape)
    
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Purples",
        showscale=True,
    ))
    
    fig.update_layout(
        title="Confusion Matrix — Best Model",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        height=450,
    )
    return fig


def actual_vs_predicted_chart(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    r2: float,
) -> go.Figure:
    """Scatter plot of actual vs predicted with R-squared annotation."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(color="#667eea", opacity=0.6, size=6),
        name="Predictions",
    ))
    
    # Perfect prediction line
    all_vals = list(y_true) + list(y_pred)
    min_val, max_val = min(all_vals), max(all_vals)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#f5576c", dash="dash", width=2),
        name="Perfect Prediction",
    ))
    
    fig.update_layout(
        title=f"Actual vs Predicted (R² = {r2:.4f})",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        height=450,
    )
    return fig
