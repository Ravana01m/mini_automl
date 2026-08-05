"""Model leaderboard component."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.components.charts import (
    metric_bar_chart,
    radar_chart,
    cv_box_plot,
    training_time_chart,
    confusion_matrix_chart,
    actual_vs_predicted_chart,
)


def render_leaderboard_tab(
    leaderboard: pd.DataFrame,
    task_type: str,
    cv_results: dict[str, list[float]] | None = None,
    best_model_name: str | None = None,
    fitted_models: dict | None = None,
    X_test: Any = None,
    y_test: Any = None,
) -> None:
    """Render the leaderboard table and comparison charts."""
    st.header("🏆 Model Leaderboard")
    
    if leaderboard.empty:
        st.warning("No model results available yet.")
        return
    
    # Display columns to show (hide internal ones)
    display_cols = [c for c in leaderboard.columns if c not in (
        "status", "error", "grid_best_params", "optuna_best_params"
    )]
    
    # Highlight best model
    st.dataframe(
        leaderboard[display_cols].style.highlight_max(
            subset=[c for c in display_cols if c not in ("model", "train_time_s", "rmse_mean", "rmse_std")],
            color="#667eea40",
        ),
        use_container_width=True,
        height=min(400, 40 * len(leaderboard) + 60),
    )
    
    if best_model_name:
        st.success(f"🥇 **Best Model: {best_model_name}**")
    
    st.markdown("---")
    
    # Charts
    st.subheader("📊 Model Comparison Charts")
    
    # Metric bar chart
    if task_type == "classification":
        metrics = ["accuracy_mean", "f1_weighted_mean"]
        primary_metric_name = "F1 Weighted"
    else:
        metrics = ["r2_mean", "rmse_mean"]
        primary_metric_name = "R²"
    
    available_metrics = [m for m in metrics if m in leaderboard.columns]
    if available_metrics:
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = metric_bar_chart(leaderboard, available_metrics, best_model_name or "")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_radar = radar_chart(leaderboard, available_metrics, top_n=3)
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # CV box plot and training time
    col1, col2 = st.columns(2)
    with col1:
        if cv_results:
            fig_cv = cv_box_plot(cv_results, primary_metric_name)
            st.plotly_chart(fig_cv, use_container_width=True)
    with col2:
        if "train_time_s" in leaderboard.columns:
            fig_time = training_time_chart(leaderboard)
            st.plotly_chart(fig_time, use_container_width=True)
    
    # Confusion matrix or actual vs predicted for best model
    if fitted_models and best_model_name and y_test is not None and X_test is not None:
        import numpy as np
        st.subheader("🔍 Best Model Deep Dive")
        best_model = fitted_models.get(best_model_name)
        if best_model:
            try:
                y_pred = best_model.predict(X_test)
                if task_type == "classification":
                    labels = [str(c) for c in sorted(set(y_test))]
                    fig_cm = confusion_matrix_chart(y_test, y_pred, labels)
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    from sklearn.metrics import r2_score
                    r2 = r2_score(y_test, y_pred)
                    fig_scatter = actual_vs_predicted_chart(
                        np.array(y_test), np.array(y_pred), r2
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate best model charts: {e}")
