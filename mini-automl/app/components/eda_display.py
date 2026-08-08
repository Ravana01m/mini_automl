"""EDA dashboard component — renders automated EDA visualizations."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_eda_tab(eda_report: dict[str, Any] | None) -> None:
    """Render the EDA dashboard tab."""
    if eda_report is None:
        st.warning("EDA report not available.")
        return
    
    st.header("📊 Exploratory Data Analysis")
    
    # Summary statistics
    summary = eda_report.get("summary", {})
    if summary:
        st.subheader("Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        shape = summary.get("shape", (0, 0))
        col1.metric("Rows", f"{shape[0]:,}")
        col2.metric("Columns", f"{shape[1]:,}")
        col3.metric("Duplicates", f"{summary.get('n_duplicates', 0):,}")
        col4.metric("Memory", summary.get("memory_usage", "N/A"))
    
    # Target distribution
    target_fig = eda_report.get("target_fig")
    if target_fig is not None:
        st.subheader("Target Variable Distribution")
        st.pyplot(target_fig)
    
    # Missing values
    missing_fig = eda_report.get("missing_fig")
    if missing_fig is not None:
        st.subheader("Missing Values")
        st.pyplot(missing_fig)
    
    # Correlation heatmap
    corr_fig = eda_report.get("correlation_fig")
    if corr_fig is not None:
        st.subheader("Feature Correlations")
        st.pyplot(corr_fig)
    
    # Numeric feature distributions
    numeric_figs = eda_report.get("numeric_figs", [])
    if numeric_figs:
        st.subheader("Numeric Feature Distributions")
        for fig in numeric_figs:
            st.pyplot(fig)
    
    # Categorical feature distributions
    cat_figs = eda_report.get("categorical_figs", [])
    if cat_figs:
        st.subheader("Categorical Feature Distributions")
        for fig in cat_figs:
            st.pyplot(fig)
    
    # Descriptive stats table
    stats = eda_report.get("stats")
    if stats is not None:
        st.subheader("Descriptive Statistics")
        st.dataframe(stats, width="stretch")
