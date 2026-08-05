"""SHAP explainability display component."""

from __future__ import annotations

import streamlit as st


def render_shap_tab(explainer: object | None) -> None:
    """Render SHAP summary and per-prediction explanations."""
    st.header("🔬 Model Explainability (SHAP)")
    
    if explainer is None:
        st.warning("SHAP explanations not available for this model.")
        return
    
    if not hasattr(explainer, 'shap_values_') or explainer.shap_values_ is None:
        st.warning("SHAP values have not been computed yet.")
        return
    
    # Summary plot
    st.subheader("🌍 Global Feature Importance")
    st.caption("Shows how each feature impacts the model's predictions across all samples.")
    try:
        summary_fig = explainer.summary_plot()
        st.pyplot(summary_fig)
    except Exception as e:
        st.warning(f"Could not generate summary plot: {e}")
    
    # Per-prediction explanation
    st.markdown("---")
    st.subheader("🔍 Per-Prediction Explanation")
    st.caption("Select a sample to see how each feature contributed to its prediction.")
    
    max_idx = explainer.shap_values_.shape[0] - 1
    sample_idx = st.slider(
        "Select sample index:",
        min_value=0,
        max_value=max_idx,
        value=0,
        step=1,
    )
    
    try:
        waterfall_fig = explainer.waterfall_plot(idx=sample_idx)
        st.pyplot(waterfall_fig)
    except Exception as e:
        st.warning(f"Could not generate waterfall plot: {e}")
