"""Best model metrics display component."""

from __future__ import annotations

import streamlit as st


def render_metrics_panel(
    metrics: dict[str, float],
    model_name: str,
    task_type: str,
) -> None:
    """Render metrics cards for the best model."""
    st.subheader(f"🏅 Best Model: {model_name}")
    
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        display_name = name.replace("_mean", "").replace("_", " ").title()
        if isinstance(value, float):
            col.metric(display_name, f"{value:.4f}")
        else:
            col.metric(display_name, str(value))
