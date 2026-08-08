"""Model download component."""

from __future__ import annotations

import io
import os

import joblib
import streamlit as st


def render_download_tab(
    pipeline: object | None,
    model_name: str | None = None,
) -> None:
    """Render the model download button."""
    st.header("📥 Download Best Model")
    
    if pipeline is None:
        st.warning("No trained pipeline available yet. Run the pipeline first.")
        return
    
    st.markdown("""
    The download includes the **complete pipeline**:
    - ✅ Preprocessing (imputation, encoding, scaling)
    - ✅ Feature engineering (log transforms, polynomial features)
    - ✅ Feature selection
    - ✅ Trained model
    
    **Usage after download:**
    ```python
    import joblib
    import pandas as pd
    
    pipeline = joblib.load("best_model_pipeline.joblib")
    new_data = pd.read_csv("new_data.csv")
    predictions = pipeline.predict(new_data)
    ```
    """)
    
    # Serialize to buffer
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    
    filename = f"best_model_{model_name or 'pipeline'}.joblib".replace(" ", "_")
    
    st.download_button(
        label="⬇️ Download Best Model Pipeline (.joblib)",
        data=buffer,
        file_name=filename,
        mime="application/octet-stream",
        type="primary",
    )
    
    st.info(f"📦 Model: **{model_name or 'Unknown'}** | Size: **{buffer.getbuffer().nbytes / 1024:.1f} KB**")
