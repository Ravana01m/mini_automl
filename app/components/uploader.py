"""CSV upload and target column selection component."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from automl.detector import detect_task_type
from automl.utils import validate_csv, validate_target_column, ValidationError


def render_upload_section() -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Render the CSV upload and configuration UI.

    Returns:
        Tuple of (DataFrame or None, target_column or None, task_type or None).
    """
    st.header("📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload any CSV file. Max 200 MB.",
    )
    
    if uploaded_file is None:
        st.info("👆 Upload a CSV file to get started.")
        return None, None, None
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 200:
        st.error(f"❌ File too large ({file_size_mb:.1f} MB). Maximum is 200 MB.")
        return None, None, None
    
    # Load CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return None, None, None
    
    # Validate
    try:
        validate_csv(df, uploaded_file.name)
    except ValidationError as e:
        st.error(f"❌ {e}")
        return None, None, None
    
    # Display preview
    st.success(f"✅ Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    with st.expander("📊 Data Preview", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**First 5 rows:**")
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.markdown("**Column Info:**")
            info_df = pd.DataFrame({
                "Type": df.dtypes.astype(str),
                "Non-Null": df.count(),
                "Null %": (df.isnull().sum() / len(df) * 100).round(1),
                "Unique": df.nunique(),
            })
            st.dataframe(info_df, use_container_width=True)
    
    # Target column selection
    st.subheader("🎯 Select Target Column")
    target_col = st.selectbox(
        "Choose the column to predict:",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,  # Default to last column
    )
    
    # Validate target
    try:
        validate_target_column(df, target_col)
    except ValidationError as e:
        st.error(f"❌ {e}")
        return df, None, None
    
    # Detect task type
    auto_task_type = detect_task_type(df[target_col])
    
    task_type = st.radio(
        "Task Type:",
        options=["classification", "regression"],
        index=0 if auto_task_type == "classification" else 1,
        horizontal=True,
        help=f"Auto-detected: **{auto_task_type}**. Override if needed.",
    )
    
    # Show target summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Task Type", task_type.title())
    col2.metric("Target Unique Values", df[target_col].nunique())
    col3.metric("Target Null %", f"{df[target_col].isnull().mean() * 100:.1f}%")
    
    return df, target_col, task_type
