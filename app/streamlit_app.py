"""Mini AutoML Pipeline — Streamlit web application.

Main entry point. Provides tabs for:
1. Upload & Configure
2. EDA Dashboard
3. Training & Results
4. Leaderboard & Charts
5. Explainability
6. Download
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add project root to path so imports work
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from automl.pipeline_builder import AutoMLPipeline
from automl.utils import setup_logging
from app.components.uploader import render_upload_section
from app.components.eda_display import render_eda_tab
from app.components.leaderboard import render_leaderboard_tab
from app.components.metrics_display import render_metrics_panel
from app.components.shap_display import render_shap_tab
from app.components.download import render_download_tab

setup_logging()


def load_custom_css() -> None:
    """Load custom CSS styling."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Mini AutoML Pipeline",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    load_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Mini AutoML")
        st.markdown("---")
        st.markdown(
            "Upload any CSV and get automated ML model development: "
            "preprocessing, feature engineering, model comparison, "
            "tuning, explainability, and model download."
        )
        st.markdown("---")
        
        skip_optuna = st.checkbox(
            "⚡ Quick Mode (skip Optuna tuning)",
            value=False,
            help="Skip Optuna hyperparameter tuning for faster results.",
        )
        
        st.markdown("---")
        st.markdown(
            "Built with ❤️ using scikit-learn, XGBoost, LightGBM, "
            "TensorFlow, Optuna, SHAP, and Streamlit."
        )
    
    # Main content
    st.title("🤖 Mini AutoML Pipeline")
    st.markdown(
        "Upload any CSV → automated preprocessing, feature engineering, "
        "model training, tuning, explainability, and model download."
    )
    
    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "pipeline_obj" not in st.session_state:
        st.session_state.pipeline_obj = None
    
    # Tab 1: Upload
    tab_upload, tab_eda, tab_results, tab_shap, tab_download = st.tabs([
        "📁 Upload & Configure",
        "📊 EDA Dashboard",
        "🏆 Leaderboard & Charts",
        "🔬 Explainability",
        "📥 Download Model",
    ])
    
    with tab_upload:
        df, target_col, task_type = render_upload_section()
        
        if df is not None and target_col is not None and task_type is not None:
            st.markdown("---")
            if st.button("🚀 Run AutoML Pipeline", type="primary", use_container_width=True):
                _run_pipeline(df, target_col, task_type, skip_optuna)
    
    with tab_eda:
        if st.session_state.results and st.session_state.results.get("eda_report"):
            render_eda_tab(st.session_state.results["eda_report"])
        else:
            st.info("📁 Upload a dataset and run the pipeline to see EDA results.")
    
    with tab_results:
        if st.session_state.results:
            results = st.session_state.results
            
            # Metrics panel
            if results.get("best_model_name") and results.get("leaderboard") is not None:
                best_row = results["leaderboard"][
                    results["leaderboard"]["model"] == results["best_model_name"]
                ]
                if not best_row.empty:
                    metric_cols = [
                        c for c in best_row.columns
                        if c not in ("model", "status", "error", "grid_best_params", "optuna_best_params")
                    ]
                    metrics_dict = {
                        c: best_row.iloc[0][c]
                        for c in metric_cols
                        if pd.notna(best_row.iloc[0][c])
                    }
                    render_metrics_panel(
                        metrics_dict,
                        results["best_model_name"],
                        results["task_type"],
                    )
            
            st.markdown("---")
            
            # Leaderboard with charts
            render_leaderboard_tab(
                leaderboard=results.get("leaderboard", pd.DataFrame()),
                task_type=results.get("task_type", "classification"),
                cv_results=results.get("cv_results"),
                best_model_name=results.get("best_model_name"),
                fitted_models=results.get("fitted_models"),
                X_test=results.get("X_test"),
                y_test=results.get("y_test"),
            )
        else:
            st.info("📁 Upload a dataset and run the pipeline to see results.")
    
    with tab_shap:
        if st.session_state.results and st.session_state.results.get("explainer"):
            render_shap_tab(st.session_state.results["explainer"])
        else:
            st.info("📁 Upload a dataset and run the pipeline to see SHAP explanations.")
    
    with tab_download:
        if st.session_state.results and st.session_state.results.get("pipeline"):
            render_download_tab(
                st.session_state.results["pipeline"],
                st.session_state.results.get("best_model_name"),
            )
        else:
            st.info("📁 Upload a dataset and run the pipeline to download the model.")


def _run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    skip_optuna: bool,
) -> None:
    """Run the AutoML pipeline with progress tracking."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stages = {
        "validation": 5,
        "detection": 10,
        "eda": 20,
        "preprocessing": 35,
        "training": 55,
        "tuning": 75,
        "explainability": 90,
        "finalizing": 95,
    }
    
    def progress_callback(stage: str, detail: str) -> None:
        pct = stages.get(stage, 50)
        progress_bar.progress(pct / 100)
        status_text.markdown(f"**{detail}**")
    
    try:
        pipeline = AutoMLPipeline(random_state=42)
        
        # Split for evaluation
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        results = pipeline.run(
            df=df,
            target_col=target_col,
            task_type_override=task_type,
            progress_callback=progress_callback,
            skip_optuna=skip_optuna,
        )
        
        # Store results in session state
        progress_bar.progress(1.0)
        status_text.markdown("**✅ Pipeline complete!**")
        
        st.session_state.results = results
        st.session_state.pipeline_obj = pipeline
        
        st.success(
            f"🎉 Pipeline complete! Best model: **{results['best_model_name']}** "
            f"({results['task_type']})"
        )
        st.balloons()
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        st.error(f"❌ Pipeline failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
