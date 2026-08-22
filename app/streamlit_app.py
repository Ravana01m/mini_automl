"""Mini AutoML — professional Streamlit platform."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

repo_root = str(Path(__file__).parent.parent)
src_path = str(Path(__file__).parent.parent / "src")
for p in (repo_root, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import streamlit as st

from automl.config import (
    EncoderType,
    ExperimentConfig,
    FeatureSelectionStrategy,
    ImbalanceMethod,
    OutlierMethod,
    OutlierStrategy,
    ScalerType,
    TuningMode,
)
from automl.diagnostics import classification_diagnostics, regression_diagnostics
from automl.pipeline_builder import AutoMLPipeline
from automl.utils import setup_logging
from app.components.download import render_download_tab
from app.components.eda_display import render_eda_tab
from app.components.leaderboard import render_leaderboard_tab
from app.components.metrics_display import render_metrics_panel
from app.components.shap_display import render_shap_tab
from app.components.uploader import render_upload_section

setup_logging()

PIPELINE_STEPS = [
    "Data",
    "Validation",
    "EDA",
    "Preprocessing",
    "Feature Engineering",
    "Feature Selection",
    "Models",
    "Tuning",
    "Evaluation",
    "Best Model",
]


def load_custom_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def _sidebar_config() -> ExperimentConfig:
    st.title("Mini AutoML")
    st.caption("Leakage-safe automated ML for classification and regression.")
    st.markdown("---")
    seed = st.number_input("Random seed", 0, 10_000, 42)
    cv = st.slider("CV folds", 3, 8, 5)
    tuning = st.selectbox("Tuning mode", [m.value for m in TuningMode], index=1)
    trials = st.slider("Optuna trials", 5, 80, 15)
    outlier_method = st.selectbox("Outlier method", [m.value for m in OutlierMethod])
    outlier_strategy = st.selectbox("Outlier strategy", [s.value for s in OutlierStrategy], index=0)
    scaler = st.selectbox("Scaler", [s.value for s in ScalerType], index=len(list(ScalerType)) - 1)
    encoder = st.selectbox("Encoder", [e.value for e in EncoderType])
    st.markdown("**Feature engineering**")
    fe = st.checkbox("Enable feature engineering", True)
    log_tf = st.checkbox("Log1p for skewed numerics", True)
    poly = st.checkbox("Polynomial features (restricted)", False)
    interact = st.checkbox("Interaction features", False)
    ratios = st.checkbox("Ratio features", False)
    fs = st.selectbox("Feature selection", [s.value for s in FeatureSelectionStrategy], index=1)
    imbalance = st.selectbox("Imbalance handling", [m.value for m in ImbalanceMethod], index=1)
    ensemble = st.checkbox("Optional ensemble", False)
    skip_ann = st.checkbox("Skip neural network", value=True)
    st.markdown("---")
    st.caption("One failed model never stops the run. All learned transforms refit inside CV.")
    return ExperimentConfig(
        random_state=int(seed),
        cv_folds=int(cv),
        tuning_mode=TuningMode(tuning),
        optuna_trials=int(trials),
        outlier_method=OutlierMethod(outlier_method),
        outlier_strategy=OutlierStrategy(outlier_strategy),
        scaler=ScalerType(scaler),
        encoder=EncoderType(encoder),
        enable_feature_engineering=fe,
        enable_log_transform=log_tf,
        enable_polynomial=poly,
        enable_interactions=interact,
        enable_ratios=ratios,
        feature_selection=FeatureSelectionStrategy(fs),
        imbalance_method=ImbalanceMethod(imbalance),
        enable_ensemble=ensemble,
        skip_ann=skip_ann,
    )


def _pipeline_diagram() -> None:
    st.markdown(" → ".join(f"`{step}`" for step in PIPELINE_STEPS))


def main() -> None:
    st.set_page_config(
        page_title="Mini AutoML Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_custom_css()
    with st.sidebar:
        config = _sidebar_config()

    st.title("Mini AutoML Platform")
    st.markdown(
        "Upload a CSV and run a **leakage-safe** pipeline: validation, profiling, "
        "family-aware preprocessing, feature engineering, feature selection, "
        "staged model search, evaluation, and a downloadable end-to-end model."
    )
    _pipeline_diagram()

    if "results" not in st.session_state:
        st.session_state.results = None
    if "pipeline_obj" not in st.session_state:
        st.session_state.pipeline_obj = None

    tabs = st.tabs(
        [
            "1. Dataset",
            "2. Data Quality",
            "3. EDA",
            "4. Preprocessing",
            "5. Feature Engineering",
            "6. Feature Selection",
            "7. Model Training",
            "8. Leaderboard",
            "9. Evaluation",
            "10. Explainability",
            "11. Final Pipeline",
            "12. Download",
        ]
    )

    with tabs[0]:
        df, target_col, task_type = render_upload_section()
        if df is not None and target_col is not None and task_type is not None:
            st.markdown("---")
            if st.button("Run AutoML Pipeline", type="primary", width="stretch"):
                _run_pipeline(df, target_col, task_type, config)

    results = st.session_state.results
    with tabs[1]:
        if results and results.get("profile"):
            profile = results["profile"]
            st.subheader("Data quality profile")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", profile.n_rows)
            c2.metric("Columns", profile.n_cols)
            c3.metric("Missing %", f"{profile.missing_pct:.2f}")
            c4.metric("Duplicates", profile.duplicate_rows)
            st.dataframe(profile.as_frame(), width="stretch")
            if profile.recommendations:
                st.info("\n".join(f"- {r}" for r in profile.recommendations))
            if results.get("imbalance"):
                st.subheader("Class imbalance")
                st.json(results["imbalance"])
        else:
            st.info("Upload a dataset and run AutoML to see the quality profile.")

    with tabs[2]:
        if results and results.get("eda_report"):
            render_eda_tab(results["eda_report"])
        else:
            st.info("EDA appears after the pipeline runs.")

    with tabs[3]:
        if results:
            st.subheader("Preprocessing strategy")
            st.write(
                "Linear / SVM / ANN models receive imputation, encoding, and scaling. "
                "Tree and boosting models receive imputation and encoding, generally without scaling."
            )
            st.json(results.get("config", {}))
        else:
            st.info("Run the pipeline to inspect the chosen preprocessing.")

    with tabs[4]:
        if results:
            cfg = results.get("config", {})
            st.write(
                {
                    "feature_engineering": cfg.get("enable_feature_engineering"),
                    "log1p": cfg.get("enable_log_transform"),
                    "polynomial": cfg.get("enable_polynomial"),
                    "interactions": cfg.get("enable_interactions"),
                    "ratios": cfg.get("enable_ratios"),
                    "datetime": cfg.get("enable_datetime_features"),
                }
            )
        else:
            st.info("Feature engineering settings will appear after a run.")

    with tabs[5]:
        if results and results.get("feature_report"):
            report = results["feature_report"]
            st.metric("Selection method", report.get("method", "n/a"))
            st.write("Selected features", report.get("selected_features"))
            st.write("Removed features", report.get("removed_features"))
            st.caption("Feature selection is refit independently inside each CV fold.")
        else:
            st.info("Feature selection report appears after training.")

    with tabs[6]:
        if results:
            st.success(f"Best model: **{results.get('best_model_name')}**")
            if results.get("narrative"):
                st.markdown(results["narrative"])
        else:
            st.info("Start training from the Dataset tab.")

    with tabs[7]:
        if results:
            if results.get("best_model_name") and results.get("leaderboard") is not None:
                best_row = results["leaderboard"][
                    results["leaderboard"]["model"] == results["best_model_name"]
                ]
                if not best_row.empty:
                    metric_cols = [
                        c
                        for c in best_row.columns
                        if c
                        not in (
                            "model",
                            "status",
                            "error",
                            "grid_best_params",
                            "optuna_best_params",
                            "family",
                        )
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
            render_leaderboard_tab(
                leaderboard=results.get("leaderboard", pd.DataFrame()),
                task_type=results.get("task_type", "classification"),
                cv_results=results.get("cv_results"),
                best_model_name=results.get("best_model_name"),
                fitted_models=results.get("fitted_models"),
                X_test=results.get("X_test"),
                y_test=results.get("y_test"),
            )
            if results.get("comparison"):
                st.subheader("Baseline vs advanced")
                st.json(results["comparison"])
                st.caption("If advanced loses, that is reported honestly.")
        else:
            st.info("Leaderboard appears after training.")

    with tabs[8]:
        if results and results.get("pipeline") is not None:
            _render_evaluation(results)
        else:
            st.info("Evaluation charts appear after training.")

    with tabs[9]:
        if results and results.get("explainer"):
            render_shap_tab(results["explainer"])
        else:
            st.info("SHAP appears after training. A SHAP failure never crashes AutoML.")

    with tabs[10]:
        if results and results.get("model_card"):
            st.subheader("Model card")
            st.json(results["model_card"])
            st.subheader("Inference")
            st.code(
                "from automl.serialization import load_model\n"
                "model = load_model('best_model.joblib')\n"
                "preds = model.predict(raw_dataframe)\n",
                language="python",
            )
        else:
            st.info("The final pipeline card appears after a successful run.")

    with tabs[11]:
        if results and results.get("pipeline"):
            render_download_tab(results["pipeline"], results.get("best_model_name"))
        else:
            st.info("Download becomes available after training.")


def _render_evaluation(results: dict) -> None:
    model = results["pipeline"]
    X_test, y_test = results.get("X_test"), results.get("y_test")
    if X_test is None:
        st.warning("No held-out test split available.")
        return
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    except Exception as exc:
        st.error(f"Could not score the test set: {exc}")
        return
    if results["task_type"] == "regression":
        diag = regression_diagnostics(y_test, y_pred)
        st.json(diag["metrics"])
        st.pyplot(diag["figure"])
        for note in diag["notes"]:
            st.warning(note)
    else:
        diag = classification_diagnostics(y_test, y_pred, y_proba)
        st.json(diag["metrics"])
        cols = st.columns(2)
        cols[0].pyplot(diag["confusion_figure"])
        if diag["roc_figure"] is not None:
            cols[1].pyplot(diag["roc_figure"])
        if diag["pr_figure"] is not None:
            st.pyplot(diag["pr_figure"])
        st.dataframe(diag["report"], width="stretch")
        for note in diag["notes"]:
            st.warning(note)
    if results.get("test_metrics"):
        st.subheader("Held-out test metrics")
        st.json(results["test_metrics"])


def _run_pipeline(df: pd.DataFrame, target_col: str, task_type: str, config: ExperimentConfig) -> None:
    progress_bar = st.progress(0)
    status_text = st.empty()
    stages = {
        "validation": 8,
        "detection": 16,
        "eda": 28,
        "preprocessing": 40,
        "training": 70,
        "tuning": 82,
        "explainability": 92,
        "finalizing": 97,
    }

    def progress_callback(stage: str, detail: str) -> None:
        progress_bar.progress(stages.get(stage, 50) / 100)
        status_text.markdown(f"**{detail}**")

    try:
        pipeline = AutoMLPipeline(config=config)
        results = pipeline.run(
            df=df,
            target_col=target_col,
            task_type_override=task_type,
            progress_callback=progress_callback,
        )
        progress_bar.progress(1.0)
        status_text.markdown("**Pipeline complete**")
        st.session_state.results = results
        st.session_state.pipeline_obj = pipeline
        st.success(f"Best model: **{results['best_model_name']}** ({results['task_type']})")
    except Exception as exc:
        progress_bar.progress(0)
        status_text.empty()
        st.error(f"Pipeline failed: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
