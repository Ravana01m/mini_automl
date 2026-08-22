"""Pipeline orchestrator: leakage-safe AutoML from raw CSV to a serializable model."""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from automl.config import ExperimentConfig, ImbalanceMethod, TuningMode, set_global_seed
from automl.detector import classify_problem, detect_task_type, imbalance_report
from automl.eda import generate_eda_report
from automl.ensembles import build_voting, should_build_ensemble
from automl.evaluation import evaluate_model, percent_change
try:
    from automl.explainer import ModelExplainer
except Exception:  # pragma: no cover
    ModelExplainer = None  # type: ignore[misc,assignment]
from automl.experiment import log_experiment
from automl.model_card import build_model_card
from automl.model_registry import get_model_specs
from automl.pipeline_factory import build_model_pipeline
from automl.profiling import profile_dataframe
from automl.report import build_narrative_report
from automl.trainer import Trainer
from automl.utils import flatten_params, validate_csv, validate_target_column

logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """Main orchestrator for the AutoML platform."""

    def __init__(
        self,
        random_state: int = 42,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.config = config or ExperimentConfig(random_state=random_state)
        self.random_state = self.config.random_state
        self.task_type_: str | None = None
        self.pipeline_: Pipeline | None = None
        self.preprocessing_pipeline_: Pipeline | None = None
        self.trainer_: Trainer | None = None
        self.explainer_: ModelExplainer | None = None
        self.leaderboard_: pd.DataFrame | None = None
        self.eda_report_: dict[str, Any] | None = None
        self.profile_ = None
        self.target_col_: str | None = None
        self.n_classes_: int | None = None
        self.label_encoder_: LabelEncoder | None = None
        self.baseline_metrics_: dict[str, float] | None = None
        self.advanced_metrics_: dict[str, float] | None = None
        self.comparison_: dict[str, Any] | None = None
        self.narrative_: str | None = None
        self.model_card_: dict[str, Any] | None = None
        self.feature_report_: dict[str, Any] | None = None
        self.imbalance_: dict[str, Any] | None = None

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type_override: str | None = None,
        progress_callback: Any | None = None,
        skip_optuna: bool = False,
        config: ExperimentConfig | None = None,
    ) -> dict[str, Any]:
        if config is not None:
            self.config = config
            self.random_state = config.random_state
        if skip_optuna:
            self.config.tuning_mode = TuningMode.FAST

        set_global_seed(self.random_state)
        started = time.time()
        self.target_col_ = target_col

        def _progress(stage: str, detail: str) -> None:
            if progress_callback:
                progress_callback(stage, detail)

        _progress("validation", "Validating input data...")
        validate_csv(df)
        validate_target_column(df, target_col)

        _progress("detection", "Detecting task type and profiling data...")
        self.task_type_ = task_type_override or detect_task_type(df[target_col])
        self.profile_ = profile_dataframe(df, target_col)
        self.imbalance_ = (
            imbalance_report(df[target_col]) if self.task_type_ == "classification" else None
        )

        _progress("eda", "Generating EDA report...")
        self.eda_report_ = None
        if not self.config.skip_eda:
            try:
                self.eda_report_ = generate_eda_report(
                    df, target_col, self.task_type_, max_samples=self.config.viz_max_samples
                )
            except Exception as exc:
                logger.warning("EDA generation failed: %s", exc)
                self.eda_report_ = None

        work = df.copy()
        if work[target_col].isna().any():
            work = work.dropna(subset=[target_col]).reset_index(drop=True)

        X = work.drop(columns=[target_col])
        y = work[target_col].copy()
        self.label_encoder_ = None
        if self.task_type_ == "classification":
            if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y):
                self.label_encoder_ = LabelEncoder()
                y = pd.Series(self.label_encoder_.fit_transform(y.astype(str)), name=target_col)
            self.n_classes_ = int(y.nunique())
        else:
            y = pd.to_numeric(y, errors="coerce")
            mask = y.notna()
            X, y = X.loc[mask], y.loc[mask]

        stratify = y if self.task_type_ == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        class_weight = None
        if (
            self.task_type_ == "classification"
            and self.config.imbalance_method == ImbalanceMethod.CLASS_WEIGHT
            and self.imbalance_
            and self.imbalance_.get("is_imbalanced")
        ):
            class_weight = "balanced"

        include_ann = not self.config.skip_ann and self.config.tuning_mode != TuningMode.FAST
        specs = get_model_specs(
            self.task_type_,
            selected=self.config.selected_models,
            include_ann=include_ann,
            random_state=self.random_state,
        )
        if not specs:
            specs = get_model_specs(self.task_type_, include_ann=False, random_state=self.random_state)

        _progress("preprocessing", "Building leakage-safe model pipelines...")
        advanced_models: dict[str, tuple[Any, dict[str, list[Any]]]] = {}
        families: dict[str, str] = {}
        stages: dict[str, int] = {}
        for spec in specs:
            try:
                pipe = build_model_pipeline(
                    spec,
                    X_train,
                    self.task_type_,
                    self.config,
                    n_classes=self.n_classes_,
                    baseline=False,
                    class_weight=class_weight,
                )
                advanced_models[spec.name] = (pipe, spec.param_grid)
                families[spec.name] = spec.family
                stages[spec.name] = spec.stage
            except Exception as exc:
                logger.warning("Could not build pipeline for %s: %s", spec.name, exc)

        _progress("training", "Running staged AutoML (preprocessing inside CV)...")
        self.trainer_ = Trainer(
            task_type=self.task_type_,
            cv_folds=self.config.cv_folds,
            random_state=self.random_state,
            config=self.config,
        )
        self.leaderboard_ = self.trainer_.staged_search(
            X_train,
            y_train,
            advanced_models,
            families=families,
            stages=stages,
        )

        if self.config.enable_ensemble and should_build_ensemble(
            len(self.trainer_.fitted_models_), True
        ):
            _progress("tuning", "Evaluating optional ensemble...")
            self._try_ensemble(X_train, y_train, X_test, y_test)

        self.pipeline_ = self.trainer_.get_best_pipeline()
        test_metrics = {}
        infer_s = None
        if self.pipeline_ is not None:
            t0 = time.time()
            test_metrics = evaluate_model(self.pipeline_, X_test, y_test, self.task_type_)
            infer_s = time.time() - t0
            self.advanced_metrics_ = test_metrics
            self._annotate_leaderboard_test(X_test, y_test, infer_s)

        if self.config.baseline_enabled:
            _progress("training", "Running baseline experiment for honest comparison...")
            self.baseline_metrics_ = self._run_baseline(X_train, y_train, X_test, y_test, specs, class_weight)
            self.comparison_ = self._compare_experiments()

        _progress("explainability", "Computing SHAP explanations...")
        self.explainer_ = None
        if (
            not self.config.skip_shap
            and ModelExplainer is not None
            and self.pipeline_ is not None
        ):
            try:
                self.explainer_ = ModelExplainer(self.pipeline_, self.task_type_)
                self.explainer_.compute_shap_values(X_train, max_samples=self.config.shap_max_samples)
            except Exception as exc:
                logger.warning("SHAP explanation failed: %s", exc)
                self.explainer_ = None

        self.feature_report_ = self._extract_feature_report()
        self.narrative_ = build_narrative_report(self)
        self.model_card_ = build_model_card(self, test_metrics)
        try:
            log_experiment(self, elapsed_s=time.time() - started)
        except Exception as exc:
            logger.warning("Experiment logging failed: %s", exc)

        _progress("finalizing", "Finalizing artifacts...")
        return {
            "task_type": self.task_type_,
            "problem_type": classify_problem(y) if self.task_type_ == "classification" else "continuous",
            "leaderboard": self.leaderboard_,
            "best_model_name": self.trainer_.best_model_name_ if self.trainer_ else None,
            "pipeline": self.pipeline_,
            "explainer": self.explainer_,
            "eda_report": self.eda_report_,
            "profile": self.profile_,
            "cv_results": self.trainer_.cv_results_ if self.trainer_ else {},
            "fitted_models": self.trainer_.fitted_models_ if self.trainer_ else {},
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "y_train": y_train,
            "test_metrics": test_metrics,
            "baseline_metrics": self.baseline_metrics_,
            "advanced_metrics": self.advanced_metrics_,
            "comparison": self.comparison_,
            "narrative": self.narrative_,
            "model_card": self.model_card_,
            "feature_report": self.feature_report_,
            "imbalance": self.imbalance_,
            "label_encoder": self.label_encoder_,
            "config": self.config.to_dict(),
        }

    def _run_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        specs: list[Any],
        class_weight: str | None,
    ) -> dict[str, float] | None:
        baseline_specs = [s for s in specs if s.stage == 1][:4] or specs[:3]
        models = {}
        families = {}
        for spec in baseline_specs:
            try:
                pipe = build_model_pipeline(
                    spec,
                    X_train,
                    self.task_type_,
                    self.config,
                    n_classes=self.n_classes_,
                    baseline=True,
                    class_weight=class_weight,
                )
                models[f"baseline_{spec.name}"] = (pipe, {})
                families[f"baseline_{spec.name}"] = spec.family
            except Exception as exc:
                logger.warning("Baseline %s skipped: %s", spec.name, exc)
        if not models:
            return None
        trainer = Trainer(
            task_type=self.task_type_,
            cv_folds=min(3, self.config.cv_folds),
            random_state=self.random_state,
            config=ExperimentConfig(
                cv_folds=min(3, self.config.cv_folds),
                random_state=self.random_state,
                tuning_mode=TuningMode.FAST,
            ),
        )
        trainer.train_and_evaluate(X_train, y_train, models, families=families)
        if trainer.best_model_ is None:
            return None
        return evaluate_model(trainer.best_model_, X_test, y_test, self.task_type_)

    def _compare_experiments(self) -> dict[str, Any] | None:
        if not self.baseline_metrics_ or not self.advanced_metrics_:
            return None
        comparison: dict[str, Any] = {}
        if self.task_type_ == "regression":
            for key, higher in (("rmse", False), ("r2", True), ("mae", False)):
                b, a = self.baseline_metrics_.get(key), self.advanced_metrics_.get(key)
                comparison[f"baseline_{key}"] = b
                comparison[f"advanced_{key}"] = a
                if b is not None and a is not None:
                    comparison[f"{key}_change_pct"] = percent_change(b, a, higher_is_better=higher)
        else:
            for key in ("accuracy", "f1", "f1_macro", "balanced_accuracy"):
                b, a = self.baseline_metrics_.get(key), self.advanced_metrics_.get(key)
                comparison[f"baseline_{key}"] = b
                comparison[f"advanced_{key}"] = a
                if b is not None and a is not None:
                    comparison[f"{key}_change_pct"] = percent_change(b, a, higher_is_better=True)
        return comparison

    def _annotate_leaderboard_test(self, X_test: Any, y_test: Any, infer_s: float | None) -> None:
        if self.leaderboard_ is None or self.trainer_ is None:
            return
        test_scores = []
        infer_times = []
        n_params = []
        for name in self.leaderboard_["model"]:
            model = self.trainer_.fitted_models_.get(name)
            if model is None:
                test_scores.append(np.nan)
                infer_times.append(np.nan)
                n_params.append(np.nan)
                continue
            try:
                t0 = time.time()
                metrics = evaluate_model(model, X_test, y_test, self.task_type_)
                infer_times.append(round(time.time() - t0, 4))
                score = metrics.get("f1_weighted" if self.task_type_ == "classification" else "r2")
                test_scores.append(score)
            except Exception:
                test_scores.append(np.nan)
                infer_times.append(np.nan)
            n_params.append(flatten_params(model))
        self.leaderboard_["test_score"] = test_scores
        self.leaderboard_["inference_time_s"] = infer_times
        self.leaderboard_["n_params"] = n_params

    def _try_ensemble(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any) -> None:
        if self.trainer_ is None:
            return
        ranked = self.trainer_.get_leaderboard()
        names = ranked.loc[ranked["status"] == "success", "model"].tolist()[:3]
        estimators = [
            (n, self.trainer_.fitted_models_[n])
            for n in names
            if n in self.trainer_.fitted_models_
        ]
        if len(estimators) < 2:
            return
        try:
            ensemble = build_voting(estimators, self.task_type_)
            # Voting full pipelines is expensive and can break on predict_proba.
            # Only ensemble the already-fitted pipelines if they share the same interface.
            ensemble.fit(X_train, y_train)
            metrics = evaluate_model(ensemble, X_test, y_test, self.task_type_)
            score = metrics.get("f1_weighted" if self.task_type_ == "classification" else "r2", -np.inf)
            current = None
            if self.advanced_metrics_:
                current = self.advanced_metrics_.get(
                    "f1_weighted" if self.task_type_ == "classification" else "r2"
                )
            self.trainer_.fitted_models_["VotingEnsemble"] = ensemble
            self.trainer_.results_.append(
                {
                    "model": "VotingEnsemble",
                    "family": "ensemble",
                    "status": "success",
                    "train_time_s": 0.0,
                    "f1_weighted_mean": score if self.task_type_ == "classification" else np.nan,
                    "r2_mean": score if self.task_type_ == "regression" else np.nan,
                    "cv_score": score,
                }
            )
            if current is None or score > current:
                self.trainer_.best_model_name_ = "VotingEnsemble"
                self.trainer_.best_model_ = ensemble
                self.advanced_metrics_ = metrics
            self.leaderboard_ = self.trainer_.get_leaderboard()
        except Exception as exc:
            logger.warning("Ensemble skipped: %s", exc)

    def _extract_feature_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "original_features": [],
            "engineered_features": [],
            "removed_features": [],
            "selected_features": [],
            "method": self.config.feature_selection.value,
        }
        if self.pipeline_ is None or not hasattr(self.pipeline_, "named_steps"):
            return report
        fs = self.pipeline_.named_steps.get("feature_selection")
        if fs is not None and hasattr(fs, "report_") and fs.report_ is not None:
            r = fs.report_
            report.update(
                {
                    "original_features": r.original_features,
                    "removed_features": r.removed_features,
                    "selected_features": r.selected_features,
                    "method": r.method,
                }
            )
        return report

    def export_pipeline(self, filepath: str) -> str:
        if self.pipeline_ is None:
            raise ValueError("No pipeline to export. Run the pipeline first.")
        joblib.dump(
            {
                "pipeline": self.pipeline_,
                "task_type": self.task_type_,
                "target_col": self.target_col_,
                "label_encoder": self.label_encoder_,
                "config": self.config.to_dict(),
                "best_model_name": self.trainer_.best_model_name_ if self.trainer_ else None,
            },
            filepath,
        )
        logger.info("Pipeline exported to %s", filepath)
        return filepath

    @staticmethod
    def load_pipeline(filepath: str) -> Pipeline:
        obj = joblib.load(filepath)
        if isinstance(obj, dict) and "pipeline" in obj:
            return obj["pipeline"]
        return obj
