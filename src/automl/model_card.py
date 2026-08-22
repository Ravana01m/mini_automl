"""Model card generation for exported pipelines."""

from __future__ import annotations

from typing import Any


def build_model_card(pipeline: Any, test_metrics: dict[str, float] | None = None) -> dict[str, Any]:
    trainer = pipeline.trainer_
    config = pipeline.config
    return {
        "model": trainer.best_model_name_ if trainer else None,
        "dataset": {
            "rows": getattr(pipeline.profile_, "n_rows", None),
            "columns": getattr(pipeline.profile_, "n_cols", None),
            "target": pipeline.target_col_,
        },
        "task": pipeline.task_type_,
        "features": pipeline.feature_report_,
        "preprocessing": {
            "numeric_imputer": config.numeric_imputer,
            "categorical_imputer": config.categorical_imputer,
            "scaler": config.scaler.value,
            "encoder": config.encoder.value,
            "outlier_method": config.outlier_method.value,
            "outlier_strategy": config.outlier_strategy.value,
        },
        "feature_selection": config.feature_selection.value,
        "training_configuration": config.to_dict(),
        "cv_strategy": {
            "folds": config.cv_folds,
            "repeats": config.cv_repeats,
            "shuffle": config.shuffle,
            "seed": config.random_state,
        },
        "metrics": test_metrics or pipeline.advanced_metrics_,
        "limitations": [
            "Educational / research AutoML platform, not a replacement for production ML governance.",
            "Automatic feature generation can still miss domain-specific signals.",
            "Held-out test metrics can be noisy on small datasets.",
            "Neural network results depend on TensorFlow availability and runtime hardware.",
        ],
        "known_risks": [
            "Identifier-like columns may still leak if they encode the label.",
            "High-cardinality target encoding can overfit small categories.",
            "Imbalance corrections change class priors at training time.",
        ],
        "feature_importance": _importance(pipeline),
    }


def _importance(pipeline: Any) -> dict[str, float] | None:
    explainer = getattr(pipeline, "explainer_", None)
    if explainer is None or getattr(explainer, "shap_values_", None) is None:
        return None
    try:
        import numpy as np

        values = np.abs(explainer.shap_values_).mean(axis=0)
        names = explainer.feature_names_ or [f"f{i}" for i in range(len(values))]
        return {str(n): float(v) for n, v in zip(names[:30], values[:30])}
    except Exception:
        return None
