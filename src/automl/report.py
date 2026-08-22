"""Human-readable AutoML run narrative."""

from __future__ import annotations

from typing import Any


def build_narrative_report(pipeline: Any) -> str:
    profile = pipeline.profile_
    config = pipeline.config
    trainer = pipeline.trainer_
    comparison = pipeline.comparison_ or {}
    lines = [
        "# AutoML Run Report",
        "",
        "## Dataset",
        f"- Rows: {getattr(profile, 'n_rows', 'n/a')}",
        f"- Columns: {getattr(profile, 'n_cols', 'n/a')}",
        f"- Task: {pipeline.task_type_}",
        f"- Target: {pipeline.target_col_}",
        "",
        "## Data Quality",
        f"- Missing cells: {getattr(profile, 'missing_cells', 0)} ({getattr(profile, 'missing_pct', 0):.2f}%)",
        f"- Duplicate rows: {getattr(profile, 'duplicate_rows', 0)}",
        f"- Constant columns: {getattr(profile, 'constant', [])}",
        f"- Potential leakage columns: {getattr(profile, 'potential_leakage', [])}",
    ]
    if pipeline.imbalance_:
        lines += [
            f"- Class counts: {pipeline.imbalance_.get('class_counts')}",
            f"- Imbalance ratio: {pipeline.imbalance_.get('imbalance_ratio')}",
        ]
    lines += [
        "",
        "## Preprocessing",
        f"- Numeric imputation: {config.numeric_imputer}",
        f"- Categorical imputation: {config.categorical_imputer}",
        f"- Encoding: {config.encoder.value}",
        f"- Scaling: {config.scaler.value} (auto-selected per model family)",
        f"- Outlier handling: {config.outlier_method.value} / {config.outlier_strategy.value}",
        "",
        "## Feature Engineering",
        f"- Enabled: {config.enable_feature_engineering}",
        f"- Log transform: {config.enable_log_transform}",
        f"- Polynomial: {config.enable_polynomial}",
        f"- Interactions: {config.enable_interactions}",
        f"- Ratios: {config.enable_ratios}",
        "",
        "## Feature Selection",
        f"- Strategy: {config.feature_selection.value}",
        f"- Before: {len((pipeline.feature_report_ or {}).get('original_features') or [])}",
        f"- After: {len((pipeline.feature_report_ or {}).get('selected_features') or [])}",
        f"- Removed: {(pipeline.feature_report_ or {}).get('removed_features')}",
        "",
        "## Model Search",
        f"- Models tested: {len(trainer.results_) if trainer else 0}",
        f"- Tuning mode: {config.tuning_mode.value}",
        f"- Best model: {trainer.best_model_name_ if trainer else None}",
    ]
    metrics = pipeline.advanced_metrics_ or {}
    lines += ["", "## Best Model Metrics"]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    if comparison:
        lines += ["", "## Baseline vs Advanced"]
        for key, value in comparison.items():
            lines.append(f"- {key}: {value}")
        improved = any(
            str(k).endswith("_change_pct") and isinstance(v, float) and v > 0
            for k, v in comparison.items()
            if "rmse" not in k and "mae" not in k
        )
        if not improved:
            lines.append(
                "- Advanced did not uniformly beat baseline. This is reported honestly; complexity is not always better."
            )
    lines += [
        "",
        "## Why it won",
        _why_it_won(pipeline),
        "",
        "## Leakage controls",
        "- Train/test split happens before any learned transform.",
        "- Imputation, scaling, encoding, outlier bounds, feature engineering, and feature selection refit inside each CV fold.",
        "- Oversampling, if enabled, is inside the CV pipeline only.",
        "- EDA correlation is not used as the CV feature-selection decision.",
    ]
    return "\n".join(lines)


def _why_it_won(pipeline: Any) -> str:
    trainer = pipeline.trainer_
    if trainer is None or trainer.best_model_name_ is None:
        return "No successful model was trained."
    name = trainer.best_model_name_
    board = trainer.get_leaderboard()
    if board.empty:
        return f"{name} was selected as the only available model."
    row = board.iloc[0]
    metric = "f1_weighted_mean" if pipeline.task_type_ == "classification" else "r2_mean"
    score = row.get(metric)
    std = row.get("cv_std") or row.get("f1_weighted_std") or row.get("r2_std")
    return (
        f"{name} won on cross-validated {metric.replace('_', ' ')} "
        f"({score}, std={std}). Test metrics were computed on a held-out split "
        "and were not used to select the winner."
    )
