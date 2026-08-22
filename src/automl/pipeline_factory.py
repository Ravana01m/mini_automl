"""Assemble leakage-safe sklearn/imblearn pipelines per model family."""

from __future__ import annotations

import logging
from typing import Any

from automl.config import ExperimentConfig, ImbalanceMethod
from automl.feature_engineering import (
    DataFrameLog1p,
    DatetimeFeatureExtractor,
    FeatureEngineeringEngine,
)
from automl.feature_selection import FeatureSelectionEngine
from automl.preprocessing import build_preprocessor
from automl.quality import ColumnPruner, outlier_handler_from_config
from automl.sampling import make_pipeline_class, make_sampler
from automl.types import ModelSpec

logger = logging.getLogger(__name__)


def build_model_pipeline(
    spec: ModelSpec,
    X_sample,
    task_type: str,
    config: ExperimentConfig,
    n_classes: int | None = None,
    baseline: bool = False,
    class_weight: str | None = None,
) -> Any:
    """Return an unfitted pipeline: raw features → prediction.

    All learned steps are inside the pipeline so CV can refit them per fold.
    """
    needs_sampler = (
        task_type == "classification"
        and not baseline
        and config.imbalance_method
        in {ImbalanceMethod.SMOTE, ImbalanceMethod.RANDOM_OVERSAMPLE}
    )
    PipelineCls = make_pipeline_class(needs_sampler)
    steps: list[tuple[str, Any]] = [
        (
            "pruner",
            ColumnPruner(
                drop_constant=config.drop_constant,
                drop_all_null=config.drop_all_null,
                drop_id_like=config.drop_id_like,
            ),
        ),
    ]
    if config.enable_datetime_features:
        steps.append(("datetime", DatetimeFeatureExtractor()))
    if not baseline and config.outlier_strategy.value != "none":
        steps.append(("outliers", outlier_handler_from_config(config)))
    if not baseline and config.enable_log_transform:
        steps.append(("log1p", DataFrameLog1p(config.skew_threshold)))

    preprocessor, _, _ = build_preprocessor(
        X_sample,
        target_col="",
        task_type=task_type,
        family=spec.family,
        config=config if not baseline else ExperimentConfig(
            scaler=config.scaler,
            encoder=config.encoder,
            add_missing_indicator=False,
            numeric_imputer="median",
            categorical_imputer="most_frequent",
        ),
    )
    steps.append(("preprocessor", preprocessor))

    if not baseline and config.enable_feature_engineering:
        steps.append(
            (
                "feature_engineering",
                FeatureEngineeringEngine(
                    enable_log=config.enable_log_transform,
                    enable_polynomial=config.enable_polynomial,
                    enable_interactions=config.enable_interactions,
                    enable_ratios=config.enable_ratios,
                    enable_binning=config.enable_binning,
                    skew_threshold=config.skew_threshold,
                    poly_top_k=config.poly_top_k,
                    poly_degree=config.poly_degree,
                ),
            )
        )

    if not baseline and config.feature_selection.value != "none":
        steps.append(
            (
                "feature_selection",
                FeatureSelectionEngine(
                    task_type=task_type,
                    strategy=config.feature_selection.value,
                    correlation_threshold=config.correlation_threshold,
                    family=spec.family,
                    random_state=config.random_state,
                ),
            )
        )

    sampler = make_sampler(config.imbalance_method.value, config.random_state) if needs_sampler else None
    if sampler is not None:
        steps.append(("sampler", sampler))

    model_kwargs: dict[str, Any] = {}
    if spec.family == "neural":
        model_kwargs["epochs"] = 20 if baseline or config.tuning_mode.value == "fast" else 40
        model_kwargs["batch_size"] = 32
        if n_classes is not None:
            model_kwargs["n_classes"] = n_classes
    if class_weight and spec.supports_class_weight:
        model_kwargs["class_weight"] = class_weight
    estimator = spec.factory(**model_kwargs)
    steps.append(("model", estimator))
    return PipelineCls(steps)
