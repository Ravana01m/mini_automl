"""Experiment configuration and global reproducibility."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TuningMode(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


class FeatureSelectionStrategy(str, Enum):
    NONE = "none"
    AUTOMATIC = "automatic"
    LIGHT = "light"
    AGGRESSIVE = "aggressive"


class OutlierMethod(str, Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    PERCENTILE = "percentile"


class OutlierStrategy(str, Enum):
    CLIP = "clip"
    WINSORIZE = "winsorize"
    REPLACE = "replace"
    NONE = "none"


class ImbalanceMethod(str, Enum):
    NONE = "none"
    CLASS_WEIGHT = "class_weight"
    SMOTE = "smote"
    RANDOM_OVERSAMPLE = "random_oversample"


class ScalerType(str, Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    POWER = "power"
    QUANTILE = "quantile"
    NONE = "none"
    AUTO = "auto"


class EncoderType(str, Enum):
    AUTO = "auto"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    TARGET = "target"


@dataclass
class ExperimentConfig:
    """User-facing configuration for a single AutoML run."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    cv_repeats: int = 1
    shuffle: bool = True

    tuning_mode: TuningMode = TuningMode.STANDARD
    optuna_trials: int = 25
    optuna_timeout: int = 180
    stage1_top_k: int = 5
    stage2_top_k: int = 3
    enable_ensemble: bool = False
    skip_ann: bool = False

    numeric_imputer: str = "median"
    categorical_imputer: str = "most_frequent"
    add_missing_indicator: bool = True
    scaler: ScalerType = ScalerType.AUTO
    encoder: EncoderType = EncoderType.AUTO
    cardinality_threshold: int = 15

    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_strategy: OutlierStrategy = OutlierStrategy.CLIP
    outlier_factor: float = 1.5
    outlier_zscore: float = 3.0
    winsor_limits: tuple[float, float] = (0.01, 0.99)

    enable_feature_engineering: bool = True
    enable_log_transform: bool = True
    enable_datetime_features: bool = True
    enable_polynomial: bool = False
    enable_interactions: bool = False
    enable_ratios: bool = False
    enable_binning: bool = False
    poly_degree: int = 2
    poly_top_k: int = 6
    skew_threshold: float = 1.0

    feature_selection: FeatureSelectionStrategy = FeatureSelectionStrategy.AUTOMATIC
    correlation_threshold: float = 0.95

    imbalance_method: ImbalanceMethod = ImbalanceMethod.CLASS_WEIGHT
    selected_models: list[str] | None = None

    shap_max_samples: int = 80
    viz_max_samples: int = 1500
    drop_id_like: bool = False
    drop_constant: bool = True
    drop_all_null: bool = True

    baseline_enabled: bool = True
    skip_eda: bool = False
    skip_shap: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Enum):
                payload[key] = value.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        kwargs: dict[str, Any] = dict(data)
        enum_fields = {
            "tuning_mode": TuningMode,
            "scaler": ScalerType,
            "encoder": EncoderType,
            "outlier_method": OutlierMethod,
            "outlier_strategy": OutlierStrategy,
            "feature_selection": FeatureSelectionStrategy,
            "imbalance_method": ImbalanceMethod,
        }
        for name, enum_cls in enum_fields.items():
            if name in kwargs and not isinstance(kwargs[name], enum_cls):
                kwargs[name] = enum_cls(kwargs[name])
        return cls(**{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__})


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, TensorFlow, and Optuna where available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    except Exception:
        pass

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass


@dataclass
class RunPaths:
    """Local artifact locations."""

    experiments_dir: str = "experiments"
    models_dir: str = "models"
    reports_dir: str = "reports"


DEFAULT_CONFIG = ExperimentConfig()
