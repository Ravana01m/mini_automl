"""Mini AutoML — leakage-safe automated machine learning platform."""

from __future__ import annotations

from typing import Any

__version__ = "2.0.0"


def __getattr__(name: str) -> Any:
    if name == "AutoMLPipeline":
        from automl.pipeline_builder import AutoMLPipeline

        return AutoMLPipeline
    if name == "ExperimentConfig":
        from automl.config import ExperimentConfig

        return ExperimentConfig
    if name == "set_global_seed":
        from automl.config import set_global_seed

        return set_global_seed
    if name == "load_model":
        from automl.serialization import load_model

        return load_model
    if name == "save_model":
        from automl.serialization import save_model

        return save_model
    raise AttributeError(name)


__all__ = [
    "AutoMLPipeline",
    "ExperimentConfig",
    "load_model",
    "save_model",
    "set_global_seed",
    "__version__",
]
