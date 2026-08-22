"""Save / load complete AutoML pipelines for raw-dataframe inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_model(bundle: dict[str, Any] | Any, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return str(path)


def load_model(path: str | Path) -> Any:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return SavedPipeline(obj)
    return SavedPipeline({"pipeline": obj})


class SavedPipeline:
    """Thin inference wrapper around a fitted sklearn/imblearn pipeline."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.task_type = bundle.get("task_type")
        self.target_col = bundle.get("target_col")
        self.label_encoder = bundle.get("label_encoder")

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.target_col and self.target_col in frame.columns:
            return frame.drop(columns=[self.target_col])
        return frame

    def predict(self, raw_dataframe: pd.DataFrame):
        preds = self.pipeline.predict(self._features(raw_dataframe))
        if self.label_encoder is not None:
            try:
                return self.label_encoder.inverse_transform(preds)
            except Exception:
                return preds
        return preds

    def predict_proba(self, raw_dataframe: pd.DataFrame):
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("This pipeline does not support predict_proba.")
        return self.pipeline.predict_proba(self._features(raw_dataframe))
