"""Leakage-safe imbalance handling.

Oversampling must happen inside the CV pipeline, never on validation/test.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin

from automl.config import ImbalanceMethod

logger = logging.getLogger(__name__)


def imblearn_available() -> bool:
    try:
        import imblearn  # noqa: F401

        return True
    except Exception:
        return False


def make_sampler(method: str, random_state: int = 42) -> Any | None:
    if method in {ImbalanceMethod.NONE.value, ImbalanceMethod.CLASS_WEIGHT.value}:
        return None
    if not imblearn_available():
        logger.warning("imbalanced-learn is not installed; skipping sampler '%s'", method)
        return None
    if method == ImbalanceMethod.SMOTE.value:
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=random_state, k_neighbors=3)
    if method == ImbalanceMethod.RANDOM_OVERSAMPLE.value:
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(random_state=random_state)
    return None


def make_pipeline_class(needs_sampler: bool):
    if needs_sampler and imblearn_available():
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline
    from sklearn.pipeline import Pipeline

    return Pipeline


class IdentitySampler(BaseEstimator, TransformerMixin):
    """No-op placeholder so pipeline diagrams stay stable."""

    def fit(self, X: Any, y: Any = None) -> "IdentitySampler":
        return self

    def transform(self, X: Any) -> Any:
        return X

    def fit_resample(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y
