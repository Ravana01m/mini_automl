"""Optional voting / stacking ensembles."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.base import clone
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

logger = logging.getLogger(__name__)


def should_build_ensemble(n_successful: int, enable: bool) -> bool:
    return bool(enable and n_successful >= 2)


def build_voting(
    named_estimators: list[tuple[str, Any]],
    task_type: str,
) -> Any:
    estimators = [(name, clone(est)) for name, est in named_estimators]
    if task_type == "classification":
        return VotingClassifier(estimators=estimators, voting="soft")
    return VotingRegressor(estimators=estimators)


def build_stacking(
    named_estimators: list[tuple[str, Any]],
    task_type: str,
    random_state: int = 42,
) -> Any:
    estimators = [(name, clone(est)) for name, est in named_estimators]
    if task_type == "classification":
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=400, solver="lbfgs"),
            passthrough=False,
        )
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(random_state=random_state),
        passthrough=False,
    )
