"""Tests for Keras ANN model factory."""

from __future__ import annotations

import numpy as np
import pytest

from automl.ann_builder import (
    _callbacks,
    build_classifier_model,
    build_regressor_model,
    get_keras_estimator,
    tensorflow_available,
)

pytestmark = pytest.mark.skipif(not tensorflow_available(), reason="TensorFlow not installed")


class TestANNBuilder:
    def test_classifier_builds_and_compiles(self) -> None:
        model = build_classifier_model(meta={"n_features_in_": 4, "n_classes_": 3})
        assert model.input_shape[-1] == 4
        assert model.optimizer is not None

    def test_regressor_builds_and_compiles(self) -> None:
        model = build_regressor_model(meta={"n_features_in_": 5})
        assert model.output_shape[-1] == 1

    def test_classifier_output_shape(self) -> None:
        binary = build_classifier_model(meta={"n_features_in_": 3, "n_classes_": 2})
        assert binary.output_shape[-1] == 1
        multi = build_classifier_model(meta={"n_features_in_": 3, "n_classes_": 4})
        assert multi.output_shape[-1] == 4

    def test_regressor_output_shape(self) -> None:
        model = build_regressor_model(input_dim=6)
        assert model.count_params() > 0

    def test_scikeras_wrapper_fit_predict(self) -> None:
        pytest.importorskip("scikeras")
        rng = np.random.RandomState(0)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        est = get_keras_estimator("classification", epochs=3, batch_size=16, hidden_units=(8, 4))
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == len(y)

    def test_early_stopping_monitors_validation_loss(self) -> None:
        monitors = [getattr(cb, "monitor", None) for cb in _callbacks()]
        assert monitors.count("val_loss") >= 2
        assert "loss" not in monitors
