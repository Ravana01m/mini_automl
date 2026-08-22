"""Sklearn-compatible Keras ANN factory.

Architecture (classification and regression):
Input → Dense → BatchNorm → Activation → Dropout → Dense → BatchNorm → Dropout → Output

EarlyStopping and ReduceLROnPlateau monitor validation loss, never training loss.
Input dimension is inferred at fit time via SciKeras metadata so the ANN can
live inside a CV pipeline after feature selection.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def tensorflow_available() -> bool:
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


def _resolve_dim(meta: dict[str, Any] | None, kwargs: dict[str, Any], key: str, default: int) -> int:
    if meta and meta.get(key) is not None:
        return int(meta[key])
    if kwargs.get(key) is not None:
        return int(kwargs[key])
    alias = key.rstrip("_")
    if kwargs.get(alias) is not None:
        return int(kwargs[alias])
    if key == "n_features_in_" and kwargs.get("input_dim") is not None:
        return int(kwargs["input_dim"])
    if key == "n_classes_" and kwargs.get("n_classes") is not None:
        return int(kwargs["n_classes"])
    return default


def _get_optimizer(name: str, learning_rate: float) -> Any:
    import tensorflow as tf

    name = (name or "adam").lower()
    if name == "adamw":
        try:
            return tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        except Exception:
            logger.warning("AdamW unavailable; falling back to Adam")
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def _stack_hidden(
    model: Any,
    hidden_units: tuple[int, ...],
    dropout_rate: float,
    activation: str,
) -> None:
    import tensorflow as tf

    for i, units in enumerate(hidden_units):
        model.add(tf.keras.layers.Dense(int(units), name=f"dense_{i}"))
        model.add(tf.keras.layers.BatchNormalization(name=f"bn_{i}"))
        model.add(tf.keras.layers.Activation(activation, name=f"act_{i}"))
        dr = dropout_rate if i == 0 else max(dropout_rate - 0.1, 0.1)
        model.add(tf.keras.layers.Dropout(dr, name=f"dropout_{i}"))


def build_classifier_model(
    hidden_units: tuple[int, ...] = (128, 64),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    activation: str = "relu",
    optimizer: str = "adam",
    meta: dict[str, Any] | None = None,
    input_dim: int | None = None,
    n_classes: int | None = None,
    **kwargs: Any,
) -> Any:
    """Build a Keras classification model. Input dim is inferred at fit time."""
    import tensorflow as tf

    n_features_in = _resolve_dim(meta, {**kwargs, "input_dim": input_dim}, "n_features_in_", input_dim or 1)
    n_cls = _resolve_dim(meta, {**kwargs, "n_classes": n_classes}, "n_classes_", n_classes or 2)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_features_in,)))
    _stack_hidden(model, hidden_units, dropout_rate, activation)
    if n_cls <= 2:
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output"))
        loss = "binary_crossentropy"
    else:
        model.add(tf.keras.layers.Dense(n_cls, activation="softmax", name="output"))
        loss = "sparse_categorical_crossentropy"
    model.compile(
        optimizer=_get_optimizer(optimizer, learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def build_regressor_model(
    hidden_units: tuple[int, ...] = (128, 64),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    activation: str = "relu",
    optimizer: str = "adam",
    meta: dict[str, Any] | None = None,
    input_dim: int | None = None,
    **kwargs: Any,
) -> Any:
    """Build a Keras regression model. Input dim is inferred at fit time."""
    import tensorflow as tf

    n_features_in = _resolve_dim(meta, {**kwargs, "input_dim": input_dim}, "n_features_in_", input_dim or 1)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_features_in,)))
    _stack_hidden(model, hidden_units, dropout_rate, activation)
    model.add(tf.keras.layers.Dense(1, activation="linear", name="output"))
    model.compile(
        optimizer=_get_optimizer(optimizer, learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def scikeras_available() -> bool:
    try:
        import scikeras  # noqa: F401

        return True
    except Exception:
        return False


def _callbacks(patience: int = 8) -> list[Any]:
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 2),
            min_lr=1e-6,
        ),
    ]


def get_keras_estimator(
    task_type: str,
    input_dim: int | None = None,
    n_classes: int | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    hidden_units: tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.3,
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
) -> BaseEstimator:
    """Return a scikeras-wrapped Keras estimator.

    validation_split is enabled so EarlyStopping can monitor val_loss.
    """
    if not tensorflow_available():
        raise RuntimeError("TensorFlow is not available in this environment.")
    if not scikeras_available():
        raise RuntimeError("SciKeras is not available in this environment.")

    common = dict(
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        activation=activation,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=_callbacks(),
        validation_split=0.2,
        verbose=0,
        random_state=42,
    )
    if input_dim is not None:
        common["input_dim"] = input_dim

    if task_type == "classification":
        from scikeras.wrappers import KerasClassifier

        if n_classes is not None:
            common["n_classes"] = n_classes
        return KerasClassifier(model=build_classifier_model, **common)
    from scikeras.wrappers import KerasRegressor

    return KerasRegressor(model=build_regressor_model, **common)
