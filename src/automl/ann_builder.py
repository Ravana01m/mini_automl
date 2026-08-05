"""Keras ANN model factory for classification and regression.

Builds TensorFlow/Keras neural networks wrapped with scikeras so they
behave as native sklearn estimators — compatible with Pipeline, GridSearchCV,
cross_val_score, and joblib serialization.
"""

from __future__ import annotations

import logging

import tensorflow as tf
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def build_classifier_model(
    input_dim: int,
    n_classes: int,
    hidden_units: tuple[int, ...] = (128, 64),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build a Keras classification model.

    Architecture: Input -> [Dense -> BatchNorm -> Dropout] x N -> Softmax output
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    
    for i, units in enumerate(hidden_units):
        model.add(tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}"))
        model.add(tf.keras.layers.BatchNormalization(name=f"bn_{i}"))
        dr = dropout_rate if i == 0 else max(dropout_rate - 0.1, 0.1)
        model.add(tf.keras.layers.Dropout(dr, name=f"dropout_{i}"))
    
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax", name="output"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_regressor_model(
    input_dim: int,
    hidden_units: tuple[int, ...] = (128, 64),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build a Keras regression model.

    Architecture: Input -> [Dense -> BatchNorm -> Dropout] x N -> Linear output
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    
    for i, units in enumerate(hidden_units):
        model.add(tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}"))
        model.add(tf.keras.layers.BatchNormalization(name=f"bn_{i}"))
        dr = dropout_rate if i == 0 else max(dropout_rate - 0.1, 0.1)
        model.add(tf.keras.layers.Dropout(dr, name=f"dropout_{i}"))
    
    model.add(tf.keras.layers.Dense(1, activation="linear", name="output"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_keras_estimator(
    task_type: str,
    input_dim: int,
    n_classes: int | None = None,
    epochs: int = 100,
    batch_size: int = 32,
) -> BaseEstimator:
    """Return a scikeras-wrapped Keras estimator.

    Args:
        task_type: 'classification' or 'regression'.
        input_dim: Number of input features.
        n_classes: Number of classes (classification only).
        epochs: Maximum training epochs.
        batch_size: Training batch size.

    Returns:
        KerasClassifier or KerasRegressor (sklearn-compatible).
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]
    
    if task_type == "classification":
        from scikeras.wrappers import KerasClassifier
        
        return KerasClassifier(
            model=build_classifier_model,
            input_dim=input_dim,
            n_classes=n_classes or 2,
            hidden_units=(128, 64),
            dropout_rate=0.3,
            learning_rate=1e-3,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            random_state=42,
        )
    else:
        from scikeras.wrappers import KerasRegressor
        
        return KerasRegressor(
            model=build_regressor_model,
            input_dim=input_dim,
            hidden_units=(128, 64),
            dropout_rate=0.3,
            learning_rate=1e-3,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            random_state=42,
        )
