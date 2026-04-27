"""
lstm_model.py
-------------
LSTM-based sequence model for Vietlott draw prediction.

Architecture:
  Input  → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(6) → Sigmoid
  Output is 6 values in [0, 1]; multiply by max_val to get ball estimates.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy TensorFlow import so the module can be imported even without TF
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed. LSTM model unavailable.")


# ── Model builder ─────────────────────────────────────────────────────────────
def build_lstm_model(seq_len: int = 10, n_features: int = 6) -> "tf.keras.Model":
    """
    Build and compile the LSTM model.

    Parameters
    ----------
    seq_len    : look-back window (timesteps)
    n_features : number of balls per draw (6)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for the LSTM model. "
                           "Install it with: pip install tensorflow")

    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_features, activation="sigmoid"),  # outputs in [0,1]
    ], name="vietlott_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    logger.info("LSTM model built: %d parameters",
                model.count_params())
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train_lstm(X: np.ndarray,
               y: np.ndarray,
               seq_len: int = 10,
               epochs: int = 100,
               batch_size: int = 32,
               validation_split: float = 0.1,
               save_path: str | None = None) -> tuple:
    """
    Train the LSTM model.

    Parameters
    ----------
    X              : (samples, seq_len, 6) float32  – input sequences
    y              : (samples, 6)          float32  – target draws
    seq_len        : must match X.shape[1]
    epochs         : training epochs
    batch_size     : mini-batch size
    validation_split: fraction of data for validation
    save_path      : if provided, save the trained model here (.keras)

    Returns
    -------
    (model, history)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required. pip install tensorflow")

    model = build_lstm_model(seq_len=seq_len, n_features=X.shape[2])

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=7, min_lr=1e-6),
    ]

    logger.info("Training LSTM for up to %d epochs …", epochs)
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=cb_list,
        verbose=1,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        model.save(save_path)
        logger.info("Model saved to %s", save_path)

    return model, history


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_next_raw(model: "tf.keras.Model",
                     last_sequence: np.ndarray,
                     max_val: int = 55) -> np.ndarray:
    """
    Run one forward pass and return 6 raw continuous estimates.

    Parameters
    ----------
    last_sequence : (1, seq_len, 6) float32 – the most recent window
    max_val       : 55 or 45

    Returns
    -------
    1-D array of 6 floats (ball number estimates, not necessarily integers)
    """
    raw = model.predict(last_sequence, verbose=0)[0]  # shape (6,)
    return raw * max_val


def load_lstm_model(save_path: str) -> "tf.keras.Model":
    """Load a previously saved Keras model."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required. pip install tensorflow")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model file not found: {save_path}")
    return tf.keras.models.load_model(save_path)
