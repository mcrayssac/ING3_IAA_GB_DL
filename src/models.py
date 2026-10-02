"""Architecture de la baseline MLP."""

import tensorflow as tf

from src.data import CHANNELS, CLASS_NAMES, IMAGE_SIZE, SEED


K = len(CLASS_NAMES)
HIDDEN_UNITS = 128
INPUT_SHAPE = (*IMAGE_SIZE, CHANNELS)


def build_mlp(hidden_units: int = HIDDEN_UNITS, seed: int = SEED) -> tf.keras.Model:
    """Construit un MLP avec une initialisation reproductible."""
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dense(K, activation="softmax"),
        ],
        name="baseline_mlp",
    )
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, K)
    assert K == len(CLASS_NAMES) == model.layers[-1].units
    return model
