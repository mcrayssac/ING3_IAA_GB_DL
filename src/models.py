"""Architectures des modèles : baseline MLP et CNN."""

import tensorflow as tf

from src.data import CHANNELS, CLASS_NAMES, IMAGE_SIZE, SEED


K = len(CLASS_NAMES)
HIDDEN_UNITS = 128
INPUT_SHAPE = (*IMAGE_SIZE, CHANNELS)
CNN_FILTERS = (32, 64, 128)
KERNEL_SIZE = 3


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


def build_cnn(
    filters: tuple[int, ...] = CNN_FILTERS,
    kernel_size: int = KERNEL_SIZE,
    dense_units: int = HIDDEN_UNITS,
    seed: int = SEED,
) -> tf.keras.Model:
    """Construit un CNN : blocs Conv2D + ReLU puis MaxPooling, puis classifieur dense."""
    if len(filters) < 2:
        raise ValueError("Le CNN doit comporter au moins deux blocs de convolution.")
    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.Input(shape=INPUT_SHAPE)]
    for count in filters:
        layers += [
            tf.keras.layers.Conv2D(count, kernel_size, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
        ]
    layers += [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dense(K, activation="softmax"),
    ]
    model = tf.keras.Sequential(layers, name="cnn")
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, K)
    assert K == len(CLASS_NAMES) == model.layers[-1].units
    return model
