from pathlib import Path

import numpy as np
from PIL import Image

from src.data import CHANNELS, CLASS_NAMES, IMAGE_SIZE, K, load_dataset


def _write_dataset(root: Path) -> None:
    """Crée un jeu FER2013 minimal organisé en dossiers."""
    for split, count in (("train", 10), ("test", 2)):
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for index in range(count):
                pixels = np.full((40, 40), label * 20 + index, dtype=np.uint8)
                Image.fromarray(pixels).save(class_dir / f"{index}.png")


def test_load_dataset_shapes_and_classes(tmp_path: Path) -> None:
    """Le chargement produit les shapes et classes attendues."""
    _write_dataset(tmp_path)
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(tmp_path)

    assert CLASS_NAMES == ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
    assert K == len(CLASS_NAMES) == 7
    for X, y in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        assert X.shape == (len(y), *IMAGE_SIZE, CHANNELS)
        assert X.dtype == np.float32
        assert 0.0 <= X.min() <= X.max() <= 1.0
        assert y.shape == (len(y),)
        assert y.dtype == np.int64
    assert set(y_train) == set(y_val) == set(y_test) == set(range(K))
