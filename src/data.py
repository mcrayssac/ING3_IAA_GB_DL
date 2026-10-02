"""Chargement et préparation des données FER2013."""

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (48, 48)
CHANNELS = 1
K = 7
CLASS_NAMES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
VALIDATION_SIZE = 0.15
SEED = 42

assert K == len(CLASS_NAMES)


def preprocess_face(image: Image.Image | np.ndarray) -> np.ndarray:
    """Redimensionne une image en niveaux de gris et normalise ses pixels."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == CHANNELS:
        array = array[..., 0]
    if array.dtype.kind == "f":
        array = array * 255.0 if array.max() <= 1.0 else array
    array = np.clip(array, 0, 255).astype(np.uint8)
    grayscale = Image.fromarray(array).convert("L")
    resized = grayscale.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    face = np.asarray(resized, dtype=np.float32)[..., np.newaxis] / 255.0

    assert face.shape == (*IMAGE_SIZE, CHANNELS)
    assert face.dtype == np.float32
    assert 0.0 <= face.min() <= face.max() <= 1.0
    return face


def _load_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge un dossier train ou test organisé par classe."""
    faces: list[np.ndarray] = []
    labels: list[int] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dossier de classe absent : {class_dir}")
        for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
            with Image.open(image_path) as image:
                faces.append(preprocess_face(image))
            labels.append(label)

    if not faces:
        raise ValueError(f"Aucune image trouvée dans : {split_dir}")
    X = np.stack(faces).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    assert X.shape == (len(y), *IMAGE_SIZE, CHANNELS)
    assert X.dtype == np.float32
    assert 0.0 <= X.min() <= X.max() <= 1.0
    assert y.shape == (len(y),)
    return X, y


def load_dataset(
    data_dir: str | Path,
    validation_size: float = VALIDATION_SIZE,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge FER2013 et crée une validation stratifiée depuis le train."""
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size doit être compris entre 0 et 1.")

    root = Path(data_dir)
    X_train_full, y_train_full = _load_split(root / "train")
    X_test, y_test = _load_split(root / "test")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=seed,
        stratify=y_train_full,
    )

    for X, y in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        assert X.shape == (len(y), *IMAGE_SIZE, CHANNELS)
        assert X.dtype == np.float32
        assert 0.0 <= X.min() <= X.max() <= 1.0
        assert y.shape == (len(y),)
    return X_train, y_train, X_val, y_val, X_test, y_test
