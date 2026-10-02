from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import CHANNELS, CLASS_NAMES, IMAGE_SIZE, K, SEED, load_dataset, preprocess_face


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, dict[int, int], dict[int, int]]:
    """Crée 70 images train et 14 images test aux pixels tous distincts."""
    examples = {"train": {}, "test": {}}
    for split, count, offset in (("train", 10, 1), ("test", 2, 101)):
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            for index in range(count):
                pixel = offset + label * count + index
                examples[split][pixel] = label
                pixels = np.full((40, 40), pixel, dtype=np.uint8)
                Image.fromarray(pixels).save(class_dir / f"{index}.png")
    return tmp_path, examples["train"], examples["test"]


def _image_ids(X: np.ndarray) -> np.ndarray:
    """Retrouve l'identifiant de chaque image synthétique normalisée."""
    return np.rint(X[:, 0, 0, 0] * 255.0).astype(np.int64)


def _assert_examples(X: np.ndarray, y: np.ndarray, expected: dict[int, int]) -> None:
    """Vérifie tous les exemples, leurs labels et leurs pixels attendus."""
    ids = _image_ids(X)
    assert len(ids) == len(set(ids)) == len(expected)
    assert set(ids) == set(expected)
    for face, label, pixel in zip(X, y, ids):
        assert label == expected[pixel]
        expected_face = np.full((48, 48, 1), np.float32(pixel) / 255.0, dtype=np.float32)
        np.testing.assert_array_equal(face, expected_face)


def test_load_dataset_shapes_and_classes(dataset) -> None:
    """Le chargement produit les shapes et classes attendues."""
    root, _, _ = dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(root)

    assert CLASS_NAMES == ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
    assert K == len(CLASS_NAMES) == 7
    assert IMAGE_SIZE == (48, 48)
    assert CHANNELS == 1
    assert (len(X_train), len(X_val), len(X_test)) == (59, 11, 14)
    for X, y in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        assert X.shape == (len(y), *IMAGE_SIZE, CHANNELS)
        assert X.dtype == np.float32
        assert 0.0 <= X.min() <= X.max() <= 1.0
        assert y.shape == (len(y),)
        assert y.dtype == np.int64
        assert np.all((0 <= y) & (y < K))
    assert set(y_train) == set(y_val) == set(y_test) == set(range(K))


def test_validation_is_stratified(dataset) -> None:
    """Les onze images de validation représentent chacune des sept classes."""
    root, _, _ = dataset
    _, y_train, _, y_val, _, _ = load_dataset(root)
    train_counts = np.bincount(y_train, minlength=K)
    val_counts = np.bincount(y_val, minlength=K)

    assert sorted(val_counts) == [1, 1, 1, 2, 2, 2, 2]
    np.testing.assert_array_equal(train_counts + val_counts, np.full(K, 10))


def test_split_is_reproducible(dataset) -> None:
    """La seed 42 reproduit les images et les labels de chaque sortie."""
    root, _, _ = dataset
    assert SEED == 42
    first = load_dataset(root)
    second = load_dataset(root, seed=42)

    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)


def test_train_validation_preserve_original_examples(dataset) -> None:
    """Train et validation sont disjoints et conservent les 70 originaux."""
    root, train_examples, _ = dataset
    X_train, y_train, X_val, y_val, _, _ = load_dataset(root)

    assert set(_image_ids(X_train)).isdisjoint(_image_ids(X_val))
    _assert_examples(
        np.concatenate((X_train, X_val)),
        np.concatenate((y_train, y_val)),
        train_examples,
    )


def test_official_test_is_preserved_and_independent_of_seed(dataset) -> None:
    """Le test conserve exclusivement ses 14 exemples quelle que soit la seed."""
    root, train_examples, test_examples = dataset
    first = load_dataset(root, seed=42)
    second = load_dataset(root, seed=123)

    assert not np.array_equal(first[0], second[0])
    for result in (first, second):
        X_test, y_test = result[4:]
        _assert_examples(X_test, y_test, test_examples)
        assert set(_image_ids(X_test)).isdisjoint(train_examples)
    np.testing.assert_array_equal(first[4], second[4])
    np.testing.assert_array_equal(first[5], second[5])


@pytest.mark.parametrize("image_format", ["gray", "rgb", "normalized"])
def test_preprocess_face_formats(image_format: str) -> None:
    """Le prétraitement accepte gris, RGB et gris déjà normalisé."""
    image = np.full((40, 40), 128, dtype=np.uint8)
    if image_format == "rgb":
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    elif image_format == "normalized":
        image = image.astype(np.float32)[..., np.newaxis] / 255.0

    face = preprocess_face(image)

    assert face.shape == (48, 48, 1)
    assert face.dtype == np.float32
    np.testing.assert_array_equal(face, np.full((48, 48, 1), np.float32(128) / 255.0))


@pytest.mark.parametrize("split", ["train", "test"])
def test_empty_split_raises_value_error(dataset, split: str) -> None:
    """Un dossier sans image conserve l'erreur explicite prévue au chargement."""
    root, _, _ = dataset
    for image_path in (root / split).glob("*/*.png"):
        image_path.unlink()

    with pytest.raises(ValueError, match="Aucune image trouvée"):
        load_dataset(root)
