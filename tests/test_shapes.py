from pathlib import Path
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.data import CHANNELS, CLASS_NAMES, IMAGE_SIZE, K, SEED, load_dataset, preprocess_face
from src.models import build_cnn, build_mlp
from src import evaluate as evaluation
from src import train as mlp_training


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


def test_mlp_shapes_and_probabilities() -> None:
    """Le MLP accepte les images et produit une distribution finie par classe."""
    model = build_mlp()
    batch = np.random.default_rng(SEED).random((4, 48, 48, 1), dtype=np.float32)
    probabilities = model(batch, training=False).numpy()

    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, K)
    assert K == len(CLASS_NAMES) == model.layers[-1].units
    assert model.layers[1].units == 128
    assert model.layers[0].__class__.__name__ == "Flatten"
    assert model.layers[1].activation.__name__ == "relu"
    assert model.layers[-1].activation.__name__ == "softmax"
    assert model.count_params() == 295943
    assert probabilities.shape == (len(batch), K)
    assert np.isfinite(probabilities).all()
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_cnn_shapes_and_probabilities() -> None:
    """Le CNN C0 enchaîne trois blocs conv/pooling et produit une distribution."""
    model = build_cnn()
    batch = np.random.default_rng(SEED).random((4, 48, 48, 1), dtype=np.float32)
    probabilities = model(batch, training=False).numpy()

    names = [layer.__class__.__name__ for layer in model.layers]
    assert names == ["Conv2D", "MaxPooling2D"] * 3 + ["Flatten", "Dense", "Dense"]
    assert model.layers[5].output.shape == (None, 6, 6, 128)
    assert model.layers[-1].activation.__name__ == "softmax"
    assert model.count_params() == 683527
    assert probabilities.shape == (len(batch), K)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="deux blocs"):
        build_cnn(filters=(32,))


def test_experiment_reexecution_preserves_other_rows(tmp_path: Path) -> None:
    """Un CSV temporaire conserve C0 et une seule ligne B0 après réexécution."""
    path = tmp_path / "logs/experiments.csv"
    row = dict(zip(mlp_training.EXPERIMENT_FIELDS, ("C0", "test temporaire", 0.2, 2.0, 10, "test")))
    mlp_training._write_experiment(path, row)
    with path.open(newline="", encoding="utf-8") as stream:
        other = next(csv.DictReader(stream))
    row = {**row, "id": "B0", "params": 295943}
    mlp_training._write_experiment(path, row)
    mlp_training._write_experiment(path, {**row, "val_acc": 0.3})
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0] == other
    assert len(rows) == 2 and rows[1]["id"] == "B0"
    assert rows[1]["val_acc"] == "0.3"


@pytest.mark.parametrize("contents", [
    "id,val_acc\nB0,0.1\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,incomplet\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,0.1,2,10,test,en_trop\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,0.1,2,10,\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,NaN,2,10,test\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,1.1,2,10,test\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,0.1,inf,10,test\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,0.1,-1,10,test\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,0.1,2,invalide,test\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,0.1,2,0,test\n",
    ",".join(mlp_training.EXPERIMENT_FIELDS) + "\nC0,test,0.1,2,10,test\nC0,test,0.2,2,10,test\n",
])
def test_invalid_csv_fails_before_training(tmp_path: Path, monkeypatch, contents: str) -> None:
    """Un CSV invalide bloque B0 avant fit et conserve les artefacts précédents."""
    logs, checkpoints = tmp_path / "logs", tmp_path / "checkpoints"
    logs.mkdir()
    checkpoints.mkdir()
    paths = (logs / "experiments.csv", logs / "B0_history.json", checkpoints / "B0.keras")
    for path, content in zip(paths, (contents, "historique précédent", "poids précédents")):
        path.write_text(content, encoding="utf-8")
    previous = [path.read_bytes() for path in paths]

    def unexpected_build(**kwargs):
        """Interdit toute construction et tout entraînement dans ce test d'échec."""
        pytest.fail("Le CSV devait être vérifié avant la construction du modèle.")

    monkeypatch.setattr(mlp_training, "build_mlp", unexpected_build)
    images = np.zeros((4, 48, 48, 1), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match="experiments.csv"):
        mlp_training.train_mlp(images, labels, images, labels, log_dir=logs, checkpoint_dir=checkpoints)
    assert [path.read_bytes() for path in paths] == previous


@pytest.mark.parametrize("run_id", ["C0", "E1", "E2", "E3"])
def test_cnn_best_epoch_after_early_stopping(tmp_path: Path, monkeypatch, run_id) -> None:
    """Simule un arrêt anticipé sans entraînement et vérifie poids et métriques associés."""
    logs, checkpoints = tmp_path / "logs", tmp_path / "checkpoints"
    csv_path = logs / "experiments.csv"
    baseline = dict(zip(mlp_training.EXPERIMENT_FIELDS, ("B0", "baseline", 0.3, 1.8, 295943, "préservé")))
    mlp_training._write_experiment(csv_path, baseline)
    previous_csv = csv_path.read_bytes()
    experiment = mlp_training.PHASE6_EXPERIMENTS.get(run_id)
    architecture = mlp_training.C0_ARCHITECTURE if experiment is None else experiment["architecture"]
    learning_rate = 1e-3 if experiment is None else experiment["learning_rate"]
    recorded = []
    monkeypatch.setattr(mlp_training, "_write_experiment", lambda path, row: recorded.append(row))

    def simulated_fit(model, train, *, validation_data, epochs, shuffle, callbacks, verbose):
        """Fournit des logs contrôlés aux vrais callbacks, sans calcul de gradient."""
        assert epochs == 30 and not shuffle
        checkpoint, stopping = callbacks
        assert checkpoint.monitor == stopping.monitor == "val_loss"
        assert checkpoint.mode == stopping.mode == "min"
        assert checkpoint.save_best_only and stopping.restore_best_weights
        assert stopping.patience == 5
        assert model.loss == "sparse_categorical_crossentropy"
        np.testing.assert_allclose(float(model.optimizer.learning_rate.numpy()), learning_rate)
        model.stop_training = False
        for callback in callbacks:
            callback.set_model(model)
            callback.set_params({"epochs": epochs, "verbose": 0})
            callback.on_train_begin()
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        # Le plateau conserve la première meilleure epoch ; accuracy culmine ensuite.
        for epoch, val_loss in enumerate([2.0, 1.0, 1.0, 1.2, 1.3, 1.4, 1.5]):
            model.layers[-1].bias.assign(np.full(K, epoch, dtype=np.float32))
            values = {"loss": 2.0, "accuracy": 0.2, "val_loss": val_loss, "val_accuracy": (epoch + 1) / 10}
            for key, value in values.items():
                history[key].append(value)
            for callback in callbacks:
                callback.on_epoch_end(epoch, values)
            if model.stop_training:
                break
        assert model.stop_training
        for callback in callbacks:
            callback.on_train_end()
        return SimpleNamespace(history=history)

    monkeypatch.setattr(mlp_training.tf.keras.Model, "fit", simulated_fit)
    images = np.zeros((4, *mlp_training.INPUT_SHAPE), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)
    model, history = mlp_training.train_cnn(
        lambda: build_cnn(**architecture), images, labels, images, labels,
        run_id=run_id, learning_rate=learning_rate, log_dir=logs, checkpoint_dir=checkpoints, verbose=0,
    )
    payload = json.loads((logs / f"{run_id}_history.json").read_text(encoding="utf-8"))
    assert payload["best_epoch"] == 2
    assert payload["config"]["epochs_ran"] == 7 < payload["config"]["epochs"] == 30
    assert payload["history"] == history
    assert payload["best_metrics"]["val_loss"] == recorded[0]["val_loss"] == 1.0
    assert payload["best_metrics"]["val_accuracy"] == recorded[0]["val_acc"] == 0.2
    assert recorded[0]["id"] == payload["id"] == run_id
    assert recorded[0]["params"] == build_cnn(**architecture).count_params()
    assert "Meilleure epoch 2/7" in recorded[0]["observation"]
    restored = mlp_training.tf.keras.models.load_model(checkpoints / f"{run_id}.keras", compile=False)
    for saved, returned in zip(restored.get_weights(), model.get_weights()):
        np.testing.assert_array_equal(saved, returned)
    np.testing.assert_array_equal(restored.layers[-1].bias.numpy(), np.ones(K))
    assert csv_path.read_bytes() == previous_csv  # Aucune ligne C0 issue de cette simulation.


def test_cnn_rejects_wrong_class_count(tmp_path: Path) -> None:
    """Refuse une fabrique incompatible avant fit et sauvegarde."""
    def wrong_factory():
        """Construit seulement un modèle invalide pour vérifier le contrat de classes."""
        return mlp_training.tf.keras.Sequential([
            mlp_training.tf.keras.Input(shape=mlp_training.INPUT_SHAPE),
            mlp_training.tf.keras.layers.Flatten(),
            mlp_training.tf.keras.layers.Dense(K - 1, activation="softmax"),
        ])

    images = np.zeros((4, *mlp_training.INPUT_SHAPE), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)
    with pytest.raises(AssertionError):
        mlp_training.train_cnn(
            wrong_factory, images, labels, images, labels,
            log_dir=tmp_path / "logs", checkpoint_dir=tmp_path / "checkpoints", smoke=True,
        )
    assert not list(tmp_path.iterdir())


def test_evaluation_and_single_test_guard(tmp_path: Path) -> None:
    """Évalue un modèle sauvegardé et refuse une seconde évaluation du test."""
    model = build_mlp()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.save(tmp_path / "M.keras")
    X = np.random.default_rng(SEED).random((14, 48, 48, 1), dtype=np.float32)
    y = np.arange(14, dtype=np.int64) % K

    result = evaluation.evaluate(model, X, y)
    assert result["y_pred"].shape == result["confidence"].shape == (14,)
    assert np.isclose(result["accuracy"], np.mean(result["y_pred"] == y))
    assert list(evaluation.class_report(y, result["y_pred"]).index[:K]) == list(CLASS_NAMES)

    y_pred, confidence = np.array([0, 1, 1, 2]), np.array([0.5, 0.9, 0.7, 0.8])
    assert evaluation.confident_examples(np.array([0, 1, 2, 2]), y_pred, confidence, True).tolist() == [1, 3, 0]
    assert evaluation.confident_examples(np.array([0, 1, 2, 2]), y_pred, confidence, False).tolist() == [2]

    raw = [np.full((40, 40), 128, dtype=np.uint8), np.zeros((60, 60, 3), dtype=np.uint8)]
    assert evaluation.predict_faces(model, raw).shape == (2, K)

    first = evaluation.evaluate_test_once("M", X, y, log_dir=tmp_path, checkpoint_dir=tmp_path)
    assert evaluation.load_test_evaluation(tmp_path) == first
    with pytest.raises(FileExistsError):
        evaluation.evaluate_test_once("M", X, y, log_dir=tmp_path, checkpoint_dir=tmp_path)


def test_augmented_batches() -> None:
    """L'augmentation A1 garde forme et plage, modifie les images et reste reproductible."""
    X = np.random.default_rng(SEED).random((16, 48, 48, 1), dtype=np.float32)
    y = np.arange(16, dtype=np.int64) % K
    plain = next(iter(mlp_training._batches(X, y, 16, SEED, shuffle=False)))[0].numpy()
    first, second = (
        next(iter(mlp_training._batches(X, y, 16, SEED, shuffle=False, augment=True)))[0].numpy()
        for _ in range(2)
    )
    assert first.shape == plain.shape == (16, 48, 48, 1)
    assert 0.0 <= first.min() <= first.max() <= 1.0
    assert not np.allclose(first, plain)
    np.testing.assert_array_equal(first, second)
