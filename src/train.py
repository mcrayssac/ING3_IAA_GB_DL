"""Entraînement B0/C0/E1–E3 et smoke tests locaux."""

import argparse
import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

from src.data import CLASS_NAMES, SEED, load_dataset
from src.models import HIDDEN_UNITS, INPUT_SHAPE, K, build_cnn, build_mlp


LEARNING_RATE = 1e-3
BATCH_SIZE = 64
B0_EPOCHS = 5
CNN_MAX_EPOCHS = 30
CNN_PATIENCE = 5
SMOKE_TRAIN_SIZE = 256
SMOKE_VAL_SIZE = 64
LOG_DIR = Path("training/logs")
CHECKPOINT_DIR = Path("training/checkpoints")
EXPERIMENT_FIELDS = ("id", "modification", "val_acc", "val_loss", "params", "observation")
METRIC_NAMES = ("loss", "accuracy", "val_loss", "val_accuracy")
FULL_TRAIN_SIZE = 24402
FULL_VAL_SIZE = 4307
RELOAD_RTOL = 1e-5
RELOAD_ATOL = 1e-6
C0_ARCHITECTURE = dict(filters=(32, 64, 128), kernel_size=3, dense_units=128, seed=SEED, dropout_rate=0.0)
PHASE6_EXPERIMENTS = {
    "E1": dict(architecture={**C0_ARCHITECTURE, "dense_units": 64}, learning_rate=LEARNING_RATE,
               modification="C0 : Dense 128 → 64 uniquement"),
    "E2": dict(architecture={**C0_ARCHITECTURE, "dropout_rate": 0.3}, learning_rate=LEARNING_RATE,
               modification="C0 : Dropout 0,3 après Dense uniquement, sans augmentation"),
    "E3": dict(architecture=dict(C0_ARCHITECTURE), learning_rate=5e-4,
               modification="C0 : learning rate 0,001 → 0,0005 uniquement"),
    # Phase 7 : même protocole que E3, seule l'augmentation du train change.
    "A1": dict(architecture=dict(C0_ARCHITECTURE), learning_rate=5e-4, augmentation=True,
               modification="E3 : data augmentation uniquement (flip, rotation, translation, zoom, contraste)"),
}
SELECTION_CRITERION = "val_loss minimale du checkpoint ; puis val_accuracy maximale ; puis id"


def _validation_digest(X: np.ndarray, y: np.ndarray) -> str:
    """Identifie exactement les images et labels de validation dans leur ordre."""
    digest = hashlib.sha256(X.tobytes())
    digest.update(y.tobytes())
    return digest.hexdigest()


def _check_data(X: np.ndarray, y: np.ndarray) -> None:
    """Vérifie le contrat des données déjà prétraitées par le chargeur."""
    assert X.shape == (len(y), *INPUT_SHAPE) and len(y) > 0
    assert X.dtype == np.float32 and np.isfinite(X).all()
    assert 0.0 <= X.min() <= X.max() <= 1.0
    assert y.shape == (len(y),) and np.issubdtype(y.dtype, np.integer)
    assert np.all((0 <= y) & (y < K))


def _augmentation(seed: int = SEED) -> tf.keras.Sequential:
    """Transformations aléatoires du train A1 ; inactives hors de training=True."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, seed=seed),  # ±18°
        tf.keras.layers.RandomTranslation(0.15, 0.15, seed=seed),
        tf.keras.layers.RandomZoom(0.15, seed=seed),
        tf.keras.layers.RandomContrast(0.2, value_range=(0, 1), seed=seed),
    ], name="augmentation")


def _batches(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool, augment: bool = False):
    """Forme des batches sans seconde normalisation ; augmente seulement si demandé."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    if augment:
        augmentation = _augmentation(seed)
        dataset = dataset.map(lambda images, labels: (augmentation(images, training=True), labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    dataset = dataset.with_options(options)
    images, labels = next(iter(dataset))
    assert images.shape == (len(labels), *INPUT_SHAPE)
    assert 0.0 <= float(tf.reduce_min(images)) <= float(tf.reduce_max(images)) <= 1.0
    return dataset


def _read_experiments(path: Path) -> list[dict]:
    """Lit le CSV et refuse les lignes incomplètes ou les résultats invalides."""
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(EXPERIMENT_FIELDS):
                raise ValueError(f"Schéma experiments.csv inattendu : {reader.fieldnames}")
            rows = list(reader)
            if any(None in row or any(value is None or not value.strip() for value in row.values()) for row in rows):
                raise ValueError("Ligne experiments.csv incomplète ou mal formée.")
            for row in rows:
                try:
                    accuracy, loss = float(row["val_acc"]), float(row["val_loss"])
                    params = int(row["params"])
                except ValueError as error:
                    raise ValueError("Valeur numérique experiments.csv invalide.") from error
                if not (np.isfinite(accuracy) and np.isfinite(loss) and 0 <= accuracy <= 1 and loss >= 0 and params > 0):
                    raise ValueError("Métrique ou nombre de paramètres experiments.csv invalide.")
            if len({row["id"] for row in rows}) != len(rows):
                raise ValueError("Identifiant experiments.csv dupliqué.")
    return rows


def _write_experiment(path: Path, row: dict) -> None:
    """Remplace un identifiant en conservant les autres expériences du CSV."""
    rows = [existing for existing in _read_experiments(path) if existing["id"] != row["id"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPERIMENT_FIELDS)
        writer.writeheader()
        writer.writerows([*rows, row])
    temporary.replace(path)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = B0_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    hidden_units: int = HIDDEN_UNITS,
    seed: int = SEED,
    smoke: bool = False,
    log_dir: str | Path = LOG_DIR,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    verbose: int = 2,
) -> tuple[tf.keras.Model, dict[str, list[float]]]:
    """Entraîne un MLP neuf et sauvegarde soit le smoke test, soit B0."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size et learning_rate doivent être positifs.")
    if smoke:
        X_train, y_train = X_train[:SMOKE_TRAIN_SIZE], y_train[:SMOKE_TRAIN_SIZE]
        X_val, y_val = X_val[:SMOKE_VAL_SIZE], y_val[:SMOKE_VAL_SIZE]
        epochs = 1
    for X, y in ((X_train, y_train), (X_val, y_val)):
        _check_data(X, y)
    log_dir, checkpoint_dir = Path(log_dir), Path(checkpoint_dir)
    if not smoke:
        _read_experiments(log_dir / "experiments.csv")  # Refuser avant fit et toute sauvegarde.

    model = build_mlp(hidden_units=hidden_units, seed=seed)
    tf.config.experimental.enable_op_determinism()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    predictions = model(X_train[:batch_size], training=False).numpy()
    assert predictions.shape == (min(batch_size, len(y_train)), K)
    assert np.isfinite(predictions).all()
    assert K == len(CLASS_NAMES) == model.layers[-1].units
    result = model.fit(
        _batches(X_train, y_train, batch_size, seed, shuffle=True),
        validation_data=_batches(X_val, y_val, batch_size, seed, shuffle=False),
        epochs=epochs,
        shuffle=False,  # Le dataset train mélange déjà les images avec la seed.
        verbose=verbose,
    )
    history = {key: [float(value) for value in values] for key, values in result.history.items()}
    assert all(len(values) == epochs and np.isfinite(values).all() for values in history.values())
    run_id = "smoke" if smoke else "B0"
    config = {
        "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "hidden_units": hidden_units,
        "train_size": len(y_train), "val_size": len(y_val),
        "optimizer": "Adam", "loss": "sparse_categorical_crossentropy",
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_dir / f"{run_id}.keras")
    payload = {"id": run_id, "config": config, "history": history}
    (log_dir / f"{run_id}_history.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not smoke:
        _write_experiment(log_dir / "experiments.csv", {
            "id": "B0", "modification": "Baseline MLP Flatten -> Dense ReLU -> Dense softmax",
            "val_acc": history["val_accuracy"][-1], "val_loss": history["val_loss"][-1],
            "params": model.count_params(),
            "observation": (
                f"Dernière epoch {epochs}/{epochs} ; accuracy et loss de la même epoch ; "
                "poids finaux, aucune restauration des meilleurs poids ; "
                + json.dumps(config, ensure_ascii=False)
            ),
        })
    return model, history


def train_cnn(
    model_factory: Callable[[], tf.keras.Model],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = CNN_MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
    smoke: bool = False,
    log_dir: str | Path = LOG_DIR,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    verbose: int = 2,
    run_id: str = "C0",
    modification: str = "CNN de départ",
    augmentation: bool = False,
) -> tuple[tf.keras.Model, dict[str, list[float]]]:
    """Applique le protocole CNN à un modèle neuf fourni par une fabrique sans argument."""
    if run_id not in ("C0", *PHASE6_EXPERIMENTS):
        raise ValueError("Identifiant CNN attendu : C0, E1, E2, E3 ou A1.")
    if not 1 <= epochs <= CNN_MAX_EPOCHS or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs doit être dans [1,30], batch_size et learning_rate positifs.")
    if smoke:
        X_train, y_train = X_train[:SMOKE_TRAIN_SIZE], y_train[:SMOKE_TRAIN_SIZE]
        X_val, y_val = X_val[:SMOKE_VAL_SIZE], y_val[:SMOKE_VAL_SIZE]
        epochs = 1
    for X, y in ((X_train, y_train), (X_val, y_val)):
        _check_data(X, y)
    log_dir, checkpoint_dir = Path(log_dir), Path(checkpoint_dir)
    if not smoke:
        _read_experiments(log_dir / "experiments.csv")

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = model_factory()
    assert model.input_shape == (None, *INPUT_SHAPE)
    assert model.output_shape == (None, K)
    assert K == len(CLASS_NAMES) == model.layers[-1].units
    assert model.layers[-1].activation.__name__ == "softmax"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    prediction_tensor = model(X_train[:batch_size], training=False)
    predictions = prediction_tensor.numpy()
    assert predictions.shape == (min(batch_size, len(y_train)), K)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-6)
    run_id = "smoke" if smoke else run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_dir / f"{run_id}.keras", monitor="val_loss", mode="min", save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=CNN_PATIENCE, restore_best_weights=True,
        ),
    ]
    result = model.fit(
        _batches(X_train, y_train, batch_size, seed, shuffle=True, augment=augmentation),
        validation_data=_batches(X_val, y_val, batch_size, seed, shuffle=False),
        epochs=epochs, shuffle=False, callbacks=callbacks, verbose=verbose,
    )
    history = {key: [float(value) for value in values] for key, values in result.history.items()}
    epochs_ran = len(history["val_loss"])
    assert 1 <= epochs_ran <= epochs
    assert all(len(values) == epochs_ran and np.isfinite(values).all() for values in history.values())
    best_index = int(np.argmin(history["val_loss"]))
    best_metrics = {key: values[best_index] for key, values in history.items()}
    config = {
        "seed": seed, "epochs": epochs, "epochs_ran": epochs_ran, "batch_size": batch_size,
        "learning_rate": learning_rate, "train_size": len(y_train), "val_size": len(y_val),
        "model_name": model.name, "class_names": list(CLASS_NAMES),
        "optimizer": "Adam", "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"],
        "monitor": "val_loss", "mode": "min", "save_best_only": True,
        "patience": CNN_PATIENCE, "restore_best_weights": True,
        "validation_sha256": _validation_digest(X_val, y_val),
        "tensorflow_version": tf.__version__, "keras_version": tf.keras.__version__,
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "prediction_device": prediction_tensor.device,
        "train_sha256": _validation_digest(X_train, y_train),
        "augmentation": augmentation,
        "architecture": {
            "filters": [layer.filters for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)],
            "kernel_size": list(next(layer.kernel_size for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D))),
            "dense_units": model.layers[-3].units if isinstance(model.layers[-2], tf.keras.layers.Dropout) else model.layers[-2].units,
            "dropout_rate": next((layer.rate for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)), 0.0),
            "seed": seed,
        },
        "params": model.count_params(),
    }
    payload = {
        "id": run_id, "config": config, "history": history,
        "best_epoch": best_index + 1, "best_metrics": best_metrics,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{run_id}_history.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not smoke:
        _write_experiment(log_dir / "experiments.csv", {
            "id": run_id, "modification": modification,
            "val_acc": best_metrics["val_accuracy"], "val_loss": best_metrics["val_loss"],
            "params": model.count_params(),
            "observation": (
                f"Meilleure epoch {best_index + 1}/{epochs_ran} sur val_loss ; maximum {epochs} ; "
                "métriques de la même epoch, meilleurs poids restaurés ; "
                + json.dumps(config, ensure_ascii=False)
            ),
        })
    return model, history


def train_experiment(run_id, X_train, y_train, X_val, y_val, *, smoke=False,
                     log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR, verbose=2):
    """Exécute une variante isolée de C0, complète sur Colab GPU ou smoke temporaire."""
    experiment = PHASE6_EXPERIMENTS[run_id]
    if not smoke:
        import google.colab
        assert tf.config.list_physical_devices("GPU"), "Runtime Colab GPU requis."
        assert len(y_train) == FULL_TRAIN_SIZE and len(y_val) == FULL_VAL_SIZE
        reference = json.loads((Path(log_dir) / "C0_history.json").read_text())
        assert _validation_digest(X_val, y_val) == reference["config"]["validation_sha256"]
    return train_cnn(
        lambda: build_cnn(**experiment["architecture"]), X_train, y_train, X_val, y_val,
        epochs=CNN_MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=experiment["learning_rate"], seed=SEED,
        smoke=smoke, run_id=run_id, modification=experiment["modification"],
        augmentation=experiment.get("augmentation", False),
        log_dir=log_dir, checkpoint_dir=checkpoint_dir, verbose=verbose,
    )


def verify_cnn_run(
    X_val: np.ndarray, y_val: np.ndarray, *,
    log_dir: str | Path = LOG_DIR, checkpoint_dir: str | Path = CHECKPOINT_DIR,
    run_id: str = "C0",
    X_train: np.ndarray | None = None, y_train: np.ndarray | None = None,
) -> dict:
    """Contrôle un run CNN réel et évalue seulement sa validation identifiée par hash."""
    _check_data(X_val, y_val)
    payload = json.loads((Path(log_dir) / f"{run_id}_history.json").read_text(encoding="utf-8"))
    config, history = payload["config"], payload["history"]
    assert payload["id"] == run_id and set(history) == set(METRIC_NAMES)
    experiment = PHASE6_EXPERIMENTS.get(run_id)
    learning_rate = LEARNING_RATE if experiment is None else experiment["learning_rate"]
    expected = {
        "seed": SEED, "epochs": CNN_MAX_EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate, "train_size": FULL_TRAIN_SIZE, "val_size": FULL_VAL_SIZE,
        "optimizer": "Adam", "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"],
        "class_names": list(CLASS_NAMES), "monitor": "val_loss", "mode": "min",
        "save_best_only": True, "patience": CNN_PATIENCE, "restore_best_weights": True,
    }
    assert all(config[key] == value for key, value in expected.items())
    assert config["gpu_devices"] and "GPU:" in config["prediction_device"]
    assert len(y_val) == FULL_VAL_SIZE and _validation_digest(X_val, y_val) == config["validation_sha256"]
    if experiment is not None:
        reference = json.loads((Path(log_dir) / "C0_history.json").read_text())
        assert config["validation_sha256"] == reference["config"]["validation_sha256"]
        assert config["augmentation"] is experiment.get("augmentation", False)
        assert X_train is not None and y_train is not None
        _check_data(X_train, y_train)
        assert len(y_train) == FULL_TRAIN_SIZE
        assert config["train_sha256"] == _validation_digest(X_train, y_train)
        architecture = experiment["architecture"]
        assert config["architecture"] == {**architecture, "filters": list(architecture["filters"]),
                                          "kernel_size": [architecture["kernel_size"]] * 2}
    epochs_ran = len(history["val_loss"])
    assert 1 <= epochs_ran == config["epochs_ran"] <= CNN_MAX_EPOCHS
    for key, values in history.items():
        assert len(values) == epochs_ran and np.isfinite(values).all()
        assert np.all(np.asarray(values) >= 0)
        if "accuracy" in key:
            assert np.all(np.asarray(values) <= 1)
    best_index = int(np.argmin(history["val_loss"]))
    assert payload["best_epoch"] == best_index + 1
    assert payload["best_metrics"] == {key: values[best_index] for key, values in history.items()}
    if epochs_ran < CNN_MAX_EPOCHS:
        assert epochs_ran - (best_index + 1) == CNN_PATIENCE
    rows = _read_experiments(Path(log_dir) / "experiments.csv")
    trace = json.loads((Path(log_dir) / "C0_colab_environment.txt").read_text(encoding="utf-8"))
    assert trace["gpu_devices"] and trace["test_used"] is False
    assert trace["train_size"] == FULL_TRAIN_SIZE and trace["val_size"] == FULL_VAL_SIZE
    assert trace["versions"]["tensorflow"] == config["tensorflow_version"]
    assert trace["versions"]["keras"] == config["keras_version"]
    assert all(row in rows for row in trace["preserved_rows"])
    project_root = Path(__file__).resolve().parents[1]
    for relative, digest in trace["preserved_artifacts_sha256"].items():
        assert hashlib.sha256((project_root / relative).read_bytes()).hexdigest() == digest
    local_rows = _read_experiments(project_root / LOG_DIR / "experiments.csv")
    assert all(row in local_rows for row in trace["preserved_rows"])
    row = next(row for row in rows if row["id"] == run_id)
    if experiment is not None:
        assert row["modification"] == experiment["modification"]
    for field, metric in (("val_acc", "val_accuracy"), ("val_loss", "val_loss")):
        np.testing.assert_allclose(float(row[field]), history[metric][best_index], rtol=1e-7, atol=1e-8)
    restored = tf.keras.models.load_model(Path(checkpoint_dir) / f"{run_id}.keras")
    expected_model = build_cnn(**(C0_ARCHITECTURE if experiment is None else experiment["architecture"]))
    assert restored.input_shape == (None, *INPUT_SHAPE) and restored.output_shape == (None, K)
    assert K == len(CLASS_NAMES) == restored.layers[-1].units
    assert restored.count_params() == int(row["params"]) == expected_model.count_params()
    assert len(restored.layers) == len(expected_model.layers)
    for saved, expected_layer in zip(restored.layers, expected_model.layers):
        assert type(saved) is type(expected_layer)
        assert {key: value for key, value in saved.get_config().items() if key != "name"} == {
            key: value for key, value in expected_layer.get_config().items() if key != "name"
        }
    assert restored.loss == expected["loss"] and isinstance(restored.optimizer, tf.keras.optimizers.Adam)
    np.testing.assert_allclose(float(restored.optimizer.learning_rate.numpy()), learning_rate)
    assert int(restored.optimizer.iterations.numpy()) == (best_index + 1) * int(np.ceil(FULL_TRAIN_SIZE / BATCH_SIZE))
    predictions = restored(X_val[:BATCH_SIZE], training=False).numpy()
    assert predictions.shape == (BATCH_SIZE, K) and np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-6)
    metrics = restored.evaluate(_batches(X_val, y_val, BATCH_SIZE, SEED, shuffle=False), verbose=0, return_dict=True)
    for key in ("loss", "accuracy"):
        np.testing.assert_allclose(metrics[key], history[f"val_{key}"][best_index], rtol=RELOAD_RTOL, atol=RELOAD_ATOL)
    report = {"epochs_ran": epochs_ran, "best_epoch": best_index + 1, "validation_reloaded": metrics}
    print(f"{run_id} vérifié (validation uniquement) :", report)
    return report


def smoke_cnn(X_train, y_train, X_val, y_val, output_dir: str | Path) -> None:
    """Entraîne le vrai CNN une epoch et contrôle la relecture de ses meilleurs poids."""
    output_dir = Path(output_dir)
    model, history = train_cnn(
        build_cnn, X_train, y_train, X_val, y_val, smoke=True,
        log_dir=output_dir / "logs", checkpoint_dir=output_dir / "checkpoints",
    )
    restored = tf.keras.models.load_model(output_dir / "checkpoints/smoke.keras")
    assert model.count_params() == restored.count_params() == 683527
    assert restored.input_shape == (None, *INPUT_SHAPE) and restored.output_shape == (None, K)
    assert K == len(CLASS_NAMES) == restored.layers[-1].units
    for saved, returned in zip(restored.get_weights(), model.get_weights()):
        np.testing.assert_array_equal(saved, returned)
    np.testing.assert_allclose(restored(X_val[:SMOKE_VAL_SIZE]), model(X_val[:SMOKE_VAL_SIZE]), rtol=1e-7, atol=1e-8)
    payload = json.loads((output_dir / "logs/smoke_history.json").read_text(encoding="utf-8"))
    assert payload["history"] == history and payload["best_epoch"] == 1
    assert payload["config"]["train_size"] <= SMOKE_TRAIN_SIZE and payload["config"]["val_size"] <= SMOKE_VAL_SIZE
    metrics = restored.evaluate(_batches(X_val[:SMOKE_VAL_SIZE], y_val[:SMOKE_VAL_SIZE], BATCH_SIZE, SEED, False), verbose=0, return_dict=True)
    for key in ("loss", "accuracy"):
        np.testing.assert_allclose(metrics[key], history[f"val_{key}"][0], rtol=RELOAD_RTOL, atol=RELOAD_ATOL)
    assert not (output_dir / "logs/experiments.csv").exists()
    print(f"Smoke CNN conforme : 256/64 au maximum, 1 epoch, poids rechargés identiques ; {output_dir}")


def main() -> None:
    """Lance un smoke local ou contrôle C0, sans entraînement complet."""
    parser = argparse.ArgumentParser(description="Smoke MLP/CNN : 256 train, 64 val, 1 epoch ; ou contrôle C0.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--output-dir", type=Path, help="Dossier de sortie du smoke CNN (temporaire par défaut).")
    parser.add_argument("--check-c0", action="store_true", help="Contrôler C0 sans entraînement, sur la validation complète.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("training"))
    args = parser.parse_args()
    X_train, y_train, X_val, y_val, _, _ = load_dataset(args.data_dir)
    if args.check_c0:
        verify_cnn_run(X_val, y_val, log_dir=args.artifact_dir / "logs", checkpoint_dir=args.artifact_dir / "checkpoints")
        return
    if args.model == "cnn":
        if args.output_dir is None:
            with TemporaryDirectory(prefix="fer2013-cnn-smoke-") as directory:
                smoke_cnn(X_train, y_train, X_val, y_val, directory)
        else:
            smoke_cnn(X_train, y_train, X_val, y_val, args.output_dir)
        return
    train_mlp(X_train, y_train, X_val, y_val, smoke=True)
    print("Smoke test terminé ; smoke_history.json et smoke.keras sauvegardés, aucune ligne B0.")


if __name__ == "__main__":
    main()
