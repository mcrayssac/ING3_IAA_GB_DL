"""Entraînement court du MLP, distinct du futur protocole CNN."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data import CLASS_NAMES, SEED, load_dataset
from src.models import HIDDEN_UNITS, INPUT_SHAPE, K, build_mlp


LEARNING_RATE = 1e-3
BATCH_SIZE = 64
B0_EPOCHS = 5
SMOKE_TRAIN_SIZE = 256
SMOKE_VAL_SIZE = 64
LOG_DIR = Path("training/logs")
CHECKPOINT_DIR = Path("training/checkpoints")
EXPERIMENT_FIELDS = ("id", "modification", "val_acc", "val_loss", "params", "observation")


def _check_data(X: np.ndarray, y: np.ndarray) -> None:
    """Vérifie le contrat des données déjà prétraitées par le chargeur."""
    assert X.shape == (len(y), *INPUT_SHAPE) and len(y) > 0
    assert X.dtype == np.float32 and np.isfinite(X).all()
    assert 0.0 <= X.min() <= X.max() <= 1.0
    assert y.shape == (len(y),) and np.issubdtype(y.dtype, np.integer)
    assert np.all((0 <= y) & (y < K))


def _batches(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool):
    """Forme des batches sans appliquer une seconde normalisation."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    dataset = dataset.with_options(options)
    images, labels = next(iter(dataset))
    assert images.shape == (len(labels), *INPUT_SHAPE)
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


def main() -> None:
    """Lance uniquement le smoke test local sur le split de phase 1."""
    parser = argparse.ArgumentParser(description="Smoke test MLP : 256 train, 64 val, 1 epoch.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    X_train, y_train, X_val, y_val, _, _ = load_dataset(args.data_dir)
    train_mlp(X_train, y_train, X_val, y_val, smoke=True)
    print("Smoke test terminé ; smoke_history.json et smoke.keras sauvegardés, aucune ligne B0.")


if __name__ == "__main__":
    main()
