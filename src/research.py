"""Approfondissement CNN : échelle d'améliorations, recherche aléatoire, moyenne sur plusieurs seeds."""

import argparse
import json
import platform
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import load_dataset
from src.evaluate import class_report, evaluate
from src.models import INPUT_SHAPE, K
from src.train import LOG_DIR, _batches, _check_data, _validation_digest


RESEARCH_DIR = Path("training/research")
RESEARCH_CHECKPOINT_DIR = Path("training/checkpoints/research")
SEEDS = (42, 43, 44)
SEARCH_SEED = 2026
SELECTION = "val_loss moyenne sur 3 seeds (minimale), puis val_accuracy moyenne (maximale)"
# R0 reprend A1 en local : architecture C0, Adam 0,0005, batch 64, augmentation, 30 epochs, patience 5.
BASE_CONFIG = dict(
    filters=(32, 64, 128), convs_per_block=1, batch_norm=False, conv_dropout=0.0,
    dense_units=128, dense_dropout=0.0, learning_rate=5e-4, batch_size=64,
    max_epochs=30, patience=5, reduce_lr=False, class_weight=False, augmentation=True,
)
LADDER = (
    ("R0", "A1 refait en local (référence même matériel)", {}),
    ("R1", "BatchNorm après chaque convolution", {"batch_norm": True}),
    ("R2", "Deux convolutions par bloc (type VGG)", {"convs_per_block": 2}),
    ("R3", "Dropout 0,25 après chaque bloc et 0,5 avant la sortie", {"conv_dropout": 0.25, "dense_dropout": 0.5}),
    ("R4", "60 epochs, ReduceLROnPlateau, patience 8", {"max_epochs": 60, "patience": 8, "reduce_lr": True}),
    ("R5", "Poids de classes équilibrés", {"class_weight": True}),
)
SEARCH_SPACE = dict(
    batch_size=(32, 64, 128), conv_dropout=(0.0, 0.1, 0.25, 0.4),
    dense_dropout=(0.0, 0.3, 0.5), dense_units=(128, 256),
)  # learning_rate : tirage log-uniforme entre 2e-4 et 2e-3.


def build_research_cnn(config: dict, seed: int) -> tf.keras.Model:
    """CNN paramétrable ; à 1 convolution par bloc, sans BatchNorm ni Dropout, il reprend build_cnn."""
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    stack = [tf.keras.Input(shape=INPUT_SHAPE)]
    for filters in config["filters"]:
        for _ in range(config["convs_per_block"]):
            if config["batch_norm"]:
                stack += [layers.Conv2D(filters, 3, padding="same", use_bias=False),
                          layers.BatchNormalization(), layers.Activation("relu")]
            else:
                stack.append(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        stack.append(layers.MaxPooling2D(2))
        if config["conv_dropout"]:
            stack.append(layers.Dropout(config["conv_dropout"], seed=seed))
    stack += [layers.Flatten(), layers.Dense(config["dense_units"], activation="relu")]
    if config["dense_dropout"]:
        stack.append(layers.Dropout(config["dense_dropout"], seed=seed))
    stack.append(layers.Dense(K, activation="softmax"))
    model = tf.keras.Sequential(stack, name="research_cnn")
    assert model.input_shape == (None, *INPUT_SHAPE) and model.output_shape == (None, K)
    return model


def run_research(run_id: str, label: str, config: dict, seed: int, data: tuple, *,
                 out_dir: str | Path = RESEARCH_DIR, checkpoint_dir: str | Path = RESEARCH_CHECKPOINT_DIR,
                 verbose: int = 2) -> dict:
    """Entraîne une configuration pour une seed, ou relit le résultat déjà enregistré."""
    out_dir, checkpoint_dir = Path(out_dir), Path(checkpoint_dir)
    record_path = out_dir / f"{run_id}_s{seed}.json"
    if record_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))
    X_train, y_train, X_val, y_val = data
    for X, y in ((X_train, y_train), (X_val, y_val)):
        _check_data(X, y)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = build_research_cnn(config, seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=config["patience"], restore_best_weights=True)]
    if config["reduce_lr"]:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-5))
    class_weight = None
    if config["class_weight"]:
        counts = np.bincount(y_train, minlength=K)
        class_weight = {k: float(len(y_train) / (K * counts[k])) for k in range(K)}
    print(f"[{run_id} seed {seed}] {label}", flush=True)
    start = time.perf_counter()
    result = model.fit(
        _batches(X_train, y_train, config["batch_size"], seed, shuffle=True, augment=config["augmentation"]),
        validation_data=_batches(X_val, y_val, config["batch_size"], seed, shuffle=False),
        epochs=config["max_epochs"], shuffle=False, callbacks=callbacks,
        class_weight=class_weight, verbose=verbose,
    )
    history = {key: [float(value) for value in values] for key, values in result.history.items()}
    assert all(np.isfinite(values).all() for values in history.values())
    best_index = int(np.argmin(history["val_loss"]))
    reloaded = evaluate(model, X_val, y_val)  # Poids restaurés de la meilleure epoch.
    for key in ("loss", "accuracy"):
        np.testing.assert_allclose(reloaded[key], history[f"val_{key}"][best_index], rtol=1e-4, atol=1e-5)
    report = class_report(y_val, reloaded["y_pred"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_dir / f"{run_id}_s{seed}.keras")
    record = {
        "run_id": run_id, "label": label, "seed": seed,
        "config": {**config, "filters": list(config["filters"])},
        "params": model.count_params(), "epochs_ran": len(history["val_loss"]), "best_epoch": best_index + 1,
        "best_metrics": {key: values[best_index] for key, values in history.items()},
        "reloaded_val": {"loss": reloaded["loss"], "accuracy": reloaded["accuracy"]},
        "val_macro_f1": float(report.loc["macro avg", "f1-score"]),
        "val_f1": {name: float(report.loc[name, "f1-score"]) for name in report.index[:K]},
        "history": history, "duration_s": round(time.perf_counter() - start, 1),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "gpu": [device.name for device in tf.config.list_physical_devices("GPU")],
                     "tensorflow": tf.__version__},
        "train_sha256": _validation_digest(X_train, y_train), "val_sha256": _validation_digest(X_val, y_val),
        "class_weight": class_weight, "test_used": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[{run_id} seed {seed}] val_loss={record['best_metrics']['val_loss']:.4f} "
          f"val_acc={record['best_metrics']['val_accuracy']:.4f} epoch {best_index + 1}/{record['epochs_ran']} "
          f"({record['duration_s'] / 60:.1f} min)", flush=True)
    return json.loads(record_path.read_text(encoding="utf-8"))  # Même forme qu'à la relecture.


def summarize(out_dir: str | Path = RESEARCH_DIR) -> pd.DataFrame:
    """Moyenne et écart-type par configuration, triés selon le critère de sélection."""
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(out_dir).glob("*_s*.json"))]
    if not records:
        return pd.DataFrame()
    rows = pd.DataFrame([{
        "id": r["run_id"], "modification": r["label"], "seed": r["seed"], "params": r["params"],
        "val_loss": r["best_metrics"]["val_loss"], "val_acc": r["best_metrics"]["val_accuracy"],
        "macro_f1": r["val_macro_f1"], "best_epoch": r["best_epoch"], "minutes": r["duration_s"] / 60,
    } for r in records])
    summary = rows.groupby(["id", "modification", "params"]).agg(
        seeds=("seed", "count"), val_loss_mean=("val_loss", "mean"), val_loss_std=("val_loss", "std"),
        val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"),
        macro_f1_mean=("macro_f1", "mean"), best_epoch_mean=("best_epoch", "mean"), minutes=("minutes", "sum"),
    ).reset_index()
    return summary.sort_values(["val_loss_mean", "val_acc_mean"], ascending=[True, False]).reset_index(drop=True)


def _mean_val_loss(records: list[dict]) -> float:
    return float(np.mean([r["best_metrics"]["val_loss"] for r in records]))


def run_ladder(data: tuple, *, out_dir: str | Path = RESEARCH_DIR,
               checkpoint_dir: str | Path = RESEARCH_CHECKPOINT_DIR, seeds=SEEDS, verbose: int = 2) -> dict:
    """Ajoute un changement à la fois et le garde seulement si la val_loss moyenne baisse."""
    current, best_loss, decisions = dict(BASE_CONFIG), None, []
    for run_id, label, change in LADDER:
        config = {**current, **change}
        records = [run_research(run_id, label, config, seed, data, out_dir=out_dir,
                                checkpoint_dir=checkpoint_dir, verbose=verbose) for seed in seeds]
        mean_loss = _mean_val_loss(records)
        adopted = best_loss is None or mean_loss < best_loss
        if adopted:
            current, best_loss = config, mean_loss
        decisions.append({"id": run_id, "label": label, "change": change,
                          "val_loss_mean": mean_loss, "adopted": adopted})
        ladder = {"decisions": decisions, "best_config": {**current, "filters": list(current["filters"])},
                  "best_val_loss_mean": best_loss, "selection": SELECTION}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "ladder.json").write_text(json.dumps(ladder, indent=1) + "\n", encoding="utf-8")
    return ladder


def run_search(data: tuple, count: int = 6, *, out_dir: str | Path = RESEARCH_DIR,
               checkpoint_dir: str | Path = RESEARCH_CHECKPOINT_DIR, seeds=SEEDS, verbose: int = 2) -> list:
    """Tire des réglages autour du meilleur palier (seed 42), puis confirme les deux meilleurs sur 3 seeds."""
    base = json.loads((Path(out_dir) / "ladder.json").read_text(encoding="utf-8"))["best_config"]
    rng = np.random.default_rng(SEARCH_SEED)
    candidates = []
    for index in range(1, count + 1):
        config = {**base, **{key: values[rng.integers(len(values))] for key, values in SEARCH_SPACE.items()},
                  "learning_rate": float(10 ** rng.uniform(np.log10(2e-4), np.log10(2e-3)))}
        config = {key: value.item() if hasattr(value, "item") else value for key, value in config.items()}
        label = (f"Recherche : lr {config['learning_rate']:.1e}, batch {config['batch_size']}, "
                 f"dropout {config['conv_dropout']}/{config['dense_dropout']}, dense {config['dense_units']}")
        record = run_research(f"S{index}", label, config, seeds[0], data, out_dir=out_dir,
                              checkpoint_dir=checkpoint_dir, verbose=verbose)
        candidates.append((record["best_metrics"]["val_loss"], f"S{index}", label, config))
    for _, run_id, label, config in sorted(candidates)[:2]:
        for seed in seeds[1:]:
            run_research(run_id, label, config, seed, data, out_dir=out_dir,
                         checkpoint_dir=checkpoint_dir, verbose=verbose)
    return candidates


def main() -> None:
    """Lance l'échelle, la recherche aléatoire ou un smoke test temporaire."""
    parser = argparse.ArgumentParser(description="Approfondissement CNN (validation uniquement).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--search", type=int, default=0, help="Nombre de configurations aléatoires.")
    parser.add_argument("--smoke", action="store_true", help="R0 sur 256/64 images, 1 epoch, dossier temporaire.")
    args = parser.parse_args()
    X_train, y_train, X_val, y_val, _, _ = load_dataset(args.data_dir)  # Le test n'est jamais transmis.
    reference = json.loads((LOG_DIR / "C0_history.json").read_text(encoding="utf-8"))
    assert _validation_digest(X_val, y_val) == reference["config"]["validation_sha256"], "Split différent de C0."
    if args.smoke:
        with TemporaryDirectory(prefix="fer2013-research-smoke-") as directory:
            record = run_research("R0", "smoke", {**BASE_CONFIG, "max_epochs": 1},
                                  42, (X_train[:256], y_train[:256], X_val[:64], y_val[:64]),
                                  out_dir=directory, checkpoint_dir=directory)
            print("Smoke research conforme :", record["params"], "paramètres")
        return
    data = (X_train, y_train, X_val, y_val)
    if args.ladder:
        run_ladder(data)
    if args.search:
        run_search(data, args.search)
    print(summarize().to_string())
    summarize().to_csv(RESEARCH_DIR / "summary.csv", index=False)


if __name__ == "__main__":
    main()
