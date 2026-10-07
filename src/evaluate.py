"""Évaluation d'un modèle sauvegardé : métriques, rapport par classe et test unique."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from src.data import CLASS_NAMES, preprocess_face
from src.train import BATCH_SIZE, CHECKPOINT_DIR, LOG_DIR, _check_data, _validation_digest


TEST_RESULT_NAME = "test_evaluation.json"


def evaluate(model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> dict:
    """Renvoie loss, accuracy, classe prédite et probabilité maximale de chaque image."""
    _check_data(X, y)
    metrics = model.evaluate(X, y, batch_size=BATCH_SIZE, verbose=0, return_dict=True)
    probabilities = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    assert probabilities.shape == (len(y), len(CLASS_NAMES)) and np.isfinite(probabilities).all()
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "y_pred": probabilities.argmax(axis=1),
        "confidence": probabilities.max(axis=1),
    }


def class_report(y: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Precision, recall, F1 et effectif par classe, puis moyennes."""
    report = classification_report(
        y, y_pred, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    return pd.DataFrame(report).T


def confident_examples(
    y: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, correct: bool, count: int = 8,
) -> np.ndarray:
    """Indices des prédictions correctes (ou erronées) les plus confiantes."""
    candidates = np.flatnonzero((np.asarray(y_pred) == np.asarray(y)) == correct)
    order = np.argsort(-np.asarray(confidence)[candidates], kind="stable")
    return candidates[order][:count]


def predict_faces(model: tf.keras.Model, images) -> np.ndarray:
    """Applique preprocess_face() à des images brutes puis renvoie les probabilités."""
    X = np.stack([preprocess_face(image) for image in images])
    return model.predict(X, batch_size=BATCH_SIZE, verbose=0)


def evaluate_test_once(
    run_id: str, X_test: np.ndarray, y_test: np.ndarray, *,
    log_dir: str | Path = LOG_DIR, checkpoint_dir: str | Path = CHECKPOINT_DIR,
) -> dict:
    """Évalue le test officiel une seule fois ; refuse si un résultat existe déjà."""
    result_path = Path(log_dir) / TEST_RESULT_NAME
    if result_path.exists():
        raise FileExistsError(f"Test officiel déjà évalué : {result_path}")
    checkpoint = Path(checkpoint_dir) / f"{run_id}.keras"
    result = evaluate(tf.keras.models.load_model(checkpoint), X_test, y_test)
    payload = {
        "run_id": run_id,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "test_sha256": _validation_digest(X_test, y_test),
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "y_pred": result["y_pred"].tolist(),
        "confidence": np.round(result["confidence"], 6).tolist(),
    }
    text = json.dumps(payload, indent=1, allow_nan=False) + "\n"  # Avant création : pas de fichier vide.
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as stream:  # Création exclusive : jamais d'écrasement.
        stream.write(text)
    return payload


def load_test_evaluation(log_dir: str | Path = LOG_DIR) -> dict | None:
    """Relit l'unique évaluation du test, ou None si elle n'a pas encore eu lieu."""
    path = Path(log_dir) / TEST_RESULT_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
