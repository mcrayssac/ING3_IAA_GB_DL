"""Contrats de la détection sans poids téléchargés ni données FER2013."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.data import CLASS_NAMES, K, preprocess_face
from src.detect import _decode_yolo, _yolo_candidates, annotate_faces, detect_expressions
from src.evaluate import predict_faces


class FakeDetector:
    """Simule les boîtes YuNet pour vérifier les cas limites."""

    def __init__(self, rows):
        """Prépare les sorties du détecteur."""
        self.rows = rows

    def setInputSize(self, size):
        """Mémorise les dimensions transmises à YuNet."""
        self.size = size

    def detect(self, frame):
        """Vérifie l'entrée et renvoie les boîtes simulées."""
        assert frame.shape[:2] == self.size[::-1]
        return 1, self.rows


class FakeClassifier:
    """Capture le batch pour contrôler le prétraitement et les probabilités."""

    input_shape = (None, 48, 48, 1)
    output_shape = (None, K)
    layers = [SimpleNamespace(units=K)]

    def __init__(self):
        """Initialise le compteur d'inférences."""
        self.calls = 0

    def predict(self, batch, **kwargs):
        """Renvoie une distribution softmax distincte du score détecteur."""
        self.calls += 1
        self.batch = batch
        scores = np.full((len(batch), K), 0.05, dtype=np.float32)
        scores[:, 3] = 0.7
        return scores


def rows(*boxes):
    """Construit les quinze colonnes de sortie de YuNet."""
    values = np.zeros((len(boxes), 15), dtype=np.float32)
    for row, (x, y, width, height, score) in zip(values, boxes):
        row[:4], row[14] = (x, y, width, height), score
    return values


@pytest.mark.parametrize("detections", [None, rows(), rows((50, 50, 2, 2, 0.9))])
def test_no_valid_face_skips_classifier(detections):
    """Zéro visage ou crop hors image évite une inférence vide."""
    model = FakeClassifier()
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    result = detect_expressions(frame, FakeDetector(detections), model)
    assert result == [] and model.calls == 0
    np.testing.assert_array_equal(annotate_faces(frame, result), frame)
    assert predict_faces(model, []).shape == (0, K)
    assert model.calls == 0


def test_multiple_crops_clipping_rgb_and_batch():
    """Deux crops bornés partagent un batch RGB et conservent leurs scores."""
    frame = np.full((20, 30, 3), (255, 0, 0), dtype=np.uint8)  # Bleu BGR.
    detections = rows(
        (-2, -1, 10, 10, 0.99), (12, 8, 20, 20, 0.95),
        (5, 5, 0, 4, 0.99), (5, 5, -2, 4, 0.99),
        (np.nan, 0, 2, 2, 0.99), (5, 5, 2, 2, 1.1),
    )
    model = FakeClassifier()
    result = detect_expressions(frame, FakeDetector(detections), model)
    assert len(result) == 2 and model.calls == 1
    assert [r["box_xyxy"] for r in result] == [[0, 0, 8, 9], [12, 8, 30, 20]]
    expected_rgb = cv2.cvtColor(frame[:9, :8], cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(model.batch[0], preprocess_face(expected_rgb))
    assert model.batch.shape == (2, 48, 48, 1)
    assert model.batch.dtype == np.float32
    for item in result:
        assert set(item) == {
            "box_xyxy", "detector_score", "class_id", "expression", "expression_probability",
        }
        assert item["expression"] == CLASS_NAMES[item["class_id"]] == "happy"
        assert item["expression_probability"] == pytest.approx(0.7)
        assert item["detector_score"] > item["expression_probability"]
    original = frame.copy()
    annotated = annotate_faces(frame, result)
    assert annotated.shape == frame.shape and annotated.dtype == frame.dtype
    assert not np.array_equal(annotated, frame)
    np.testing.assert_array_equal(frame, original)


def test_resized_boxes_return_original_coordinates():
    """Les dimensions arrondies redonnent des coordonnées dans la frame source."""
    detector = FakeDetector(rows((1, 1, 3, 2, 0.99)))
    result = detect_expressions(
        np.zeros((11, 21, 3), dtype=np.uint8), detector, FakeClassifier(), max_side=10,
    )
    assert detector.size == (10, 5)
    assert result[0]["box_xyxy"] == [2, 2, 9, 7]


@pytest.mark.parametrize("invalid", ["classes", "shape", "nan", "normalization"])
def test_classifier_contract_rejects_invalid_output(invalid):
    """Un mauvais nombre de classes, shape ou softmax est refusé."""
    model = FakeClassifier()
    if invalid == "classes":
        model.layers = [SimpleNamespace(units=K - 1)]
    else:
        outputs = {
            "shape": np.zeros((1, K - 1)),
            "nan": np.full((1, K), np.nan),
            "normalization": np.full((1, K), 0.5),
        }
        model.predict = lambda *args, **kwargs: outputs[invalid]
    with pytest.raises(AssertionError):
        predict_faces(model, [np.zeros((10, 10, 3), dtype=np.uint8)])


def test_yolo_decoding_letterbox_and_nms() -> None:
    """La sortie YOLO devient des lignes YuNet dans l'image d'origine, après seuil et NMS."""
    # Image 1280 x 960 réduite de moitié (640 x 480) puis centrée : 80 px de bande en haut.
    output = np.array([[
        [100, 102, 400, 500],  # cx
        [180, 182, 300, 400],  # cy
        [40, 40, 50, 20],      # w
        [60, 60, 50, 20],      # h
        [0.9, 0.8, 0.3, 0.7],  # score : la 2e chevauche la 1re, la 3e est sous le seuil.
    ]], dtype=np.float32)

    faces = _decode_yolo(output, scale=(0.5, 0.5), pad=(0, 80), score_threshold=0.5, nms_threshold=0.45)

    assert faces.shape == (2, 15) and np.all(faces[:, 4:14] == 0)
    np.testing.assert_allclose(faces[0, :4], [160, 140, 80, 120])
    np.testing.assert_allclose(faces[1, :4], [980, 620, 40, 40])
    np.testing.assert_allclose(faces[:, 14], [0.9, 0.7])
    assert _decode_yolo(output, (0.5, 0.5), (0, 80), score_threshold=0.95).shape == (0, 15)
    boxes, scores = _yolo_candidates(output, (0.5, 0.5), (0, 80), score_threshold=0.5)
    assert boxes.shape == (3, 4)  # Avant NMS : la boîte chevauchante est encore là.
    np.testing.assert_allclose(scores, [0.9, 0.8, 0.7])
