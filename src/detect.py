"""Détection YuNet et classification par le modèle final sur des images BGR, sans entraînement."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
import tensorflow as tf

from src.data import CHANNELS, CLASS_NAMES, IMAGE_SIZE, K
from src.evaluate import predict_faces


DETECTOR_PATH = Path("training/checkpoints/face_detection_yunet_2023mar.onnx")
CLASSIFIER_PATH = Path("training/checkpoints/S5.keras")
DETECTOR_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECTOR_LICENSE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE"
)
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
MAX_SIDE = 1280
LINE_WIDTH = 2
FONT_SCALE = 0.6
TEXT_LINE_HEIGHT = 24
DEMO_DIR = Path("data/demo")
# Photos NASA (domaine public aux États-Unis), sans lien avec FER2013.
DEMO_IMAGES = {
    "apollo11.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/3d/Apollo_11_Crew.jpg",
    "apollo12.jpg": (
        "https://assets.science.nasa.gov/dynamicimage/assets/science/psd/"
        "solar/2023/09/a/Apollo_12_crew.jpg"
    ),
    "apollo13.jpg": "https://upload.wikimedia.org/wikipedia/commons/1/17/Apollo_13_Prime_Crew.jpg",
}


def _download(url: str, path: Path) -> None:
    """Télécharge un fichier absent sans écraser les fichiers présents."""
    if path.exists():
        return
    request = Request(url, headers={"User-Agent": "FER2013-educational-demo/1.0"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def download_demo_assets(
    detector_path: str | Path = DETECTOR_PATH, demo_dir: str | Path = DEMO_DIR,
) -> None:
    """Télécharge YuNet, sa licence et trois photos NASA dans des dossiers ignorés."""
    detector_path, demo_dir = Path(detector_path), Path(demo_dir)
    _download(DETECTOR_URL, detector_path)
    assert hashlib.sha256(detector_path.read_bytes()).hexdigest() == DETECTOR_SHA256
    _download(DETECTOR_LICENSE_URL, detector_path.with_suffix(".LICENSE"))
    for name, url in DEMO_IMAGES.items():
        _download(url, demo_dir / name)


def load_models(
    detector_path: str | Path = DETECTOR_PATH, classifier_path: str | Path = CLASSIFIER_PATH,
    *, score_threshold: float = SCORE_THRESHOLD, nms_threshold: float = NMS_THRESHOLD,
    top_k: int = TOP_K,
):
    """Charge une fois YuNet et le classifieur final puis vérifie le contrat du classifieur."""
    for path in (detector_path, classifier_path):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Poids absents : {path}. Télécharger YuNet avec --download-demo "
                "et transférer le checkpoint du modèle final depuis les artefacts sauvegardés."
            )
    assert hashlib.sha256(Path(detector_path).read_bytes()).hexdigest() == DETECTOR_SHA256
    if not (0 <= score_threshold <= 1 and 0 <= nms_threshold <= 1 and top_k > 0):
        raise ValueError("Seuils dans [0, 1] et top_k strictement positif requis.")
    # Backend OpenCV explicite : le modèle 2023 a des dimensions ONNX fixes.
    detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (320, 320), score_threshold, nms_threshold, top_k,
        cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    classifier = tf.keras.models.load_model(classifier_path, compile=False)
    assert classifier.input_shape == (None, *IMAGE_SIZE, CHANNELS)
    assert K == len(CLASS_NAMES) == classifier.layers[-1].units == classifier.output_shape[-1]
    assert classifier.layers[-1].activation.__name__ == "softmax"
    return detector, classifier


def _check_frame(frame_bgr: np.ndarray) -> None:
    """Vérifie une image OpenCV BGR uint8 non vide."""
    assert isinstance(frame_bgr, np.ndarray) and frame_bgr.dtype == np.uint8
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3
    assert frame_bgr.shape[0] > 0 and frame_bgr.shape[1] > 0


def detect_expressions(
    frame_bgr: np.ndarray, detector, classifier, *, max_side: int = MAX_SIDE,
) -> list[dict]:
    """Renvoie boîtes xyxy, score détecteur, classe et softmax pour une frame BGR."""
    _check_frame(frame_bgr)
    if max_side <= 0:
        raise ValueError("max_side doit être strictement positif.")
    height, width = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    resized = cv2.resize(
        frame_bgr, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    ) if scale < 1 else frame_bgr
    detector.setInputSize((resized.shape[1], resized.shape[0]))
    _, faces = detector.detect(resized)  # x, y, w, h, 5 points faciaux, score.
    if faces is None or len(faces) == 0:
        return []
    assert faces.ndim == 2 and faces.shape[1] == 15
    # Facteurs exacts après arrondi ; les boîtes restent dans l'image originale.
    scale_x, scale_y = width / resized.shape[1], height / resized.shape[0]
    results, crops_rgb = [], []
    for face in faces:
        x, y, box_width, box_height = face[:4]
        detector_score = float(face[14])
        if not np.isfinite(face).all() or box_width <= 0 or box_height <= 0:
            continue
        if not 0 <= detector_score <= 1:
            continue
        x1 = int(np.clip(np.floor(x * scale_x), 0, width))
        y1 = int(np.clip(np.floor(y * scale_y), 0, height))
        x2 = int(np.clip(np.ceil((x + box_width) * scale_x), 0, width))
        y2 = int(np.clip(np.ceil((y + box_height) * scale_y), 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        # Pillow/preprocess_face attend RGB ; YuNet et les frames OpenCV sont BGR.
        crops_rgb.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        results.append({"box_xyxy": [x1, y1, x2, y2], "detector_score": detector_score})
    if not results:
        return []
    probabilities = predict_faces(classifier, crops_rgb)  # Un batch pour tous les visages.
    assert probabilities.shape == (len(results), K)
    for result, scores in zip(results, probabilities):
        class_id = int(scores.argmax())
        result.update(
            class_id=class_id, expression=CLASS_NAMES[class_id],
            expression_probability=float(scores[class_id]),
        )
    return results


def annotate_faces(frame_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
    """Dessine les résultats sur une copie BGR sans modifier l'image source."""
    _check_frame(frame_bgr)
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    display_scale = max(1.0, max(height, width) / MAX_SIDE)
    font_scale = FONT_SCALE * display_scale
    line_height = round(TEXT_LINE_HEIGHT * display_scale)
    line_width = round(LINE_WIDTH * display_scale)
    for index, result in enumerate(results, start=1):
        x1, y1, x2, y2 = result["box_xyxy"]  # x2/y2 exclusifs pour le crop.
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), line_width)
        labels = (
            f"#{index} {result['expression']} p={result['expression_probability']:.2f}",
            f"face score={result['detector_score']:.2f}",
        )
        text_x = min(x1, max(0, width - max(
            cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width)[0][0]
            for label in labels
        )))
        text_y = max(line_height, min(y1 - line_height, height - line_height))
        for line, label in enumerate(labels):
            position = (text_x, text_y + line * line_height)
            cv2.putText(annotated, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), line_width + 2, cv2.LINE_AA)
            cv2.putText(annotated, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 255, 0), line_width, cv2.LINE_AA)
    return annotated


def main() -> None:
    """Exécute une démo image et sauvegarde annotations PNG et résultats JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--download-demo", action="store_true")
    parser.add_argument("--detector", type=Path, default=DETECTOR_PATH)
    parser.add_argument("--classifier", type=Path, default=CLASSIFIER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIR / "annotated")
    args = parser.parse_args()
    if args.download_demo:
        download_demo_assets(args.detector)
    detector, classifier = load_models(args.detector, args.classifier)
    paths = args.images or [DEMO_DIR / name for name in DEMO_IMAGES]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Image absente ou illisible : {path}. Utiliser --download-demo.")
        _check_frame(frame)
        results = detect_expressions(frame, detector, classifier)
        output = args.output_dir / f"{path.stem}_annotated.png"
        assert cv2.imwrite(str(output), annotate_faces(frame, results))
        payload = {"image": str(path), "results": results}
        output.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(f"{path.name} : {len(results)} visage(s), annotation : {output}")


if __name__ == "__main__":
    main()
