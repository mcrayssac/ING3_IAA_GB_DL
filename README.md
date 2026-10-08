# Facial expression recognition

We built a seven-class facial expression classifier on FER2013, then connected it to a face detector to process photographs containing several people. This one-week engineering project follows our decisions from a dense baseline to the final YOLO11n-face + S5 system.

**Start with the [project report](docs/project_story.pdf)** for our reasoning, experiments and limits. Browse the [shared Colab notebook](https://colab.research.google.com/drive/17XRdvetj4kuyDCKIXMM8AbEYYKdtxzFP?usp=sharing) to see the saved results and executed demonstration before setting anything up.

## Overview

YOLO11n-face locates faces, our S5 convolutional network assigns an expression to each prepared crop. S5 reached **62.44% test accuracy and 0.562 macro F1**. We selected configurations using validation results and consulted the same test set twice, first for A1, then S5. The later research phase comprised 28 local runs across 12 configurations.

![Two-stage workflow: YOLO detects and filters faces, then grayscale 48×48 crops enter S5 for seven expression scores and PNG/JSON output. The ISS example finds ten faces among eleven people. On 80 NASA photos, YOLO produced 401 boxes, including 42 extras versus YuNet. S5's FER2013 test results are 62.44% accuracy and 0.562 macro F1.](docs/overview.svg)

*The final workflow, illustrated with the recorded ISS example. The score box separates YOLO's counts on 80 NASA photographs from S5's accuracy and macro F1 on the FER2013 test set.*

Ambiguous labels, rotated faces and missed detections remain limitations. Expression scores are uncalibrated and do not establish what someone feels. The NASA comparison has no complete reference boxes, so it does not measure detector precision or recall.

| Find | Resource |
|---|---|
| Notebook walkthrough | [Jupytext source](notebooks/projet.py) |
| Implementation | [Data](src/data.py), [models](src/models.py), [training](src/train.py), [evaluation](src/evaluate.py), [research](src/research.py), [detection](src/detect.py) |
| Saved measurements | [Initial runs and test evaluations](training/logs/), [research summary](training/research/summary.csv) |
| Execution and checks | [Colab guide](docs/COLAB.md), [tests](tests/) |
| Technical notes and history | [Decisions](SPEC.md), [progress](TODO.md), [journal](JOURNAL.md) |

## Setup

From the repository root, use Python 3.12 and an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run the photograph demo

Request `S5.keras` from the collaborators and place it in `training/checkpoints/`. Weights and data are excluded from Git. The standalone demo does not need FER2013.

```bash
python -m src.detect --download-demo
python -m src.detect path/to/photo.jpg
```

The first command downloads detector assets and Apollo photographs, then runs the demo. It does **not** download S5. Both commands write annotated PNGs and JSON results to `data/demo/annotated/`.

### Open the notebook

Prepare FER2013 as described below and request the saved checkpoints needed for the notebook from the collaborators.

```bash
jupytext --to ipynb notebooks/projet.py
jupyter lab notebooks/projet.ipynb
```

Select the environment's kernel. Training flags and `RUN_TEST` default to `False`; running cells still loads data, checks validation predictions and runs available demonstrations. Viewing the shared Colab's saved outputs requires no local setup. To execute it, follow the [archive-upload and runtime instructions](docs/COLAB.md).

## Data sources

- [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013): cropped 48×48 grayscale faces, using the original seven labels. Place image folders under `data/train/` and `data/test/`, each containing `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad` and `surprise`. A CSV alone is unsupported.
- NASA photographs: three Apollo portraits and an 80-photo detector comparison, separate from FER2013. The [photo manifest](docs/nasa_photos.csv) records sources, dates, credits and public-domain status for the comparison set; Apollo credits appear in the report.

## Collaborators

- Maxime CRAYSSAC
- PAUL PITIOT

## References

- [Goodfellow et al. (2013)](https://arxiv.org/abs/1307.0414): FER2013 challenge.
- [Khaireddin and Chen (2021)](https://arxiv.org/abs/2105.03588): published FER2013 comparison.
- [YuNet / OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet): alternative detector.
- [YOLO Face](https://github.com/akanametov/yolo-face) and [ONNX weights](https://huggingface.co/deepghs/yolo-face): pretrained face detection.

The report contains the full bibliography and evidence index.

## License

Project code is licensed under [Apache-2.0](LICENSE). External data, photographs and pretrained weights retain their own terms. See [Kaggle's terms](https://www.kaggle.com/terms), [NASA's image guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/) and the linked model sources before reuse.
