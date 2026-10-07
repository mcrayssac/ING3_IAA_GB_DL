# Classification d’expressions faciales

Projet ING3 IA : reconnaître sept expressions faciales sur FER2013 et appliquer le modèle à plusieurs visages dans une photographie. Une expression prédite ne décrit pas avec certitude l’émotion ressentie.

## Méthode

Images en niveaux de gris, entrée `(48, 48, 1)`, pixels normalisés par `/255.0`, labels entiers et sortie softmax à sept classes. `CLASS_NAMES` et l’unique `preprocess_face()` sont définis dans `src/data.py`.

Le train officiel est séparé en 24 402 images d’entraînement et 4 307 de validation (15 %, stratification, seed 42). Les 7 178 images du test officiel sont réservées à une évaluation finale unique. La sélection utilise la loss de validation minimale, puis l’accuracy en cas d’égalité.

Le **MLP B0 est le modèle de référence** : Flatten → Dense(128, ReLU) → Dense(7, softmax), 295 943 paramètres. Le CNN C0 apprend des motifs spatiaux avec trois blocs convolutifs (32/64/128 filtres), puis une couche dense. Les expériences comparent taille de Dense (E1), Dropout (E2), learning rate (E3) et augmentation du train (A1). A1 reprend E3 avec une initialisation neuve et ajoute flip, rotation, translation, zoom et contraste, sans augmenter validation/test.

## Résultats

Métriques des poids sauvegardés, même split et seed :

| Modèle | Modification | Accuracy validation | Loss validation | Paramètres |
|---|---|---:|---:|---:|
| B0 | Référence MLP | 0,360808 | 1,660339 | 295 943 |
| C0 | Référence CNN | 0,556536 | 1,232023 | 683 527 |
| E1 | Dense 64 au lieu de 128 | 0,542373 | 1,228255 | 388 103 |
| E2 | Dropout 0,3 après Dense | 0,557232 | 1,195606 | 683 527 |
| E3 | Adam à 0,0005 | 0,559322 | 1,192440 | 683 527 |
| A1 | E3 + augmentation du train | **0,577200** | **1,126252** | 683 527 |

Le CNN améliore nettement l’accuracy par rapport au MLP sur ce protocole. La comparaison ne mesure pas le seul effet de l’architecture : B0 utilise 5 epochs, les CNN au plus 30 avec arrêt anticipé. Une seule seed ne démontre pas une supériorité générale. A1 améliore la loss mais son F1 macro validation (0,502) reste inférieur à C0 (0,512) et E3 (0,507), notamment sur des classes difficiles.

**Modèle final : A1**, epoch 24 sur 29. Test officiel déjà évalué une seule fois : accuracy **0,575508**, loss **1,131685**, F1 macro **0,508**. `training/logs/test_evaluation.json` fait foi ; conserver `FINAL_MODEL_ID="A1"` et `RUN_TEST=False`. Les historiques, configurations et l’analyse par classe sont dans le notebook et `training/logs/`.

## Installation et exécution

Depuis la racine, avec Python 3.11 et un environnement isolé :

```bash
python3.11 -m venv .venv  # Seulement si aucun environnement adapté n’existe.
source .venv/bin/activate
python -m pip install -r requirements.txt
jupytext --to ipynb notebooks/projet.py
jupyter lab notebooks/projet.ipynb
python -m pytest -q
```

Sélectionner le kernel de cet environnement dans JupyterLab. Télécharger [FER2013 sur Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) en dossiers d’images, puis placer `train/` et `test/` directement dans `data/`, chacun contenant les sept dossiers de `CLASS_NAMES`. Un CSV seul ne convient pas au chargeur.

Le notebook source est `notebooks/projet.py` (Jupytext). Le `.ipynb` est généré et ignoré par Git. Par défaut, tous les flags d’entraînement sont `False` : le notebook recharge les artefacts existants. Transférer les checkpoints sauvegardés dans `training/checkpoints/` pour reproduire les prédictions. Les données et poids ne sont pas fournis par Git.

**Même notebook sur Colab.** Depuis la racine, `scripts/colab_bundle.sh` crée `dist/projet.ipynb`, `dist/projet-code.zip` et `dist/fer2013.zip`. Importer `dist/projet.ipynb` dans Colab puis « Exécuter tout » : la première cellule détecte Colab, demande les deux archives et les extrait ; sur Mac, elle utilise le dépôt local. Le dépôt étant privé, aucun `git clone` n’est fait depuis Colab.

Les runs complets utilisent Colab GPU ; le [guide Colab](docs/COLAB.md) décrit le transfert et l’installation, et le [notebook d’exécution](https://colab.research.google.com/drive/1qt1Swy4VEE67PCs_VjhMG7T5WUWxGNfy) conserve les runs historiques. Les configurations B0/C0/E1–E3/A1 et leurs commandes sont dans le notebook source. En local, `python -m src.train --data-dir data` réalise uniquement un smoke test (au plus 256 train / 64 validation, une epoch).

## Phase 8 : pipeline final multi-visages

C’est le système complet du sujet (Figure 4) et la démonstration du modèle final. **YuNet + modèle final (`FINAL_MODEL_ID`, A1)** : détection → boîtes → crops → prétraitement commun → batch CNN → annotation expression et score. YuNet détecte des visages ; un YOLO COCO fournit des boîtes de personnes et ne suffit pas. L’API OpenCV existante et les petits poids ONNX sous licence MIT évitent une dépendance supplémentaire.

Avec `training/checkpoints/A1.keras` déjà présent :

```bash
python -m src.detect --download-demo
# Puis, sans téléchargement :
python -m src.detect data/demo/apollo11.jpg data/demo/apollo12.jpg data/demo/apollo13.jpg
# Autre photographie :
python -m src.detect chemin/photo.jpg --output-dir data/demo/annotated
```

Le téléchargement récupère YuNet, sa licence et trois photos NASA hors FER2013 dans des dossiers ignorés. Les sorties PNG/JSON sont dans `data/demo/annotated/`. La section dédiée du notebook fournit aussi la démo, la provenance des images, les notions de détection et les observations. Elle peut être exécutée sans charger FER2013, après définition de `PROJECT_ROOT`. Aucun entraînement ni évaluation du test n’est nécessaire.

API réutilisable : `load_models()` une fois, puis `detect_expressions(frame_bgr, detector, classifier)` et `annotate_faces(frame_bgr, results)`. L’entrée est BGR uint8 ; les crops sont convertis en RGB avant `predict_faces()`. Les résultats séparent `detector_score` et `expression_probability` (softmax), avec coordonnées `box_xyxy`, classe et expression. L’annotation renvoie une copie BGR. Aucun suivi temporel n’est implémenté.

## Carte du projet

| Partie du sujet | Section du notebook | Code | Résultats | Tests |
|---|---|---|---|---|
| 1 Données | Phase 1 | `src/data.py` | - | chargement, split, prétraitement |
| 2 Référence dense | Phase 2 | `src/models.py` (`build_mlp`), `src/train.py` | `training/logs/B0_history.json` | MLP |
| 3 CNN | Phase 3 | `src/models.py` (`build_cnn`) | - | CNN |
| 4 Entraînement | Phase 4 | `src/train.py` (`train_cnn`, `verify_cnn_run`) | `C0_history.json` | callbacks, epochs |
| 5 Évaluation | Phase 5, test officiel | `src/evaluate.py` | `test_evaluation.json` | évaluation, test unique |
| 6 Expériences | Phase 6 | `src/train.py` (`PHASE6_EXPERIMENTS`) | `E1`-`E3_history.json`, `experiments.csv` | CSV, expériences |
| 7 Enrichissement | Phase 7 (A1) | `src/train.py` (`_augmentation`) | `A1_history.json` | augmentation |
| 8 Multi-visages | Phase 8, pipeline final | `src/detect.py` | `data/demo/annotated/` (ignoré) | `tests/test_detection.py` |
| 9 Vidéo | non faite (API de la phase 8 réutilisable) | - | - | - |

Autres fichiers : `notebooks/projet.py` (source Jupytext du notebook unique), `scripts/colab_bundle.sh` (archives Colab), `docs/COLAB.md` (procédures et preuves des runs Colab), `SPEC.md` (décisions techniques), `TODO.md` (avancement et conformité au sujet), `JOURNAL.md` (historique). Checkpoints dans `training/checkpoints/` et archives d’artefacts dans `training/archives/`, tous deux ignorés par Git.

## Limites

FER2013 contient des images de faible résolution, des annotations ambiguës et des biais de population. `disgust` et `fear` restent difficiles. Les scores softmax ne sont pas calibrés et peuvent être élevés sur des erreurs.

La démo locale a produit trois boîtes sur chacune des trois photos NASA, inspectées visuellement. Ces portraits principalement frontaux et peu diversifiés ne prouvent pas la robustesse aux occlusions, petits visages ou autres populations. Sans annotations de référence, aucune précision, aucun rappel ni mAP du détecteur n’est mesuré. Les crops réels diffèrent de FER2013 ; les cinq points faciaux YuNet ne sont pas utilisés pour l’alignement. La validation de cette extension est locale, sans nouvelle exécution Colab.

## Sources et auteurs

- [FER2013 / Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) et [conditions d’utilisation](https://www.kaggle.com/terms).
- [YuNet / OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [poids ONNX](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx), [licence MIT - Shiqi Yu](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE), [API FaceDetectorYN](https://docs.opencv.org/4.13.0/df/d20/classcv_1_1FaceDetectorYN.html). Wu, Peng et Yu, *YuNet: A Tiny Millisecond-level Face Detector*, 2023.
- [YOLO - Redmon et al.](https://arxiv.org/abs/1506.02640) et [classes COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).
- Photos, crédit **NASA** : [Apollo 11](https://commons.wikimedia.org/wiki/File:Apollo_11_Crew.jpg), [Apollo 12](https://science.nasa.gov/resource/apollo-12-crew/), [Apollo 13](https://commons.wikimedia.org/wiki/File:Apollo_13_Prime_Crew.jpg), [règles d’utilisation NASA](https://www.nasa.gov/nasa-brand-center/images-and-media/). Domaine public aux États-Unis ; usage pédagogique sans soutien implicite de la NASA.

**Paul PITIOT · Maxime CRAYSSAC**

ING3 Option IA Groupe B - Deep Learning - Hanane Zerdoum
