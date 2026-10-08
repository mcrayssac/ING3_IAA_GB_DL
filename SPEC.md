# SPEC - Décisions techniques

Référence commune aux phases. Le sujet fait foi ; le suivi des tâches est dans `TODO.md`.

## Données et prétraitement
- FER2013 ([Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)), dossiers `data/train` et `data/test`, non versionnés.
- Classes, dans cet ordre : `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` (`CLASS_NAMES`).
- Entrée : `(48, 48, 1)`, `float32`, pixels dans `[0, 1]`. `preprocess_face()` (`src/data.py`) est l'unique prétraitement, y compris pour l'évaluation, la démo et les visages détectés.
- Labels entiers, loss `sparse_categorical_crossentropy`, sortie `Dense(7, softmax)`.

## Split et test
- Validation : 15 % du train officiel, stratifiée, seed 42 (24 402 train / 4 307 validation).
- Test officiel (7 178 images) : jamais utilisé pour l'entraînement ni les réglages. Une seule évaluation, sur le modèle final.

## Expériences
| Id | Rôle |
|---|---|
| B0 | Baseline MLP (fait) |
| C0 | CNN de départ |
| E1 | Architecture |
| E2 | Régularisation (Dropout ou L2, sans augmentation) |
| E3 | Learning rate / optimiseur / batch |
| A1 | Data augmentation (enrichissement, phase 7) |

Chaque expérience change un seul axe par rapport à sa référence. Le modèle final est choisi sur la validation.
Phase 6 : référence commune C0 ; E1 Dense 128 → 64, E2 Dropout 0,3 après Dense (sans augmentation), E3 Adam learning rate 0,001 → 0,0005. Les autres réglages restent ceux de C0 : filtres 32/64/128, kernel 3, batch 64, seed 42, 30 epochs maximum, patience 5 sur val_loss, meilleurs poids restaurés. Critère fixé avant les runs : val_loss minimale du checkpoint, puis val_accuracy maximale en cas d'égalité exacte, puis id pour un ordre stable. Le candidat de phase 6 reste provisoire jusqu'à A1.

## Artefacts
- Historique et configuration : `training/logs/<id>_history.json`.
- Résumé : `training/logs/experiments.csv`, schéma `id,modification,val_acc,val_loss,params,observation`, une ligne par id.
- Poids : `training/checkpoints/<id>.keras`, non versionnés. Les métriques du CSV sont celles de l'epoch des poids sauvegardés.
- Test officiel : `training/logs/test_evaluation.json`, écrit une seule fois par `evaluate_test_once` (création exclusive) et versionné ; le notebook le relit ensuite.
- Seconde évaluation déclarée du test (modèle final S5, approfondissement) : `training/logs/test_evaluation_deepdive.json`, même garde. Le checkpoint final est copié en `training/checkpoints/S5.keras`.
- Approfondissement CNN : `src/research.py` ; un JSON par run et par seed, `ladder.json` (décisions de l'échelle) et `summary.csv` dans `training/research/` (versionnés) ; checkpoints dans `training/checkpoints/research/` (ignorés). Sélection : val_loss moyenne sur les seeds 42, 43 et 44, validation uniquement.
- Démonstration multi-visages : `src/detect.py` (YuNet OpenCV + A1), `tests/test_detection.py` pour les cas limites. Extension nécessaire, sans nouvelle dépendance ni modification du CNN.
- YuNet ONNX et sa licence : `training/checkpoints/` ; photos NASA et annotations de démo : `data/demo/`. Tous ignorés par Git ; provenance, téléchargement et observations dans le notebook.

## Exécution
- Runs complets sur Google Colab avec GPU ; en local, seulement le smoke test (`python -m src.train`) et les tests (`python -m pytest -q`).
- Seed 42 pour Python, NumPy et TensorFlow.
- Notebook : source `notebooks/projet.py` (Jupytext), `.ipynb` généré et ignoré. Flags `RUN_*` à `False` par défaut : le notebook recharge alors les artefacts sauvegardés au lieu de réentraîner.

## Git
Une branche par phase (`phase-N-nom-court`). Aucun commit sans demande explicite. Données, poids, ZIP et `.ipynb` ne sont jamais versionnés.
