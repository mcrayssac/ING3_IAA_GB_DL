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

## Artefacts
- Historique et configuration : `training/logs/<id>_history.json`.
- Résumé : `training/logs/experiments.csv`, schéma `id,modification,val_acc,val_loss,params,observation`, une ligne par id.
- Poids : `training/checkpoints/<id>.keras`, non versionnés. Les métriques du CSV sont celles de l'epoch des poids sauvegardés.

## Exécution
- Runs complets sur Google Colab avec GPU ; en local, seulement le smoke test (`python -m src.train`) et les tests (`python -m pytest -q`).
- Seed 42 pour Python, NumPy et TensorFlow.
- Notebook : source `notebooks/projet.py` (Jupytext), `.ipynb` généré et ignoré. Flags `RUN_*` à `False` par défaut : le notebook recharge alors les artefacts sauvegardés au lieu de réentraîner.

## Git
Une branche par phase (`phase-N-nom-court`). Aucun commit sans demande explicite. Données, poids, ZIP et `.ipynb` ne sont jamais versionnés.
