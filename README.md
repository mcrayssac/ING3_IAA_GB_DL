# Classification d'expressions faciales

Projet Deep Learning : classification de sept expressions faciales à partir du dataset FER2013.


## Installation locale

Utiliser Python 3.11 dans un environnement isolé et lancer les commandes depuis la racine du dépôt. JupyterLab utilise le kernel choisi ; celui enregistré ci-dessous pointe vers le Python de `.venv`.

```bash
# Créer .venv uniquement s'il n'existe pas déjà.
if [ ! -e .venv ]; then python3.11 -m venv .venv; fi
source .venv/bin/activate
python --version  # Doit afficher Python 3.11.x.
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ing3-iaa-gb-dl --display-name "Python (ING3 IAA GB DL)"
jupytext --to ipynb notebooks/projet.py
jupyter lab notebooks/projet.ipynb
```

Dans JupyterLab, choisir le noyau **Python (ING3 IAA GB DL)**. Si un `.venv` existe, le réutiliser seulement s'il est en Python 3.11 ; sinon créer un environnement sous un autre nom et adapter son chemin d'activation.

Avant d'exécuter le notebook, télécharger la version de [FER2013 sur Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) organisée en dossiers par classe, puis extraire l'archive. Le dépôt ne fournit pas ces données : `data/` est volontairement ignoré par Git.

Placer les dossiers extraits `train/` et `test/` directement dans `data/` à la racine du dépôt :

```text
data/
├── train/   # angry, disgust, fear, happy, neutral, sad, surprise
└── test/    # mêmes sept dossiers ; réservé à l'évaluation finale
```

Chaque dossier de classe doit contenir directement ses fichiers image (JPEG/PNG, ou autre format reconnu par Pillow). Le chargeur tente d'ouvrir chaque fichier : ne pas y laisser d'archive, de CSV ou de fichier annexe. Un fichier `fer2013.csv` seul ne convient pas à ce chargeur. Si l'extraction crée un dossier intermédiaire (`archive/` ou `fer2013/`), déplacer son `train/` et son `test/` dans `data/`, sans conserver ce niveau supplémentaire.

Le notebook résout la même racine depuis le dépôt ou depuis `notebooks/`. Une fois les données installées, régénérer manuellement le notebook depuis sa source avec `jupytext --to ipynb notebooks/projet.py` depuis la racine, le rouvrir, sélectionner **Python (ING3 IAA GB DL)**, puis utiliser **Restart Kernel and Run All Cells**. Ne pas exécuter les cellules suivantes si le chargement échoue : `y_train` n'est alors pas défini.

Pour l'entraînement complet, utiliser Google Colab avec GPU. Les données et les poids ne sont jamais versionnés.

## Vérifications

```bash
python -m pytest -q
```

`notebooks/projet.py` est la source du notebook. Le fichier `.ipynb` est généré, ignoré par Git et ne doit pas être modifié à la main.

## Baseline MLP : smoke local et B0 Colab

B0 sert de référence au futur CNN : `Flatten → Dense(128, relu) → Dense(7, softmax)`, **295943 paramètres**. Il utilise le split complet (24 402 train / 4 307 validation), seed 42 dont TensorFlow, Adam à `1e-3`, batch 64 et 5 epochs, sans callbacks ni restauration. Le test officiel est réservé à l'évaluation finale.

Les résultats de l'epoch 5 sont `val_accuracy = 0.360808` et `val_loss = 1.660339`. L'historique/configuration est dans `training/logs/B0_history.json`, le résumé dans `training/logs/experiments.csv` et le modèle final dans `training/checkpoints/B0.keras` (ignoré par Git). Le [guide B0 Colab](docs/B0_COLAB.md) décrit la reproduction, le transfert et les vérifications, ainsi que les preuves disponibles.

**Smoke local uniquement**, depuis la racine avec l'environnement existant :

```bash
.venv/bin/python -m src.train --data-dir data
```

Il utilise au plus 256 train / 64 validation / 1 epoch et écrit `smoke_history.json` / `smoke.keras`, sans ligne B0. Le chargeur prétraite une seule fois les images ; le test officiel, chargé séparément, n'est transmis ni au modèle ni à `fit`. Ces métriques servent au contrôle technique local.

## Phase 6 : expériences et relais à Maxime

`PHASE6_EXPERIMENTS` dans `src/train.py` fixe trois variantes indépendantes de C0 : E1 réduit uniquement Dense de 128 à 64 (388 103 paramètres), E2 ajoute uniquement Dropout 0,3 après Dense (683 527 paramètres, sans augmentation), E3 réduit uniquement le learning rate Adam de 0,001 à 0,0005. Les autres réglages restent ceux de C0 : seed 42, split complet 24 402/4 307, batch 64, maximum 30 epochs, patience 5 sur val_loss et meilleurs poids restaurés.

Le critère fixé avant les runs est la val_loss minimale du checkpoint, puis l'accuracy validation maximale en cas d'égalité exacte. Les cellules de phase 6 de `notebooks/projet.py` gardent `RUN_E1`, `RUN_E2`, `RUN_E3` à False par défaut ; elles relisent les historiques, affichent les courbes/tableau et vérifient les checkpoints disponibles sans réentraîner. Un artefact absent est signalé ; le relais reste en attente tant que tous les runs ne sont pas vérifiés.

Pour les runs complets, générer le notebook avec `.venv/bin/jupytext --to ipynb notebooks/projet.py`, transférer le code local `src/`, `notebooks/projet.py`, les références `training/logs/B0*`, `training/logs/C0*`, `experiments.csv` et les checkpoints B0/C0 dans Colab, ainsi que FER2013 (procédure d'archives : `docs/B0_COLAB.md`). Dans un runtime GPU, exécuter les définitions et le chargement, puis activer seulement les trois flags E1–E3. Les runs ne nécessitent aucune installation supplémentaire dans Colab. Récupérer `training/logs/E*_history.json`, `training/logs/experiments.csv` et `training/checkpoints/E*.keras` ; les checkpoints restent ignorés par Git.

Contrôle d'un run réel, après chargement du split avec `load_dataset` : `verify_cnn_run(X_val, y_val, run_id="E1", X_train=X_train, y_train=y_train)` (idem E2/E3). Pour un smoke temporaire : `train_experiment("E1", X_train, y_train, X_val, y_val, smoke=True, log_dir=<temp>/logs, checkpoint_dir=<temp>/checkpoints)` ; au plus 256/64/1, aucune ligne dans le CSV officiel.

Maxime reproduira la configuration du candidat provisoire avec une initialisation neuve seed 42 et ajoutera seulement l'augmentation pour A1. Le checkpoint sert de référence de comparaison ; poursuivre son entraînement ajouterait un second effet. Le choix définitif attend A1 : `RUN_TEST=False`, `FINAL_MODEL_ID=None`.

Runs réels du 7 octobre 2026 sur Tesla T4 (TensorFlow 2.21.0 / Keras 3.13.2), dans le [notebook d'exécution Colab](https://colab.research.google.com/drive/1qt1Swy4VEE67PCs_VjhMG7T5WUWxGNfy), cellules phase 6 ajoutées à la suite de C0 :

| Id | Epoch retenue / exécutées | val_acc | val_loss | Paramètres |
|---|---:|---:|---:|---:|
| C0 | 7 / 12 | 0,556536 | 1,232023 | 683 527 |
| E1 | 6 / 11 | 0,542373 | 1,228255 | 388 103 |
| E2 | 8 / 13 | 0,557232 | 1,195606 | 683 527 |
| E3 | 7 / 12 | 0,559322 | 1,192440 | 683 527 |

**Relais à Maxime : E3**, selon la val_loss minimale fixée avant les runs. Sa configuration est l'architecture C0 sans Dropout, avec **Adam à 0,0005**, batch 64, seed 42 et les callbacks C0. Référence : `training/checkpoints/E3.keras` (epoch 7, ignoré par Git), configuration complète : `training/logs/E3_history.json`. Reproduire sur Colab GPU avec `train_experiment("E3", X_train, y_train, X_val, y_val)` ou `RUN_E3=True` seul ; utiliser cette configuration avec un modèle neuf pour A1. E2 est proche (écart de loss 0,003166) : une seule seed ne démontre pas une supériorité générale. L'analyse des trois variations et du surapprentissage est dans le notebook.

Les historiques/checkpoints ont été récupérés, leurs empreintes contrôlées et les métriques JSON/CSV confrontées à la validation rechargée en local (écart de loss inférieur à 2e-7). La relecture flags False est vérifiée en local et dans Colab avec `fit` interdit ; les trois courbes ont été inspectées, et le cas sans artefacts ne produit ni métrique ni candidat. Les historiques/checkpoints B0/C0 et leurs lignes CSV sont préservés. Aucun run E1–E3 ni analyse n'est encore nécessaire ; le choix définitif et le test unique attendent A1 en phase 7.

## Phase 7 : A1 sur Colab

A1 reprend E3 avec des poids neufs et ajoute uniquement l'augmentation des batches d'entraînement (`_augmentation()` dans `src/train.py` : flip horizontal, rotation ±18°, translation et zoom ±15 %, contraste ±20 %). La validation et le test ne sont jamais augmentés, et `A1.keras` garde l'architecture d'E3. Même procédure que E1–E3 : transférer aussi `training/logs/E*` et les checkpoints E1–E3, puis activer seulement `RUN_A1` dans la section phase 7 du notebook. Récupérer `training/logs/A1_history.json`, `training/logs/experiments.csv` et `training/checkpoints/A1.keras`. Contrôle : `verify_cnn_run(X_val, y_val, run_id="A1", X_train=X_train, y_train=y_train)`. Smoke temporaire : `train_experiment("A1", ..., smoke=True, log_dir=<temp>/logs, checkpoint_dir=<temp>/checkpoints)`.

**Résultats A1 et modèle final.** A1 retient l'epoch 24 sur 29 : val_loss 1,126252 et val_acc 0,577200, contre 1,192440 et 0,559322 pour E3. Selon le critère fixé avant les runs, le modèle final est **A1** (`training/checkpoints/A1.keras`, ignoré par Git). Le test officiel a été évalué une seule fois : accuracy 0,575508, loss 1,131685, F1 macro 0,508. Le résultat est dans `training/logs/test_evaluation.json` ; `RUN_TEST` reste à False et le notebook relit ce fichier.

## Protocole fixé

| Élément | Choix |
|---|---|
| Entrée | `(48, 48, 1)`, `float32`, pixels normalisés dans `[0, 1]` |
| Classes | `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` |
| Labels | Entiers, `sparse_categorical_crossentropy` |
| Validation | 15 % du train, split stratifié, seed 42 |
| Test | Dossier officiel, jamais utilisé pour les réglages |

`preprocess_face()` dans `src/data.py` est la fonction de référence, déjà utilisée lors du chargement des images. L'entraînement consomme directement ces images prétraitées ; les futurs modules d'évaluation et de démo devront réutiliser cette fonction pour les images brutes.

## Structure

```text
src/data.py             chargement, split et prétraitement
src/models.py           build_mlp() et build_cnn(), entrée image et sortie softmax
src/train.py            entraînement B0/C0/E1–E3/A1, augmentation, smoke et contrôle des checkpoints
src/evaluate.py         évaluation, rapport par classe, exemples et test unique
notebooks/projet.py     source Jupytext du notebook
tests/test_shapes.py    contrats des données, intégrité du split et sortie MLP
training/logs/          historiques/configurations réels et experiments.csv
docs/B0_COLAB.md        reproduction Colab et vérification des résultats
SPEC.md                 décisions techniques
TODO.md                 phases du projet
```

## Limites et sources

Une expression prédite n'est pas une émotion certaine ni un diagnostic. FER2013 est de faible résolution et comporte des ambiguïtés d'annotation, des biais de population et des conditions de prise de vue variées.

- [FER2013 sur Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [Conditions d'utilisation Kaggle](https://www.kaggle.com/terms)

## Crédits
<br>**Paul PITIOT** - **Maxime CRAYSSAC**<br/>
**ING3 Option IA Groupe B - Deep Learning - Hanane Zerdoum**
