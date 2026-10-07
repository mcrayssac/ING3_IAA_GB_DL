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
src/train.py            entraînement MLP, smoke local et enregistrement B0
src/evaluate.py         évaluation, rapport par classe, exemples et test unique
notebooks/projet.py     source Jupytext du notebook
tests/test_shapes.py    contrats des données, intégrité du split et sortie MLP
training/logs/          historique B0 réel et experiments.csv
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
