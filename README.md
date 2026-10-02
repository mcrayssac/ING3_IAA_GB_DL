# Classification d'expressions faciales

Projet Deep Learning : classification de sept expressions faciales à partir du dataset FER2013.

## État du projet

La phase 1 données est terminée : chargement, prétraitement commun et séparation entraînement/validation/test sont testés. Les modèles, entraînements et résultats restent à réaliser ; aucune performance n'est encore annoncée.

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

## Protocole fixé

| Élément | Choix |
|---|---|
| Entrée | `(48, 48, 1)`, `float32`, pixels normalisés dans `[0, 1]` |
| Classes | `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` |
| Labels | Entiers, `sparse_categorical_crossentropy` |
| Validation | 15 % du train, split stratifié, seed 42 |
| Test | Dossier officiel, jamais utilisé pour les réglages |

`preprocess_face()` dans `src/data.py` est la fonction de référence, déjà utilisée lors du chargement des images. Les futurs modules d'entraînement, d'évaluation et de démo devront la réutiliser.

## Structure

```text
src/data.py             chargement, split et prétraitement
notebooks/projet.py     source Jupytext du notebook
tests/test_shapes.py    contrats des données et intégrité du split
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