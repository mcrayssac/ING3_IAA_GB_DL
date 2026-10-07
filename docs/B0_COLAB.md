# B0 : reproduction sur Google Colab

B0 est la baseline MLP : `Flatten → Dense(128, relu) → Dense(7, softmax)`, entrée `(48, 48, 1)`, 295943 paramètres. Conserver seed 42 (Python, NumPy, TensorFlow), Adam à `1e-3`, batch 64 et 5 epochs sur le train/validation complet. Aucun callback ni retour aux meilleurs poids : les résultats et le modèle sauvegardés concernent la dernière epoch. Le test officiel ne sert ni à l'entraînement ni aux réglages.

## 1. Préparer le transfert sur le Mac

Depuis la racine du dépôt, avec FER2013 dans `data/train` et `data/test` et l'environnement local existant :

```bash
.venv/bin/jupytext --to ipynb notebooks/projet.py
transfer_dir=$(mktemp -d /tmp/fer2013-phase2.XXXXXX)
zip -q -r "$transfer_dir/projet-code.zip" src notebooks/projet.py requirements.txt -x '*/__pycache__/*'
if [ -f training/logs/experiments.csv ]; then
    zip -q "$transfer_dir/projet-code.zip" training/logs/experiments.csv
fi
zip -q -r "$transfer_dir/fer2013.zip" data/train data/test -x '*.DS_Store'
echo "$transfer_dir"
```

Les ZIP contiennent le code local, même non poussé, et les images. Le CSV existant est inclus pour préserver les autres expériences. Les archives sont placées hors du dépôt ; le `.ipynb` généré reste ignoré et s'édite uniquement via `notebooks/projet.py`.

## 2. Préparer Colab

Importer `notebooks/projet.ipynb` dans [Colab](https://colab.research.google.com/), puis choisir un runtime Python 3 avec GPU. Ajouter ces cellules **avant les cellules du projet** et les exécuter dans l'ordre.

```python
%cd /content
from google.colab import files
from pathlib import Path
from zipfile import ZipFile

uploaded = files.upload()  # Sélectionner projet-code.zip et fer2013.zip.
del uploaded
project = Path("/content/fer2013-project")
project.mkdir(parents=True, exist_ok=True)
for name in ("projet-code.zip", "fer2013.zip"):
    with ZipFile(Path("/content") / name) as archive:
        archive.extractall(project)
```

**Ne pas installer `requirements.txt` dans Colab** : il fixe les versions locales. Utiliser les bibliothèques natives de Colab, vérifier les imports et conserver leurs versions. Si une installation précédente a créé des conflits, récupérer d'abord les résultats utiles, puis choisir **Exécution > Déconnecter et supprimer l'environnement d'exécution**, reconnecter avec GPU et renvoyer les ZIP. Un simple redémarrage Python ne restaure pas les paquets d'origine ; l'extraction seule ne nécessite aucun redémarrage.

```python
%cd /content/fer2013-project
import sys
from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

lines = ["Python " + sys.version.split()[0]]
for package in ("tensorflow", "keras", "numpy", "matplotlib", "pillow", "scikit-learn"):
    lines.append(package + " " + version(package))
gpus = tf.config.list_physical_devices("GPU")
lines.append("GPU TensorFlow : " + repr(gpus))
print("\n".join(lines))
assert gpus, "GPU TensorFlow indisponible : vérifier le runtime et l'installation."
trace = Path("training/logs/B0_colab_environment.txt")
trace.parent.mkdir(parents=True, exist_ok=True)
trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

Ne pas entraîner si un import ou l'assertion GPU échoue : conserver le message exact avant de modifier l'installation. La trace indique les versions et le GPU détecté par TensorFlow ; elle ne mesure pas l'utilisation du GPU pendant chaque opération.

## 3. Exécuter B0

Exécuter les cellules de phase 1 : le chargeur produit 24 402 images train, 4 307 validation et 7 178 test. Seul le train officiel de 28 709 images est partagé, de façon stratifiée à 15 % avec seed 42. Les pixels sont déjà normalisés ; ne pas les diviser à nouveau.

Dans la phase 2, passer `RUN_B0 = False` à `RUN_B0 = True`, puis exécuter construction, entraînement et courbes dans l'ordre. Un MLP neuf est entraîné pendant 5 epochs, 382 batches par epoch. Avec `RUN_B0 = False`, aucun entraînement B0 n'a lieu ; depuis la phase 4, ses courbes sont rechargées depuis le JSON s'il existe. Conserver aussi la sortie des 5 epochs avec les résultats.

Une réexécution remplace l'historique, le modèle et la seule ligne B0 du CSV, en conservant les autres expériences. Un CSV au schéma incorrect, incomplet, aux métriques invalides ou aux identifiants dupliqués est refusé avant entraînement et sauvegarde. Garder une copie des anciens artefacts avant toute réexécution volontaire.

## 4. Télécharger les résultats

Avant de fermer le runtime temporaire, exécuter :

```python
from google.colab import files
from pathlib import Path
from zipfile import ZipFile

with ZipFile("/content/B0-artifacts.zip", "w") as archive:
    for relative in ("training/logs/B0_history.json", "training/logs/experiments.csv", "training/checkpoints/B0.keras"):
        archive.write(relative, arcname=relative)
    trace = Path("training/logs/B0_colab_environment.txt")
    if trace.is_file():
        archive.write(trace, arcname=str(trace))
files.download("/content/B0-artifacts.zip")
```

Extraire dans un dossier séparé, vérifier, puis recopier aux mêmes chemins locaux. Sauvegarder les fichiers homonymes avant remplacement. Si d'autres expériences ont été ajoutées localement depuis l'upload, conserver leurs lignes et remplacer uniquement B0. JSON/CSV sont destinés au dépôt ; images, `.keras`, ZIP et `.ipynb` générés sont ignorés.

## 5. Vérifier avant clôture

- JSON : id B0, configuration ci-dessus, tailles complètes, quatre séries `loss`, `accuracy`, `val_loss`, `val_accuracy` de 5 nombres finis ; accuracies dans `[0, 1]`, losses non négatives.
- CSV : schéma `id,modification,val_acc,val_loss,params,observation`, une seule ligne B0, 295943 paramètres, métriques de l'epoch 5 cohérentes avec le JSON (tolérance relative `1e-7`, absolue `1e-8`), observation précisant cette epoch.
- Modèle : chargement par `tf.keras.models.load_model`, architecture attendue, entrée `(None, 48, 48, 1)`, sortie `(None, 7)`, 295943 paramètres ; probabilités synthétiques finies et de somme 1. Le chargement seul ne prouve pas le lien des poids avec l'epoch 5 du JSON.
- Traces Colab : versions, GPU détecté et cinq epochs terminées. Distinguer une sortie conservée d'une simple déclaration d'exécution. Ne jamais évaluer le test officiel pour cette review.

Actualiser README/TODO/JOURNAL selon les preuves, sans cocher la validation globale du notebook Colab des livrables. En l'absence d'une preuve nécessaire, laisser la clôture en attente plutôt que relancer B0 pour combler la documentation.

## Résultats récupérés et review du 2 octobre 2026

JSON/CSV complets et concordants : 24 402 train / 4 307 validation, protocole attendu, une seule ligne B0 et observation « Dernière epoch 5/5 ». Le comptage des fichiers train locaux retrouve 28 709 exemples et le même split.

| Epoch | Accuracy train | Loss train | Accuracy validation | Loss validation |
|---|---|---|---|---|
| 1 | 0.286247 | 1.782291 | 0.309728 | 1.730711 |
| 2 | 0.327268 | 1.703407 | 0.336893 | 1.700858 |
| 3 | 0.346201 | 1.677842 | 0.358254 | 1.679070 |
| 4 | 0.346693 | 1.668129 | 0.354075 | 1.674736 |
| 5 | 0.350135 | 1.655288 | 0.360808 | 1.660339 |

Les pertes diminuent côté train et validation. L'accuracy de validation baisse légèrement à l'epoch 4 puis remonte ; aucun écart croissant n'indique un surapprentissage marqué sur ces cinq epochs. L'accuracy finale reste modeste, sans seuil obligatoire pour cette baseline ni conclusion sur le test officiel.

`B0.keras` se recharge dans l'environnement local existant (TensorFlow 2.21.0 / Keras 3.15.1), avec l'architecture et les probabilités attendues. Adam est à `1e-3` et possède 1910 mises à jour, cohérentes avec `5 × 382`. Les métadonnées du fichier indiquent Keras 3.13.2 et une sauvegarde le `2026-10-02@13:10:57`.

La validation du modèle rechargé, uniquement sur les 4307 exemples issus du train, donne `accuracy = 0.36080798506736755` (identique au JSON) et `loss = 1.6603387594223022` (écart `2.384185791015625e-7`). La tolérance relative `1e-5` / absolue `1e-6` tient compte des calculs entre environnements. Cela étaye la concordance avec les résultats finaux ; les métriques et le compteur Adam ne certifient pas l'identité de chaque poids avec un checkpoint d'origine non conservé séparément.

La trace texte fournie lors de la review rapporte les 5 epochs complètes, 382 batches chacune, les mêmes métriques arrondies et le split 24402/4307/7178 depuis `/content/fer2013-project`. Versions rapportées : Python 3.13.15, TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.1.3, Matplotlib 3.10.0, Pillow 11.3.0, scikit-learn 1.6.1. L'utilisateur confirme que la cellule terminée par `assert tf.config.list_physical_devices("GPU")` s'est exécutée sans erreur. Aucune sortie donnant les périphériques GPU n'a été conservée : ce succès est rapporté par l'utilisateur, sans preuve indépendante du matériel exact ni de son utilisation pendant `fit`. Les durées et les artefacts seuls ne prouvent pas la présence du GPU.

La phase 2 est validée sur ces vérifications et la trace fournie. La validation globale « Exécuter tout » sur runtime Colab neuf reste une tâche des livrables.

Contrôles locaux : 23 tests réussis, dont conservation des autres expériences et refus des CSV invalides avant construction/entraînement/sauvegarde ; source du notebook exécutée avec `RUN_B0=False` depuis racine et `notebooks/`, sans entraînement ni courbes B0 ; `git diff --check` sans erreur. Les artefacts existants sont restés identiques byte pour byte. Aucun nouvel entraînement complet ni évaluation du test officiel, aucune modification de dépendance et aucune action commit/push/merge request.

Références : [entrées/sorties Colab](https://colab.research.google.com/notebooks/io.ipynb), [FAQ et réinitialisation du runtime](https://research.google.com/colaboratory/faq.html).

## C0 — phase 4 sur Colab GPU

C0 utilise directement `build_cnn(filters=(32,64,128), kernel_size=3, dense_units=128, seed=42)` : 683 527 paramètres. Adam `1e-3`, batch 64, maximum 30 epochs, cross-entropie sparse et accuracy. Les deux callbacks surveillent `val_loss`, mode `min` : meilleur checkpoint uniquement, patience 5 et restauration des meilleurs poids. La première epoch minimale fournit **ensemble** val_acc et val_loss ; B0 conserve sa dernière epoch. Le run réel a été vérifié le 7 octobre 2026 (résultats ci-dessous).

### Transférer le code courant et les artefacts

Depuis la racine du dépôt sur le Mac, ces commandes incluent les modifications non commitées, sans push, et ne changent aucun artefact B0 :

```bash
.venv/bin/jupytext --to ipynb notebooks/projet.py
c0_transfer_dir=$(mktemp -d /tmp/fer2013-c0-transfer.XXXXXX)
zip -q -r "$c0_transfer_dir/projet-code.zip" src notebooks/projet.py SPEC.md training/logs training/checkpoints/B0.keras -x '*/__pycache__/*' '*smoke*' '*C0*' '*.DS_Store'
zip -q -r "$c0_transfer_dir/fer2013.zip" data/train data/test -x '*.DS_Store'
printf '%s\n' "$c0_transfer_dir"
```

Importer le `.ipynb` généré dans Colab. Utiliser un runtime neuf **Python 3 avec GPU** et les bibliothèques natives ; aucune installation de `requirements.txt`. Les données test sont présentes pour les cellules de présentation existantes ; elles ne sont jamais transmises à `fit` ou `evaluate`. B0_history.json, le CSV et B0.keras sont transférés pour la comparaison et leur préservation. Si FER2013 est déjà extrait dans ce runtime au bon chemin, seule l'archive code est nécessaire.

Ajouter et exécuter ces deux cellules avant les cellules du notebook :

```python
%cd /content
from google.colab import files
from pathlib import Path
from zipfile import ZipFile

uploaded = files.upload()  # Sélectionner projet-code.zip et fer2013.zip.
del uploaded
project = Path('/content/fer2013-project')
project.mkdir(parents=True, exist_ok=True)
for name in ('projet-code.zip', 'fer2013.zip'):
    with ZipFile(Path('/content') / name) as archive:
        archive.extractall(project)
```

```python
%cd /content/fer2013-project
import tensorflow as tf
from importlib.metadata import version

print({package: version(package) for package in ('tensorflow', 'keras', 'numpy', 'matplotlib', 'pandas', 'pillow', 'scikit-learn')})
print('GPU TensorFlow :', tf.config.list_physical_devices('GPU'))
assert tf.config.list_physical_devices('GPU'), 'Arrêter : GPU indisponible.'
!nvidia-smi
```

### Entraîner puis contrôler dans le même runtime

Garder `RUN_B0=False`. Passer **uniquement** `RUN_C0=True` dans la cellule phase 4 du notebook importé, puis exécuter toutes les cellules du projet dans l'ordre. Le split affiché doit être **24 402 train / 4 307 validation**, seed 42, sept classes. L'architecture et le summary de phase 3 restent inchangés. Ne pas relancer un C0 terminé : il remplacerait sa seule ligne CSV et ses artefacts.

La cellule C0 sauvegarde une trace `training/logs/C0_colab_environment.txt` (JSON texte) avec GPU, versions, split et empreintes B0 avant entraînement. Après le run, elle compare les poids retournés au checkpoint, les artefacts B0 et les autres lignes CSV, puis réévalue **uniquement la validation complète du run**. Son empreinte SHA-256 dans l'historique impose les mêmes images, labels et ordre. La trace terminée ajoute les résultats rechargés et les assertions de préservation. L'historique note aussi le périphérique réel du premier calcul de prédiction ; un runtime GPU seul ne prouve pas son utilisation. Conserver la sortie des epochs et celle de `nvidia-smi`.

Contrôle réexécutable sans entraînement, dans une nouvelle cellule du même runtime :

```python
from src.train import verify_cnn_run
report = verify_cnn_run(X_val, y_val)
```

Ce contrôle exige un C0 complet : séries finies de même longueur, nombre d'epochs exécutées, première val_loss minimale, best_metrics et CSV de cette epoch, protocole et split complets, GPU, architecture du checkpoint et nombre de mises à jour Adam. Il confronte aussi la trace aux versions du run, aux empreintes B0 locales et aux lignes CSV préservées. Les métriques de validation rechargées doivent concorder avec l'historique (`rtol=1e-5`, `atol=1e-6`, tolérances numériques entre environnements, aucun seuil de performance).

Puis remettre `RUN_C0=False` dans le notebook et exécuter les cellules du projet, ou utiliser cette cellule exacte qui relit la source transférée avec les deux flags déjà à `False` :

```python
%run /content/fer2013-project/notebooks/projet.py
assert RUN_B0 is False and RUN_C0 is False
assert b0_payload is not None and c0_payload is not None
assert comparison['id'].tolist() == ['B0', 'C0']
```

Vérifier visuellement les deux paires de courbes et le tableau B0/C0, avec une seule sortie de summary CNN par exécution. Aucun entraînement n'est déclenché. S'il manque un historique, le notebook le signale sans inventer de valeur. La présentation numérique C0 provient du JSON ; l'analyse écrite du notebook décrit le run vérifié ci-dessous.

### Récupérer les résultats et les preuves

Avant de fermer le runtime :

```python
from google.colab import files
from zipfile import ZipFile

with ZipFile('/content/C0-artifacts.zip', 'w') as archive:
    for relative in (
        'training/logs/C0_history.json',
        'training/logs/experiments.csv',
        'training/checkpoints/C0.keras',
        'training/logs/C0_colab_environment.txt',
    ):
        archive.write(relative, arcname=relative)
files.download('/content/C0-artifacts.zip')
```

Sur le Mac, extraire dans un dossier temporaire, sans écraser les résultats locaux avant review :

```bash
c0_review_dir=$(mktemp -d /tmp/fer2013-c0-review.XXXXXX)
unzip -q "$HOME/Downloads/C0-artifacts.zip" -d "$c0_review_dir"
.venv/bin/python -m src.train --check-c0 --artifact-dir "$c0_review_dir/training"
printf '%s\n' "$c0_review_dir"
```

Fournir ce dossier/ZIP ainsi que la sortie des epochs, la preuve GPU et l'affichage des courbes/tableau avec les deux flags False. La review compare également les empreintes B0 et les lignes non C0 du CSV à leurs originaux locaux. Si d'autres expériences ont été ajoutées depuis le transfert, fusionner uniquement la ligne C0 après contrôle. Importer ensuite JSON/CSV/trace aux chemins prévus et C0.keras dans `training/checkpoints/` (toujours ignoré). Ne pas cocher la phase 4 avant cette review et l'analyse réelle. Aucun test officiel n'est évalué.

Reproduire le smoke local, toujours limité à 256/64/1 et dans un dossier temporaire :

```bash
c0_smoke_dir=$(mktemp -d /tmp/fer2013-cnn-smoke.XXXXXX)
.venv/bin/python -m src.train --model cnn --output-dir "$c0_smoke_dir"
.venv/bin/python -m pytest -q
git diff --check
```

Références du protocole : [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint), [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping), [runtime GPU et limites Colab](https://research.google.com/colaboratory/faq.html).

### Résultats C0 et review du 7 octobre 2026

Le [notebook d'exécution Colab](https://colab.research.google.com/drive/1qt1Swy4VEE67PCs_VjhMG7T5WUWxGNfy) utilise les archives du code local non poussé, avec les paquets natifs. La source `notebooks/projet.py` a été exécutée en activant C0 uniquement en mémoire ; le fichier conserve `RUN_B0=False` et `RUN_C0=False`. Aucune réintégration Git ni modification B0.

GPU détecté : **Tesla T4**, mémoire 15 360 MiB, pilote 580.82.07, CUDA annoncée par `nvidia-smi` 13.0. Versions : Python 3.13.15, TensorFlow 2.21.0, Keras 3.13.2, NumPy 2.1.3, Matplotlib 3.10.0, Pandas 2.2.3, Pillow 11.3.0, scikit-learn 1.6.1. L'historique conserve le premier calcul de prédiction sur `/device:GPU:0`. La sortie `nvidia-smi` est celle d'avant entraînement ; elle ne mesure pas l'utilisation pendant chaque batch.

Split complet : **24 402 train / 4 307 validation**, 382 batches par epoch, seed 42. L'empreinte des images et labels de validation dans leur ordre est `786b2f028d10b41cf6b2558ccc4b4b00edc3f275519853e4849a61e003da70e3`, identique lors du contrôle local. Les douze epochs du log et les quatre séries finies du JSON concordent. La première loss validation minimale est à l'epoch **7**, suivie de cinq epochs sans amélioration : arrêt à l'epoch **12**.

| Modèle | Epoch retenue | Epochs exécutées | val_acc | val_loss | Paramètres |
|---|---:|---:|---:|---:|---:|
| B0 | 5 (dernière) | 5 | 0,360808 | 1,660339 | 295 943 |
| C0 | 7 (meilleure val_loss) | 12 | 0,556536 | 1,232023 | 683 527 |

Le checkpoint C0, le JSON et la ligne CSV rapportent la même epoch. Adam conserve **2 674 mises à jour** (`7 × 382`) dans le checkpoint. Dans Colab, les poids rechargés et les poids retournés après restauration sont identiques tableau par tableau ; la validation rechargée donne exactement `accuracy=0.5565358996391296`, `loss=1.232022762298584`. En local (TensorFlow 2.21.0 / Keras 3.15.1), les mêmes 4 307 images donnent `accuracy=0.5565358996391296`, `loss=1.2320228815078735` : écart de loss `1.1920928955078125e-7`, compatible avec les tolérances numériques prévues.

La loss train descend de 1,679176 à 0,363075. Après l'epoch 7, la loss validation remonte de 1,232023 à 1,898607 alors que l'accuracy validation reste vers 0,55 : le surapprentissage est visible. L'accuracy validation maximale (0,558161, epoch 11) accompagne une loss de 1,643386 et n'est pas sélectionnée. C0 gagne 19,57 points d'accuracy validation par rapport à B0, mais les budgets et callbacks différents limitent l'interprétation causale de ce gain. Aucun résultat du test officiel n'est utilisé.

Les empreintes de `B0_history.json` et `B0.keras` correspondent à leurs originaux locaux ; les lignes non C0 du CSV sont inchangées. À l'import, seul le retour chariot final de la nouvelle ligne C0 a été retiré pour `git diff --check`, sans changer aucune valeur. Les deux flags False ont été exécutés dans Colab avec `Model.fit` remplacé temporairement par une fonction qui échoue : zéro fit, deux paires de courbes et tableau réel affichés, un seul summary CNN par exécution. Les PNG exportés de cette relecture ont été inspectés.

`C0_history.json`, `experiments.csv`, `C0_colab_environment.txt` et le checkpoint ont été récupérés puis contrôlés avant import. Le ZIP téléchargé contient aussi `evidence/` : preuve matérielle, sortie des douze epochs, sortie flags False, tableau CSV et courbes PNG. Ces preuves sont conservées dans le notebook Colab et le ZIP téléchargé ; le checkpoint local demeure ignoré. La phase 4 est clôturée, sans lancer les phases suivantes ni valider les livrables globaux Colab.

Contrôle reproductible depuis la racine, sans entraînement ni évaluation du test :

```bash
.venv/bin/python -m src.train --check-c0
.venv/bin/python -m pytest -q
git diff --check
```
