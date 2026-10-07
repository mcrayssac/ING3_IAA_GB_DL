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

Dans la phase 2, passer `RUN_B0 = False` à `RUN_B0 = True`, puis exécuter construction, entraînement et courbes dans l'ordre. Un MLP neuf est entraîné pendant 5 epochs, 382 batches par epoch. Avec `RUN_B0 = False`, aucun entraînement B0 ni affichage de ses courbes n'a lieu. Conserver aussi la sortie des 5 epochs avec les résultats.

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
