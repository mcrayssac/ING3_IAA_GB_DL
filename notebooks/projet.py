# %% [markdown]
# # Classification d'expressions faciales
#
# **Données.** Nous utilisons FER2013, diffusé sur
# [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013), à partir du jeu de
# données présenté par Goodfellow et al. (2013). La fiche Kaggle indique la licence
# « Database: Open Database, Contents: Database Contents » ; son usage respecte aussi
# les [conditions d'utilisation Kaggle](https://www.kaggle.com/terms).
#
# **Sommaire.** Les sections suivent les parties du sujet.
#
# | Section | Partie du sujet | Code utilisé | Flags (False par défaut) |
# |---|---|---|---|
# | Phase 1 - données | 1 | `src/data.py` | - |
# | Phase 2 - baseline MLP | 2 | `src/models.py`, `src/train.py` | `RUN_B0` |
# | Phase 3 - CNN | 3 | `src/models.py` | - |
# | Phase 4 - entraînement C0 | 4 | `src/train.py` | `RUN_C0` |
# | Phase 5 - évaluation et erreurs | 5 | `src/evaluate.py` | - |
# | Phase 6 - expériences E1 à E3 | 6 | `src/train.py` | `RUN_E1`, `RUN_E2`, `RUN_E3` |
# | Phase 7 - augmentation A1 | 7 | `src/train.py` | `RUN_A1` |
# | Approfondissement CNN | 6 | `src/research.py` | `RUN_RESEARCH` |
# | Test officiel | 5 et 6 | `src/evaluate.py` | `RUN_TEST` |
# | Phase 8 - pipeline final multi-visages | 8 et démonstration | `src/detect.py` | `DOWNLOAD_FACE_DEMO`, `RUN_CUSTOM_IMAGE` |
#
# Avec tous les flags à False, le notebook n'entraîne rien : il relit les
# résultats sauvegardés dans `training/logs/` et les checkpoints de
# `training/checkpoints/`. La première cellule détecte seule l'environnement.
# Sur un Mac, elle utilise le dépôt local. Sur Colab, elle demande les deux
# archives produites par `scripts/colab_bundle.sh` (`projet-code.zip` et
# `fer2013.zip`), puis les extrait.

# %%
# Setup automatique : Colab (upload des deux archives) ou exécution locale (Mac).
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    COLAB_ROOT = Path("/content/fer2013-project")
    expected = {"projet-code.zip": COLAB_ROOT / "src", "fer2013.zip": COLAB_ROOT / "data/train"}
    missing = [name for name, folder in expected.items() if not folder.is_dir()]
    if missing:
        from io import BytesIO
        from zipfile import ZipFile
        from google.colab import files

        print("Sélectionner :", ", ".join(missing), "(produits par scripts/colab_bundle.sh).")
        for name, content in files.upload().items():
            ZipFile(BytesIO(content)).extractall(COLAB_ROOT)
    os.chdir(COLAB_ROOT)

import tensorflow as tf

print(f"Environnement : {'Colab' if IN_COLAB else 'local'} ; "
      f"GPU TensorFlow : {tf.config.list_physical_devices('GPU') or 'aucun, calcul sur CPU'}")

# %%
from pathlib import Path
import json
import sys


# Le notebook peut être ouvert depuis la racine ou depuis notebooks/.
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "src").is_dir():
    raise RuntimeError("Ouvrir ce notebook depuis la racine du dépôt ou son dossier notebooks/.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data import CLASS_NAMES, K, load_dataset


DATA_DIR = PROJECT_ROOT / "data"  # Dossier FER2013 contenant train/ et test/.
if not all((DATA_DIR / split).is_dir() for split in ("train", "test")):
    raise FileNotFoundError(
        f"FER2013 incomplet ou absent dans {DATA_DIR}. "
        f"Extraire le dataset pour obtenir {DATA_DIR / 'train'} et {DATA_DIR / 'test'}, "
        "chacun avec les sept sous-dossiers de classes (voir README.md). "
        "Puis redémarrer le kernel et exécuter toutes les cellules dans l'ordre."
    )
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(DATA_DIR)

assert X_train.shape[1:] == (48, 48, 1)
assert X_val.shape[1:] == X_test.shape[1:] == X_train.shape[1:]
assert K == len(CLASS_NAMES)
print(f"Train : {X_train.shape[0]}, validation : {X_val.shape[0]}, test : {X_test.shape[0]}")
print(f"Dimensions : {X_train.shape[1:]}, format : {X_train.dtype}, pixels dans [0, 1]")
print("Classes :", ", ".join(CLASS_NAMES))

sources = sorted(path for split in ("train", "test") for path in (DATA_DIR / split).glob("*/*"))
assert len(sources) == len(y_train) + len(y_val) + len(y_test)
formats = set()
for path in sources:
    with Image.open(path) as image:
        formats.add((image.format, image.mode, image.size))
print(f"Total : {len(sources)} images ({len(y_train) + len(y_val)} train officiel + {len(y_test)} test)")
print("Fichiers source (format, mode, taille) :", sorted(formats))

# %% [markdown]
# Les fichiers source sont tous des JPEG en niveaux de gris (mode `L`) de 48 x 48
# pixels, d'après la cellule précédente. Le redimensionnement conserve
# néanmoins cette taille d'entrée fixe si une image diffère. La normalisation en
# `float32` dans [0, 1] stabilise l'entraînement. Les émotions sont encodées par
# des entiers dans l'ordre de `CLASS_NAMES`, compatible avec la loss sparse. Le
# dossier `test/` officiel est conservé intact : seuls les exemples de `train/`
# sont séparés de façon stratifiée (15 %, seed 42) pour former la validation.

# %%
counts = pd.DataFrame(
    {
        "train officiel": np.bincount(np.concatenate((y_train, y_val)), minlength=K),
        "test": np.bincount(y_test, minlength=K),
    },
    index=CLASS_NAMES,
)
plt.figure(figsize=(9, 4))
plt.bar(CLASS_NAMES, counts["train officiel"])
plt.title("Répartition par classe dans train + validation")
plt.ylabel("Nombre d'images")
plt.xticks(rotation=30)
plt.show()
print(pd.concat((counts, counts.sum().to_frame("total").T)).to_string())
print("La classe disgust est particulièrement minoritaire : ce déséquilibre sera suivi par classe.")

# %%
fig, axes = plt.subplots(K, 5, figsize=(10, 14))
for label, class_name in enumerate(CLASS_NAMES):
    examples = X_train[y_train == label][:5]
    for axis, image in zip(axes[label], examples):
        axis.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    axes[label, 0].set_ylabel(class_name)
fig.suptitle("Cinq exemples par classe", y=0.995)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Phase 2  - baseline MLP
#
# Ce réseau simple sert de point de comparaison pour le futur CNN.
# `Flatten` transforme chaque image `(48, 48, 1)` en un vecteur de 2 304 pixels :
# les valeurs restent identiques, mais les relations spatiales ne sont plus
# représentées explicitement. Une couche Dense relie chaque entrée à chaque
# neurone. Ses **poids** règlent l'influence des entrées et ses **biais** décalent
# les sommes calculées : `z = xW + b`.
#
# **Binaire ou multiclasse.** En classification binaire, un seul neurone de sortie
# avec une **sigmoïde** `σ(z) = 1 / (1 + e^(-z))` donne la probabilité de la
# classe 1. Ici, il y a sept expressions : la dernière couche compte donc un
# neurone par classe, `Dense(7, activation="softmax")`, et la **softmax**
# `softmax(z_i) = e^(z_i) / Σ_j e^(z_j)` rend les sept sorties positives et de
# somme 1. La classe prédite est celle de plus forte probabilité.
#
# La **propagation avant** calcule ces sommes puis leurs activations.
# Les 128 neurones cachés utilisent **ReLU**, `max(0, z)`, pour introduire une
# non-linéarité. La sortie **softmax** transforme les sept scores en valeurs
# positives dont la somme vaut 1, dans l'ordre de `CLASS_NAMES`. Ces valeurs
# ne garantissent pas une confiance calibrée : une prédiction à 90 % ne signifie
# pas automatiquement que 90 % des prédictions semblables seront correctes.
#
# La **loss** mesure l'écart avec la classe attendue. Avec nos labels entiers,
# `sparse_categorical_crossentropy` pénalise une faible probabilité de la bonne
# classe sans convertir les labels en vecteurs one-hot. La **rétropropagation**
# calcule les gradients de cette perte par rapport aux poids et aux biais.
# La **descente de gradient** les modifie pour réduire la perte ; Adam adapte
# les mises à jour à partir des gradients. Le learning rate règle leur amplitude.
# Un batch contient 64 images ; une epoch parcourt tout l'entraînement fourni.
# La validation mesure la généralisation sans mettre à jour les poids.

# %%
from src.data import SEED
from src.models import build_mlp
from src.train import train_mlp


model = build_mlp(hidden_units=128, seed=SEED)
assert model.input_shape == (None, 48, 48, 1)
assert model.output_shape == (None, K)
print(f"MLP construit : {model.count_params():,} paramètres, {K} classes.")


def load_history(run_id):
    """Recharge un historique sauvegardé sans déclencher d'entraînement."""
    path = PROJECT_ROOT / f"training/logs/{run_id}_history.json"
    if not path.is_file():
        print(f"{run_id} : historique absent ({path}) ; courbes et résultats indisponibles.")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload["history"]
    assert payload["id"] == run_id
    assert set(history) == {"loss", "accuracy", "val_loss", "val_accuracy"}
    epochs_ran = len(history["loss"])
    assert epochs_ran > 0 and all(len(values) == epochs_ran and np.isfinite(values).all() for values in history.values())
    for metric, values in history.items():
        assert np.all(np.asarray(values) >= 0)
        if "accuracy" in metric:
            assert np.all(np.asarray(values) <= 1)
    if run_id in ("C0", "E1", "E2", "E3"):
        best_index = int(np.argmin(history["val_loss"]))
        assert payload["best_epoch"] == best_index + 1
        assert payload["config"]["epochs_ran"] == epochs_ran
        assert payload["best_metrics"] == {key: values[best_index] for key, values in history.items()}
    return payload


def plot_history(run_id, history, best_epoch=None):
    """Affiche la perte et l'accuracy train/validation des epochs réellement exécutées."""
    epoch_numbers = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Perte", "Accuracy")):
        axis.plot(epoch_numbers, history[metric], label="Train")
        axis.plot(epoch_numbers, history[f"val_{metric}"], label="Validation")
        if best_epoch is not None:
            axis.axvline(best_epoch, color="gray", linestyle="--", label="Meilleure val_loss")
        axis.set(title=f"{run_id} - {title}", xlabel="Epoch", ylabel=metric, xticks=list(epoch_numbers)[::max(1, len(epoch_numbers) // 15)])
        axis.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Smoke test local.** Depuis la racine : `python -m src.train`.
# Il réutilise le split de phase 1, limite train à 256 images et validation à 64,
# puis entraîne un MLP neuf pendant une seule epoch. Il écrit uniquement
# `smoke_history.json` et `smoke.keras`, jamais une ligne B0. Il vérifie le
# fonctionnement technique ; ses métriques ne sont pas des résultats de baseline.
# Les images du chargeur sont déjà normalisées et ne sont pas divisées à nouveau.
#
# **B0 sur Colab.** Suivre le [guide B0](../docs/COLAB.md)
# (la version locale non poussée doit être uploadée). Garder les paquets natifs de Colab, placer FER2013 dans
# `data/`, activer le GPU, puis exécuter les cellules de phase 1 pour obtenir
# le split complet. Passer `RUN_B0` à `True` ci-dessous lance explicitement
# un MLP neuf avec seed 42 (Python, NumPy et TensorFlow), Adam à 0,001,
# batch 64 et 5 epochs. Le test officiel n'est transmis à aucun entraînement.
# La seed et les opérations déterministes rendent le run reproductible dans
# un même environnement ; les plateformes et versions peuvent différer.
#
# L'historique réel est sauvegardé dans `training/logs/B0_history.json` et
# les poids finaux dans `training/checkpoints/B0.keras`. Le CSV
# `training/logs/experiments.csv` rapporte la validation de la **dernière epoch**,
# accuracy et loss de la même epoch, sans restauration des meilleurs poids.
# Une réexécution remplace la ligne B0 et ses fichiers, en gardant les autres
# expériences. Avec `RUN_B0=False`, les courbes utilisent le JSON sauvegardé.

# %%
RUN_B0 = False  # Activer explicitement sur Colab pour le run complet.
B0_PARAMS = dict(epochs=5, batch_size=64, learning_rate=1e-3, hidden_units=128, seed=42)
b0_history = None
if RUN_B0:
    import google.colab  # Le run complet est réservé à Colab.

    model, b0_history = train_mlp(
        X_train, y_train, X_val, y_val,
        **B0_PARAMS,
        log_dir=PROJECT_ROOT / "training/logs",
        checkpoint_dir=PROJECT_ROOT / "training/checkpoints",
    )
else:
    print("B0 : relecture des résultats sauvegardés, sans entraînement.")
b0_payload = load_history("B0")
b0_history = None if b0_payload is None else b0_payload["history"]

# %%
if b0_history is not None:
    plot_history("B0", b0_history)
    print(f"B0  - dernière epoch {len(b0_history['loss'])} : "
          f"val_acc={b0_history['val_accuracy'][-1]:.6f}, "
          f"val_loss={b0_history['val_loss'][-1]:.6f}")

# %% [markdown]
# **Résultats B0 récupérés.** Les cinq epochs du JSON vérifié font diminuer
# la loss train de 1,782291 à 1,655288 et la loss validation de 1,730711 à
# 1,660339. L'accuracy validation passe de 0,309728 à 0,360808, avec un léger
# recul à l'epoch 4. À l'epoch 5, l'accuracy train vaut 0,350135. Aucun écart
# croissant train/validation n'indique un surapprentissage marqué sur ce run.
# Ces scores modestes servent de référence, sans seuil de performance imposé.
# Les métriques du modèle rechargé concordent sur la validation issue du train ;
# le test officiel n'a pas été évalué. Les preuves et leurs limites figurent
# dans le guide ; les courbes ci-dessus proviennent du JSON sauvegardé.
# La validation globale du notebook Colab demeure une étape distincte du suivi.

# %% [markdown]
# ## Phase 3  - CNN
#
# Le MLP aplatit l'image dès l'entrée et ne représente donc plus la position
# relative des pixels. Un réseau convolutif (CNN) la conserve. Sa première partie
# extrait des caractéristiques locales, sa seconde partie classe l'image à partir
# de ces caractéristiques.
#
# **Filtres et convolution.** Un filtre est une petite grille de poids appris.
# La convolution le fait glisser sur l'image et calcule, à chaque position, la
# somme pondérée des pixels couverts plus un biais. Les mêmes poids servent à
# toutes les positions, donc un motif appris est détecté où qu'il apparaisse.
#
# **Feature maps.** Chaque filtre produit une carte de caractéristiques (feature
# map) qui indique où son motif répond fortement. Une couche de 32 filtres produit
# 32 cartes, empilées comme des canaux.
#
# **Kernel, stride et padding.** Le kernel est la taille du filtre, ici 3 x 3.
# Le stride est le pas du glissement, ici 1 pixel (valeur par défaut de Keras).
# Le padding `same` ajoute des zéros autour de l'entrée pour que la sortie garde
# la même hauteur et la même largeur.
#
# **ReLU.** Après chaque convolution, `max(0, z)` garde les réponses positives et
# annule les autres, comme dans la couche cachée du MLP. Sans cette non-linéarité,
# l'enchaînement des couches resterait une seule opération linéaire.
#
# **Pooling.** Le MaxPooling 2 x 2 garde le maximum de chaque bloc de 2 x 2 pixels,
# avec un pas de 2. Il divise la hauteur et la largeur par deux sans paramètre
# appris. Le réseau devient moins sensible à un petit décalage du motif et les
# couches suivantes traitent moins de positions.
#
# **Flatten, Dense et sortie.** Après le dernier bloc, `Flatten` met les cartes
# bout à bout dans un vecteur. Une couche Dense de 128 neurones ReLU combine ces
# caractéristiques. La sortie Dense(7, softmax) donne une probabilité par
# expression, comme pour B0.

# %%
from src.models import build_cnn


cnn = build_cnn()
cnn.summary()
assert cnn.input_shape == (None, 48, 48, 1)
assert cnn.output_shape == (None, K)
print(f"CNN C0 : {cnn.count_params():,} paramètres, contre {model.count_params():,} pour le MLP B0.")

# %% [markdown]
# **Évolution des dimensions.** Le tableau précédent donne la forme de sortie et
# le nombre de paramètres de chaque couche. Chaque convolution garde la taille
# grâce au padding `same` et chaque pooling la divise par deux, soit
# 48 → 24 → 12 → 6. Le nombre de cartes passe de 1 canal à 32, 64 puis 128.
# `Flatten` produit donc un vecteur de 6 x 6 x 128 = 4 608 valeurs.
#
# Une convolution compte `(k x k x C_entrée + 1) x C_sortie` paramètres, avec un
# poids par case du filtre et par canal d'entrée, plus un biais par filtre.
# Une couche Dense compte `(entrées + 1) x neurones` paramètres.
#
# | Couche | Calcul | Paramètres |
# |---|---|---|
# | Conv2D 32 | (3 x 3 x 1 + 1) x 32 | 320 |
# | Conv2D 64 | (3 x 3 x 32 + 1) x 64 | 18 496 |
# | Conv2D 128 | (3 x 3 x 64 + 1) x 128 | 73 856 |
# | Dense 128 | (4 608 + 1) x 128 | 589 952 |
# | Dense 7 | (128 + 1) x 7 | 903 |
# | **Total** | | **683 527** |
#
# Comme les poids des filtres sont partagés entre positions, les trois
# convolutions ne comptent que 92 672 paramètres. La couche Dense cachée
# concentre 86 % du total.
#
# > *Nos choix :*
# > - **Trois blocs conv/pooling.** Chaque bloc voit une zone plus large de
# >   l'image d'origine, car le pooling espace les positions lues par la
# >   convolution suivante. Une sortie de la première convolution dépend de
# >   3 x 3 pixels, de la deuxième de 8 x 8 et de la troisième de 18 x 18.
# >   Les premiers filtres peuvent ainsi répondre à des
# >   contours et les suivants à des zones plus larges, comme les yeux ou la
# >   bouche. Ce rôle reste une hypothèse, car les filtres appris ne sont pas
# >   encore visualisés.
# > - **32, 64 puis 128 filtres.** Le nombre de cartes double quand la résolution
# >   est divisée par deux, pour compenser la perte de détails spatiaux par
# >   davantage de motifs.
# > - **Kernel 3 x 3 et padding `same`.** Un petit filtre a peu de poids (9 par
# >   canal d'entrée), et l'empilement des blocs élargit ensuite la zone vue. Le
# >   padding garde les bords du visage et rend les dimensions simples à suivre.
# > - **Arrêt à 6 x 6.** Avec deux blocs seulement, `Flatten` produirait
# >   12 x 12 x 64 = 9 216 valeurs et le modèle 1 199 495 paramètres. Le troisième
# >   bloc réduit donc le total de 43 % tout en conservant une carte où le haut et
# >   le bas du visage restent distincts.
# > - **Dense de 128 neurones.** C'est la largeur de la couche cachée de B0. La
# >   comparaison porte ainsi surtout sur l'extraction convolutive.
#
# > *Notre hypothèse :* C0 compte environ 2,3 fois plus de paramètres que B0
# > (683 527 contre 295 943) et exploite la structure spatiale. Il devrait donc
# > dépasser l'accuracy de validation de B0 (0,360808). Sans régularisation, un
# > surapprentissage est aussi possible. Les courbes de la phase 4 permettront de
# > vérifier ces deux points.

# %% [markdown]
# ## Phase 4 - entraînement C0 et comparaison
#
# **Protocole.** Nous conservons le split stratifié, les pixels déjà normalisés
# et les labels entiers. La cross-entropie est `-log(p_classe_attendue)` : elle
# pénalise fortement une erreur confiante. L'accuracy mesure la proportion de
# classes prédites correctement ; deux modèles de même accuracy peuvent donc
# avoir des losses différentes. Adam adapte la mise à jour de chaque poids aux
# gradients, avec un learning rate de 0,001. Un batch de 64 images détermine un
# gradient avant une mise à jour ; une epoch parcourt les 24 402 images train
# (382 batches, le dernier incomplet). La validation utilise les 4 307 images
# réservées du train officiel, sans apprentissage. Le test reste réservé.
#
# **Callbacks.** Après chaque epoch, `ModelCheckpoint(save_best_only=True)`
# conserve le modèle ayant la plus petite `val_loss` (`mode="min"`).
# `EarlyStopping(patience=5, restore_best_weights=True)` arrête après cinq epochs
# sans amélioration de cette même loss et restaure les meilleurs poids.
# Le plafond de 30 epochs borne la durée ; l'historique garde toutes les epochs
# réellement exécutées, y compris celles après la meilleure. En cas d'égalité,
# la première epoch minimale est retenue. L'accuracy affichée pour C0 vient
# de cette epoch, même si une autre epoch atteint une accuracy plus haute.
# B0 conserve ses cinq epochs et ses poids finaux, sans ces callbacks.
#
# **Lecture des courbes.** La loss doit se lire avec l'accuracy : diminuer la
# loss peut améliorer les probabilités sans changer la classe gagnante.
# Une loss train qui baisse alors que la loss validation remonte suggère un
# surapprentissage. Des résultats faibles et proches des deux côtés peuvent
# suggérer un sous-apprentissage ; un plateau seul ne suffit pas à conclure.
# Les fluctuations de validation ne justifient pas l'utilisation du test.
# La ligne verticale indique l'epoch des poids retenus, pas la fin du run.
# La comparaison B0/C0 utilise des budgets et des règles de sélection différents ;
# elle décrit ces deux protocoles, sans isoler le seul effet de l'architecture.
#
# **Exécution.** Guide et cellules exactes :
# [section C0 du guide Colab](../docs/COLAB.md#c0--phase-4-sur-colab-gpu).
# `RUN_C0=False` recharge l'historique sans entraîner. S'il manque, aucun résultat
# C0 n'est déduit de l'hypothèse de phase 3. Smoke local temporaire :
# `python -m src.train --model cnn`. Contrôle des vrais artefacts :
# `python -m src.train --check-c0` (validation complète, aucun entraînement).

# %%
from src.train import train_cnn, verify_cnn_run, _read_experiments


RUN_C0 = False  # Activer uniquement pour le vrai run Colab GPU.
C0_ARCHITECTURE = dict(filters=(32, 64, 128), kernel_size=3, dense_units=128, seed=42)
C0_PARAMS = dict(epochs=30, batch_size=64, learning_rate=1e-3, seed=42)
LOG_DIR = PROJECT_ROOT / "training/logs"
CHECKPOINT_DIR = PROJECT_ROOT / "training/checkpoints"
if RUN_C0:
    import google.colab
    import hashlib
    import tensorflow as tf
    from importlib.metadata import version

    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "GPU absent : choisir un runtime Colab GPU avant C0."
    assert len(y_train) == 24402 and len(y_val) == 4307
    assert set(np.unique(y_train)) == set(np.unique(y_val)) == set(range(K))
    preserved_paths = [path for directory in (LOG_DIR, CHECKPOINT_DIR) for path in directory.glob("B0*") if path.is_file()]
    preserved_hashes = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in preserved_paths}
    previous_rows = [row for row in _read_experiments(LOG_DIR / "experiments.csv") if row["id"] != "C0"]
    assert (LOG_DIR / "B0_history.json").is_file() and any(row["id"] == "B0" for row in previous_rows)
    assert (CHECKPOINT_DIR / "B0.keras").is_file(), "Transférer aussi le checkpoint B0 pour sa préservation."
    trace = {
        "python": sys.version, "gpu_devices": [str(device) for device in gpus],
        "versions": {package: version(package) for package in ("tensorflow", "keras", "numpy", "scikit-learn")},
        "train_size": len(y_train), "val_size": len(y_val), "test_used": False,
        "preserved_artifacts_sha256": preserved_hashes, "preserved_rows": previous_rows,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = LOG_DIR / "C0_colab_environment.txt"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print("GPU :", gpus, "; split complet :", len(y_train), len(y_val))
    cnn, c0_history = train_cnn(
        lambda: build_cnn(**C0_ARCHITECTURE), X_train, y_train, X_val, y_val,
        **C0_PARAMS, log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR,
    )
    restored_cnn = tf.keras.models.load_model(CHECKPOINT_DIR / "C0.keras")
    for saved, returned in zip(restored_cnn.get_weights(), cnn.get_weights()):
        np.testing.assert_array_equal(saved, returned)
    assert preserved_hashes == {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in preserved_paths}
    assert previous_rows == [row for row in _read_experiments(LOG_DIR / "experiments.csv") if row["id"] != "C0"]
    trace["verification"] = verify_cnn_run(X_val, y_val, log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR)
    trace["preserved_after_run"] = True
    trace["checkpoint_weights_equal_returned"] = True
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

c0_payload = load_history("C0")
c0_history = None if c0_payload is None else c0_payload["history"]

# %%
if c0_history is not None:
    plot_history("C0", c0_history, c0_payload["best_epoch"])
    print(f"C0 : {len(c0_history['loss'])} epochs exécutées ; meilleure epoch sur val_loss : {c0_payload['best_epoch']}.")

comparison_rows = []
for run_id, payload, params in (("B0", b0_payload, model.count_params()), ("C0", c0_payload, cnn.count_params())):
    if payload is None:
        print(f"{run_id} : absent de la comparaison, historique indisponible.")
        continue
    history = payload["history"]
    index = len(history["loss"]) - 1 if run_id == "B0" else payload["best_epoch"] - 1
    comparison_rows.append({
        "id": run_id, "epoch retenue": index + 1, "epochs exécutées": len(history["loss"]),
        "val_acc": history["val_accuracy"][index], "val_loss": history["val_loss"][index], "paramètres": params,
    })
comparison = pd.DataFrame(comparison_rows)
if not comparison.empty:
    from IPython.display import display

    display(comparison)
print("B0 : dernière epoch ; C0 : première meilleure epoch sur val_loss. Accuracy et loss issues de la même epoch.")

# %% [markdown]
# **Analyse du vrai historique.** La cellule suivante rapporte uniquement les
# valeurs des JSON disponibles. Les variations des dernières epochs décrivent
# la fin de l'entraînement ; les métriques de comparaison décrivent les poids
# retenus.
#
# **C0 vérifié le 7 octobre 2026 sur GPU Tesla T4.** Douze epochs ont été
# exécutées. La meilleure loss validation est 1,232023 à l'epoch 7, avec une
# accuracy validation de 0,556536 ; ces deux valeurs décrivent le checkpoint.
# Après l'epoch 7, la loss train continue de baisser (0,884061 → 0,363075),
# tandis que la loss validation augmente jusqu'à 1,898607. L'accuracy train
# atteint 0,874027, alors que celle de validation reste autour de 0,55 :
# cet écart croissant indique un surapprentissage sur ce run. Après cinq epochs
# sans meilleure val_loss, l'arrêt anticipé termine à l'epoch 12 et restaure
# les poids de l'epoch 7. La plus haute accuracy validation est à l'epoch 11
# (0,558161), mais sa loss vaut 1,643386 : cette epoch n'est donc pas retenue.
#
# Par rapport à B0 à sa dernière epoch (5), C0 à son epoch retenue (7) gagne
# 19,57 points d'accuracy validation et réduit la loss de 0,428316. C0 compte
# 683 527 paramètres contre 295 943 pour B0. Notre hypothèse d'une meilleure
# accuracy est confirmée pour ces deux runs, mais leurs budgets et règles de
# sélection différents ne permettent pas d'attribuer tout le gain à la seule
# architecture. La validation complète du checkpoint rechargé concorde avec
# le JSON dans Colab et en local ; aucune conclusion n'est tirée sur le test.

# %%
if c0_history is not None:
    best_index = c0_payload["best_epoch"] - 1
    print(f"C0 - loss train : {c0_history['loss'][0]:.6f} → {c0_history['loss'][-1]:.6f} ; "
          f"loss validation : {c0_history['val_loss'][0]:.6f} → {c0_history['val_loss'][-1]:.6f}.")
    print(f"À l'epoch retenue {best_index + 1} : accuracy train={c0_history['accuracy'][best_index]:.6f}, "
          f"validation={c0_history['val_accuracy'][best_index]:.6f} ; val_loss={c0_history['val_loss'][best_index]:.6f}.")
    if b0_history is not None:
        print(f"Écart C0 − B0 sur les epochs retenues : "
              f"val_acc={c0_history['val_accuracy'][best_index] - b0_history['val_accuracy'][-1]:+.6f}, "
              f"val_loss={c0_history['val_loss'][best_index] - b0_history['val_loss'][-1]:+.6f}.")
else:
    print("Analyse C0 en attente du vrai historique Colab GPU ; aucun score C0 disponible.")

# %% [markdown]
# ## Phase 5 - évaluation et analyse des erreurs
#
# L'accuracy ne suffit pas ici, car les classes sont déséquilibrées : disgust
# ne compte que 436 images dans le train officiel, contre 7 215 pour happy.
# Un modèle peut obtenir une accuracy correcte en ignorant presque une petite
# classe. Nous regardons donc les performances par expression.
#
# Pour une classe donnée, la **precision** est la part de prédictions justes
# parmi les images prédites dans cette classe. Le **recall** est la part des
# images de cette classe retrouvées par le modèle. Le **F1** est leur moyenne
# harmonique : il reste bas si l'une des deux est faible. La **matrice de
# confusion** croise la vraie classe (ligne) et la classe prédite (colonne).
# Nous la normalisons par ligne, donc la diagonale donne le recall de chaque
# classe.
#
# Cette analyse porte sur la **validation** (4 307 images issues du train
# officiel). Le test reste réservé à une seule évaluation du modèle final.
# Les exemples affichés sont les prédictions les plus confiantes, correctes
# ou erronées : une erreur commise avec une forte probabilité est la plus
# instructive.

# %%
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.evaluate import class_report, confident_examples, evaluate
from src.train import RELOAD_ATOL, RELOAD_RTOL


def show_examples(title, X, y, y_pred, confidence, indices):
    """Affiche des images avec vraie classe, classe prédite et probabilité."""
    if len(indices) == 0:
        print(f"{title} : aucun exemple.")
        return
    fig, axes = plt.subplots(1, len(indices), figsize=(2 * len(indices), 2.8), layout="constrained")
    for number, (axis, index) in enumerate(zip(np.atleast_1d(axes), indices), start=1):
        axis.imshow(X[index, ..., 0], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{number}. vrai : {CLASS_NAMES[y[index]]}\nprédit : {CLASS_NAMES[y_pred[index]]}\n"
                       f"p = {confidence[index]:.2f}", fontsize=8)
        axis.axis("off")
    fig.suptitle(title)
    plt.show()


def show_analysis(title, X, y, y_pred, confidence):
    """Matrice de confusion, paires confondues, rapport par classe et exemples."""
    y_pred, confidence = np.asarray(y_pred), np.asarray(confidence)
    counts = confusion_matrix(y, y_pred, labels=range(K))
    rates = counts / counts.sum(axis=1, keepdims=True)
    fig, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(rates, display_labels=CLASS_NAMES).plot(
        ax=axis, values_format=".2f", cmap="Blues", colorbar=False, xticks_rotation=30,
    )
    axis.set(title=f"{title} - matrice normalisée par vraie classe",
             xlabel="Classe prédite", ylabel="Vraie classe")
    plt.tight_layout()
    plt.show()
    pairs = [(counts[i, j], rates[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
             for i in range(K) for j in range(K) if i != j]
    print(f"{title} : {counts.sum()} images ; paires les plus confondues (vrai → prédit) :")
    for count, rate, true_name, predicted_name in sorted(pairs, reverse=True)[:5]:
        print(f"  {true_name} → {predicted_name} : {count} images ({rate:.1%} de {true_name})")
    print(class_report(y, y_pred).round(3).to_string())
    for correct, label in ((True, "correctes"), (False, "erronées")):
        indices = confident_examples(y, y_pred, confidence, correct)
        show_examples(f"{title} - prédictions {label} les plus confiantes", X, y, y_pred, confidence, indices)
    return counts


validation_results = {}
for run_id in ("B0", "C0"):
    path = CHECKPOINT_DIR / f"{run_id}.keras"
    if not path.is_file():
        print(f"{run_id} : checkpoint absent ({path}) ; analyse de validation indisponible.")
        continue
    validation_results[run_id] = evaluate(tf.keras.models.load_model(path), X_val, y_val)
    print(f"{run_id} validation : accuracy={validation_results[run_id]['accuracy']:.6f}, "
          f"loss={validation_results[run_id]['loss']:.6f}")
if "C0" in validation_results and c0_payload is None:
    print("C0 : historique absent ; concordance du checkpoint non vérifiée, analyse non affichée.")
elif "C0" in validation_results:
    for key in ("accuracy", "loss"):
        np.testing.assert_allclose(validation_results["C0"][key], c0_payload["best_metrics"][f"val_{key}"],
                                   rtol=RELOAD_RTOL, atol=RELOAD_ATOL)
    c0_val = validation_results["C0"]
    show_analysis("C0 validation", X_val, y_val, c0_val["y_pred"], c0_val["confidence"])

# %%
if {"B0", "C0"} <= validation_results.keys():
    f1 = pd.DataFrame({run_id: class_report(y_val, validation_results[run_id]["y_pred"])["f1-score"]
                       for run_id in ("B0", "C0")}).loc[[*CLASS_NAMES, "macro avg", "weighted avg"]]
    f1["écart C0 − B0"] = f1["C0"] - f1["B0"]
    print("F1 par classe sur la validation :")
    print(f1.round(3).to_string())

# %% [markdown]
# **Analyse de C0 sur la validation.** La réévaluation du checkpoint retrouve
# l'accuracy de l'epoch retenue (0,556536) et sa loss (1,232023).
#
# > *Nos observations :*
# > - **Classes bien reconnues.** happy obtient le meilleur F1 (0,754, recall
# >   0,803), suivi de surprise (F1 0,688). Les huit prédictions correctes les
# >   plus confiantes sont toutes des happy à p = 1,00, avec un large sourire.
# > - **Confusions principales.** Les paires les plus fréquentes relient sad,
# >   neutral et fear : sad → neutral (148 images, 20,4 % des sad), sad → fear
# >   (113), fear → sad (111) et neutral → sad (107). angry → fear arrive ensuite
# >   (100 images, 16,7 % des angry). Ces confusions vont dans les deux sens.
# > - **Classes difficiles.** disgust a le F1 le plus bas (0,353). Son recall
# >   n'est que de 0,231 alors que sa precision atteint 0,750. Le modèle prédit
# >   donc rarement disgust : 20 prédictions sur 4 307, dont 15 justes (0,231 x 65
# >   et 15 / 0,750), et la colonne disgust de la matrice est quasi nulle pour les
# >   autres classes. Ses images partent surtout vers fear et sad (0,22 chacune).
# >   fear (0,395), sad (0,436) et angry (0,439) suivent.
# > - **Effet du déséquilibre.** Le recall moyen par classe (macro, 0,494) est
# >   inférieur à l'accuracy (0,557), car la classe majoritaire happy est la
# >   mieux reconnue. happy attire aussi une partie des autres classes
# >   (12 à 13 % des angry, neutral et sad).
# > - **Erreurs confiantes.** Les huit erreurs les plus confiantes sont toutes
# >   prédites happy avec p ≥ 0,99. Les visages 1, 2, 4, 6 et 8 montrent un
# >   sourire ou des dents visibles, alors qu'ils sont annotés surprise, fear,
# >   angry ou neutral. L'image 7 n'est pas un visage mais un pictogramme
# >   d'avertissement, annoté neutral.
# > - **Gain par rapport à B0.** Le F1 macro passe de 0,244 à 0,512. Les plus
# >   forts gains concernent angry (+0,404), fear (+0,380) et disgust (+0,353),
# >   trois classes que B0 ne reconnaissait presque pas (F1 ≤ 0,035).
#
# > *Nos hypothèses :*
# > - sad, neutral et fear partagent souvent une bouche fermée et des sourcils
# >   peu marqués. En 48 x 48 pixels et en niveaux de gris, ces différences fines
# >   pourraient être difficiles à distinguer. Cette explication n'est pas testée.
# > - Le modèle semble associer fortement bouche ouverte ou dents visibles à
# >   happy, ce qui expliquerait les erreurs confiantes vers cette classe.
# > - Plusieurs erreurs confiantes paraissent ambiguës, et l'image 7 montre un
# >   bruit d'annotation dans FER2013. Une partie des erreurs ne serait donc pas
# >   corrigeable par le modèle seul.
# > - disgust dispose de peu d'exemples (436 en train officiel, 65 en
# >   validation), ce qui pourrait expliquer son faible recall. Son F1 repose sur
# >   65 images seulement et reste donc peu stable.
#
# > *Limites :* ces résultats décrivent un seul run (seed 42) et le checkpoint
# > de l'epoch 7, déjà en surapprentissage d'après la phase 4. Les phases 6 et 7
# > pourront réutiliser cette analyse pour comparer les candidats.

# %% [markdown]
# ## Phase 6 - Trois expériences à partir de C0
#
# Chaque run repart de poids neufs avec la seed 42, le même split stratifié,
# `/255.0`, les sept classes et les callbacks de C0. Adam, batch 64 et 30 epochs
# maximum restent communs ; EarlyStopping attend 5 epochs sans baisse de val_loss.
# Aucun run ne prend les poids ou les réglages d'une autre expérience.
#
# | Id | Seule différence avec C0 | Hypothèse à vérifier |
# |---|---|---|
# | E1 | Dense 128 → 64 | Moins de paramètres peut limiter la mémorisation. |
# | E2 | Dropout 0,3 après Dense 128 | Masquer 30 % des activations au train peut réduire le surapprentissage. |
# | E3 | Learning rate 0,001 → 0,0005 | Des pas plus petits peuvent stabiliser l'optimisation. |
#
# **Critère fixé avant les runs :** plus faible val_loss du checkpoint ; puis
# accuracy validation maximale si égalité exacte ; puis id pour un ordre stable.
# On compare donc toujours loss et accuracy de la même epoch sauvegardée.
# La loss tient compte des probabilités, notamment des erreurs trop confiantes.
# Une seule seed ne permet pas d'affirmer qu'un petit écart est significatif.
#
# **Colab GPU :** transférer le code local et les artefacts C0/B0 (voir README),
# exécuter les cellules de chargement et les définitions précédentes, puis activer
# explicitement RUN_E1, RUN_E2 et RUN_E3 ci-dessous. Ne pas activer B0, C0 ou le test.
# Avec les flags False, les historiques sont relus et les checkpoints disponibles
# sont contrôlés sur la validation, sans fit. Une absence ne produit aucun score.

# %%
import hashlib
import tensorflow as tf
from src.train import PHASE6_EXPERIMENTS, SELECTION_CRITERION, train_experiment

RUN_E1 = False
RUN_E2 = False
RUN_E3 = False


def run_phase6(run_id, enabled):
    """Lance explicitement un run GPU en préservant les artefacts et lignes des autres ids."""
    if not enabled:
        return
    import google.colab
    assert tf.config.list_physical_devices("GPU"), "Choisir un runtime Colab GPU."
    assert not RUN_B0 and not RUN_C0
    protected = [path for directory in (LOG_DIR, CHECKPOINT_DIR) for path in directory.glob("*")
                 if path.is_file() and path.stem != run_id and not path.name.startswith(run_id + "_")
                 and path.name != "experiments.csv"]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    rows = [row for row in _read_experiments(LOG_DIR / "experiments.csv") if row["id"] != run_id]
    trained, _ = train_experiment(run_id, X_train, y_train, X_val, y_val,
                                  log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR)
    saved = tf.keras.models.load_model(CHECKPOINT_DIR / f"{run_id}.keras")
    for actual, expected in zip(saved.get_weights(), trained.get_weights()):
        np.testing.assert_array_equal(actual, expected)
    verify_cnn_run(X_val, y_val, run_id=run_id, X_train=X_train, y_train=y_train,
                   log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    assert rows == [row for row in _read_experiments(LOG_DIR / "experiments.csv") if row["id"] != run_id]

# %%
run_phase6("E1", RUN_E1)

# %%
run_phase6("E2", RUN_E2)

# %%
run_phase6("E3", RUN_E3)

# %%
phase6_payloads = {run_id: load_history(run_id) for run_id in ("C0", "E1", "E2", "E3")}
phase6_csv = {row["id"]: row for row in _read_experiments(LOG_DIR / "experiments.csv")}
phase6_rows = []
phase6_verified = set()
for run_id, payload in phase6_payloads.items():
    if payload is None:
        phase6_rows.append({"id": run_id, "état": "historique absent - run Colab requis"})
        continue
    metrics = payload["best_metrics"]
    checkpoint = CHECKPOINT_DIR / f"{run_id}.keras"
    if checkpoint.is_file():
        verify_cnn_run(X_val, y_val, run_id=run_id, X_train=X_train, y_train=y_train,
                       log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR)
        phase6_verified.add(run_id)
    else:
        print(f"{run_id} : checkpoint absent ({checkpoint}) ; concordance des poids non vérifiée.")
    if run_id != "C0":
        plot_history(run_id, payload["history"], payload["best_epoch"])
    i = payload["best_epoch"] - 1
    gap = payload["history"]["accuracy"][i] - metrics["val_accuracy"]
    change = "Référence commune" if run_id == "C0" else PHASE6_EXPERIMENTS[run_id]["modification"]
    observation = (f"Epoch {i + 1}/{payload['config']['epochs_ran']} ; "
                   f"écart accuracy train/val {100 * gap:.2f} points")
    if run_id != "C0":
        observation += " ; " + phase6_csv.get(run_id, {}).get("observation", "ligne CSV absente")
    phase6_rows.append({"id": run_id, "modification": change, "val_acc": metrics["val_accuracy"],
                        "val_loss": metrics["val_loss"], "paramètres": (683527 if run_id == "C0" else payload["config"]["params"]),
                        "observation": observation, "état": "vérifié" if checkpoint.is_file() else "poids absents"})
phase6_table = pd.DataFrame(phase6_rows)
from IPython.display import display
display(phase6_table)
print("Sélection :", SELECTION_CRITERION)
PHASE6_CANDIDATE_ID = None
if phase6_verified == {"C0", "E1", "E2", "E3"}:
    PHASE6_CANDIDATE_ID = min(phase6_verified, key=lambda run_id: (
        phase6_payloads[run_id]["best_metrics"]["val_loss"],
        -phase6_payloads[run_id]["best_metrics"]["val_accuracy"], run_id))
    chosen = phase6_payloads[PHASE6_CANDIDATE_ID]
    architecture = C0_ARCHITECTURE if PHASE6_CANDIDATE_ID == "C0" else PHASE6_EXPERIMENTS[PHASE6_CANDIDATE_ID]["architecture"]
    print(f"Relais Maxime : candidat provisoire {PHASE6_CANDIDATE_ID}, "
          f"checkpoint {CHECKPOINT_DIR / (PHASE6_CANDIDATE_ID + '.keras')}")
    print("Architecture à reproduire pour A1 :", architecture)
    print("Adam, batch 64, seed 42, epochs max 30, patience 5 ; learning rate :", chosen["config"]["learning_rate"])
else:
    print("Relais du meilleur modèle en attente des quatre checkpoints et contrôles réels ; aucun choix définitif.")

# %% [markdown]
# **Résultats réels du 7 octobre 2026.** E1–E3 ont été exécutés sur Tesla T4,
# TensorFlow 2.21.0 / Keras 3.13.2, avec le split complet de C0. Les empreintes du
# split, configurations, checkpoints et métriques JSON/CSV concordent. La relecture
# locale retrouve les mêmes accuracies et des écarts de loss inférieurs à 2e-7.
# Les artefacts et lignes B0/C0 ont été préservés ; le test officiel reste intact.
#
# | Id | Epoch sauvegardée / exécutées | val_accuracy | val_loss | Paramètres |
# |---|---:|---:|---:|---:|
# | C0 | 7 / 12 | 0,556536 | 1,232023 | 683 527 |
# | E1 | 6 / 11 | 0,542373 | 1,228255 | 388 103 |
# | E2 | 8 / 13 | 0,557232 | 1,195606 | 683 527 |
# | E3 | 7 / 12 | 0,559322 | 1,192440 | 683 527 |
#
# **E1 - capacité réduite.** La Dense 64 réduit les paramètres de 43,22 %. L'écart
# d'accuracy train/validation à l'epoch retenue descend à 5,78 points (11,88 pour
# C0), mais l'accuracy validation perd 1,42 point. La loss ne baisse que de 0,003767.
# Réduire la mémorisation ne suffit donc pas à améliorer la reconnaissance ; ce
# candidat compact n'est pas le meilleur selon notre critère de validation.
#
# **E2 - Dropout.** À paramètres constants, la loss baisse de 0,036417 par rapport
# à C0 ; l'accuracy gagne seulement 0,07 point. Le Dropout semble utile ici pour la
# généralisation des probabilités. Son accuracy train est mesurée avec le masquage
# actif, ce qui limite la comparaison directe des écarts train/validation. Le
# surapprentissage persiste : la loss validation remonte à 1,407054 à l'epoch 13,
# après le minimum de 1,195606 à l'epoch 8. En prédiction, le masquage est désactivé.
#
# **E3 - pas d'optimisation plus petit.** La loss baisse de 0,039583 et l'accuracy
# gagne 0,28 point par rapport à C0. À l'epoch 7, l'accuracy train est 0,623637 contre
# 0,559322 en validation (écart 6,43 points). La dernière val_loss vaut 1,349853 :
# diminuer le learning rate ne supprime pas le surapprentissage, d'où le checkpoint
# de l'epoch 7 plutôt que les poids de l'epoch 12.
#
# **Candidat provisoire : E3**, car sa val_loss 1,192440 est la plus basse. E2 reste
# proche : écart de loss 0,003166 et d'accuracy 0,21 point. Avec une seule seed et
# une validation déjà utilisée pour sélectionner plusieurs candidats, on ne peut
# pas conclure à une supériorité générale ou statistiquement établie de E3.
#
# **Relais à Maxime pour A1.** Référence : `training/checkpoints/E3.keras` (epoch 7),
# configuration et historique : `training/logs/E3_history.json`. Reproduction GPU :
# `train_experiment("E3", X_train, y_train, X_val, y_val)` ou `RUN_E3=True` seul.
# Architecture : filtres (32,64,128), kernel 3, Dense 128, Dropout 0 ; Adam 0,0005,
# batch 64, seed 42, maximum 30 epochs, patience 5 sur val_loss et restauration.
# Maxime construira un modèle neuf de même configuration et ajoutera seulement
# l'augmentation pour A1. Reprendre les poids E3 ajouterait un effet de poursuite
# d'entraînement. Le checkpoint sert à comparer la référence. Le choix définitif
# attend A1 : `FINAL_MODEL_ID=None` et `RUN_TEST=False` restent inchangés.

# %% [markdown]
# ## Phase 7 - data augmentation (A1)
#
# **Pourquoi.** E3 surapprend après l'epoch 7. À cette epoch, son accuracy train
# vaut 0,623637 contre 0,559322 en validation, puis sa val_loss remonte jusqu'à
# 1,349853 à l'epoch 12 (section phase 6). Le réseau mémorise donc une partie des
# 24 402 images d'entraînement. La data augmentation applique à chaque batch des
# transformations aléatoires et plausibles. Le réseau ne revoit presque jamais
# exactement la même image, ce qui devrait limiter cette mémorisation.
#
# **Protocole.** A1 reprend E3 à l'identique : poids neufs (seed 42), même split
# (empreintes vérifiées), même architecture, Adam 0,0005, batch 64, 30 epochs au
# maximum et patience 5 sur val_loss. La seule différence est l'augmentation des
# batches d'entraînement. La validation et le test ne sont jamais augmentés.
# L'augmentation est appliquée dans le pipeline de données et non dans le modèle,
# donc `A1.keras` garde exactement l'architecture d'E3. Reprendre les poids d'E3
# mélangerait l'effet de l'augmentation avec celui d'epochs supplémentaires.
#
# | Transformation | Réglage | Justification |
# |---|---|---|
# | Flip horizontal | une image sur deux | Une expression reste la même en miroir. Le flip vertical est exclu, car un visage à l'envers n'existe pas dans les données. |
# | Rotation | ±18° | Inclinaison de la tête. |
# | Translation | ±15 % | Visage imparfaitement centré. |
# | Zoom | ±15 % | Cadrage plus ou moins serré. |
# | Contraste | ±20 % | Éclairage variable. |
#
# Les zones découvertes par la rotation, la translation ou le zoom sont remplies
# par réflexion des bords, et les pixels restent dans [0, 1].
#
# > *Notre hypothèse :* A1 devrait obtenir une val_loss inférieure à celle d'E3
# > (1,192440), avec une meilleure epoch plus tardive et un surapprentissage plus
# > lent. Ces transformations sont assez fortes, donc la convergence peut aussi
# > ralentir. Si la meilleure epoch approche le plafond de 30, le budget limitera
# > l'interprétation. L'accuracy train d'A1 est mesurée sur des images augmentées,
# > donc plus difficiles : l'écart train/validation n'est pas directement
# > comparable à celui d'E3.

# %%
from src.train import _augmentation

augmented = _augmentation(SEED)(np.repeat(X_train[:1], 8, axis=0), training=True).numpy()
fig, axes = plt.subplots(1, 9, figsize=(15, 2.2), layout="constrained")
for axis, image, title in zip(axes, (X_train[0], *augmented), ("original", *[f"augmentée {i}" for i in range(1, 9)])):
    axis.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1)
    axis.set_title(title, fontsize=8)
    axis.axis("off")
fig.suptitle(f"Huit tirages de l'augmentation A1 sur une image train ({CLASS_NAMES[y_train[0]]})")
plt.show()

# %% [markdown]
# > *Observation :* les huit tirages montrent des miroirs, des rotations, des
# > recadrages et des variations de contraste, et l'expression reste lisible.
# > Ils sont aussi un peu plus flous que l'original, car la rotation, la
# > translation et le zoom ré-échantillonnent les pixels par interpolation.
# > Les images d'entraînement d'A1 diffèrent donc aussi de la validation par leur
# > netteté. L'effet de cet écart n'est pas mesuré séparément.

# %%
RUN_A1 = False  # Activer seul, sur Colab GPU, comme RUN_E1 à RUN_E3.
run_phase6("A1", RUN_A1)

# %%
a1_payload = load_history("A1")
if a1_payload is not None:
    if (CHECKPOINT_DIR / "A1.keras").is_file():
        verify_cnn_run(X_val, y_val, run_id="A1", X_train=X_train, y_train=y_train,
                       log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR)
    else:
        print(f"A1 : checkpoint absent ({CHECKPOINT_DIR / 'A1.keras'}) ; concordance des poids non vérifiée.")
    plot_history("A1", a1_payload["history"], a1_payload["best_epoch"])

candidates = {run_id: payload for run_id, payload in {**phase6_payloads, "A1": a1_payload}.items()
              if payload is not None}
final_rows = [{"id": run_id, "epoch retenue": payload["best_epoch"],
               "epochs exécutées": payload["config"]["epochs_ran"],
               "val_acc": payload["best_metrics"]["val_accuracy"], "val_loss": payload["best_metrics"]["val_loss"]}
              for run_id, payload in candidates.items()]
display(pd.DataFrame(final_rows))
print("Sélection :", SELECTION_CRITERION)
SELECTED_ID = None
if a1_payload is None:
    print("A1 absent : run Colab requis ; aucun choix définitif.")
else:
    SELECTED_ID = min(candidates, key=lambda run_id: (
        candidates[run_id]["best_metrics"]["val_loss"], -candidates[run_id]["best_metrics"]["val_accuracy"], run_id))
    print(f"Modèle retenu par le critère : {SELECTED_ID}")

# %%
if a1_payload is not None and all((CHECKPOINT_DIR / f"{run_id}.keras").is_file() for run_id in ("E3", "A1")):
    for run_id in ("E3", "A1"):
        validation_results[run_id] = evaluate(tf.keras.models.load_model(CHECKPOINT_DIR / f"{run_id}.keras"),
                                              X_val, y_val)
    f1_a1 = pd.DataFrame({run_id: class_report(y_val, validation_results[run_id]["y_pred"])["f1-score"]
                          for run_id in ("E3", "A1")}).loc[[*CLASS_NAMES, "macro avg", "weighted avg"]]
    f1_a1["écart A1 − E3"] = f1_a1["A1"] - f1_a1["E3"]
    print("F1 par classe sur la validation :")
    print(f1_a1.round(3).to_string())

# %% [markdown]
# **Résultats A1 du 7 octobre 2026.** A1 a été entraîné sur Colab GPU
# (TensorFlow 2.21.0 / Keras 3.13.2) avec le split, l'architecture et le
# protocole d'E3. Son checkpoint, rechargé en local, retrouve exactement
# l'accuracy validation de l'historique et sa loss à 3e-7 près. Les lignes du CSV
# des autres runs sont inchangées.
#
# > *Nos observations :*
# > - **Meilleur checkpoint.** A1 retient l'epoch 24 sur 29 exécutées, avec une
# >   val_loss de 1,126252 et une val_accuracy de 0,577200. Par rapport à E3
# >   (epoch 7), la loss baisse de 0,066188 et l'accuracy gagne 1,79 point.
# > - **Surapprentissage retardé.** La loss train d'A1 reste au-dessus de la loss
# >   validation jusqu'à l'epoch 22. À sa dernière epoch, la val_loss d'A1 vaut
# >   1,156619, contre 1,349853 pour E3. L'accuracy train finale d'A1 (0,571838)
# >   reste proche de sa validation (0,566520), alors qu'E3 finissait à 0,770429
# >   contre 0,573717. L'accuracy train d'A1 est mesurée sur des images augmentées,
# >   ce qui limite cette comparaison.
# > - **Budget.** L'arrêt anticipé intervient à l'epoch 29, cinq epochs après le
# >   minimum, juste sous le plafond de 30. La loss train baisse encore
# >   (1,129738) : un budget plus long pourrait modifier le résultat, ce qui n'est
# >   pas testé.
# > - **Par classe.** A1 progresse sur angry (+0,039 de F1), happy (+0,035) et
# >   sad (+0,020), mais recule sur disgust (0,253 → 0,182) et fear (−0,052).
# >   Le F1 macro baisse légèrement (0,507 → 0,502) alors que le F1 pondéré
# >   augmente (0,561 → 0,569). Le gain global vient donc surtout des classes
# >   fréquentes.
#
# > *Nos hypothèses :* notre hypothèse initiale est confirmée sur ce run : val_loss
# > plus basse, meilleure epoch plus tardive et surapprentissage plus lent. Le recul
# > de disgust et fear pourrait venir d'indices fins (nez plissé, yeux écarquillés)
# > atténués par le recadrage et le flou d'interpolation. Ce n'est pas vérifié, et
# > le F1 de disgust repose sur 65 images seulement.
#
# > *Notre choix :* selon le critère fixé avant les runs (val_loss minimale du
# > checkpoint), le modèle final est **A1**. Ce choix favorise la qualité globale
# > des probabilités et l'accuracy, mais pas le F1 macro, où E3 reste légèrement
# > devant. Six modèles ont été comparés sur la même validation avec une seule
# > seed, donc la meilleure val_loss peut être un peu optimiste. Le test officiel
# > fournit une estimation indépendante.

# %% [markdown]
# ## Approfondissement CNN - démarche de recherche
#
# **Constats de départ (sections précédentes).**
# - C0 et E3 surapprennent dès l'epoch 7 : leur val_loss remonte ensuite jusqu'à
#   1,35 environ.
# - A1 retarde ce surapprentissage, mais s'arrête à l'epoch 29 sur 30 avec une loss
#   train encore en baisse. Son accuracy train (0,57) reste proche de sa
#   validation : sur des images augmentées, le réseau semble manquer de capacité ou
#   de temps d'entraînement plutôt que mémoriser.
# - disgust (436 images en train officiel) et fear gardent les F1 les plus bas ;
#   sad, neutral et fear se confondent entre eux.
#
# **Repères publiés.** La précision humaine sur FER2013 est estimée à 65 ± 5 %
# ([Goodfellow et al., 2013](https://arxiv.org/abs/1307.0414)). Un réseau unique
# de type VGG, plus profond, longuement entraîné et finement réglé, atteint
# 73,28 % sur le test ([Khaireddin et Chen, 2021](https://arxiv.org/abs/2105.03588)).
# A1 obtient 57,55 % sur le test. Ces budgets ne sont pas comparables au nôtre,
# mais ils indiquent des pistes : plus de profondeur, une normalisation des
# activations et un entraînement plus long.
#
# **Objectifs fixés avant les runs.**
# 1. Baisser la val_loss **moyenne sur 3 seeds** (42, 43 et 44) par rapport à A1
#    refait en local. Chaque seed change l'initialisation, l'ordre des batches,
#    l'augmentation et le dropout ; le split reste celui de C0 (empreinte vérifiée).
# 2. Mesurer la variabilité entre seeds (écart-type) avant d'interpréter un écart.
# 3. Suivre le F1 macro pour voir l'effet sur disgust et fear.
#
# Critère de sélection : val_loss moyenne minimale, puis val_accuracy moyenne
# maximale. Le test n'est jamais utilisé pendant cette recherche. La largeur
# reste à 32/64/128 filtres, car un réseau 64/128/256 coûte environ 92 s par epoch
# sur notre CPU, contre 30 s à 32/64/128 avec deux convolutions par bloc.
#
# **Échelle d'améliorations.** Un changement à la fois, gardé seulement s'il
# baisse la val_loss moyenne :
#
# | Id | Changement | Hypothèse |
# |---|---|---|
# | R0 | A1 refait en local | Référence sur le même matériel et variabilité des seeds |
# | R1 | BatchNorm après chaque convolution | Activations normalisées, entraînement plus rapide et plus stable ([Ioffe et Szegedy, 2015](https://arxiv.org/abs/1502.03167)) |
# | R2 | Deux convolutions par bloc | Plus de profondeur et un champ récepteur plus large à résolution égale ([Simonyan et Zisserman, 2014](https://arxiv.org/abs/1409.1556)) |
# | R3 | Dropout 0,25 par bloc et 0,5 avant la sortie | Compenser la capacité ajoutée |
# | R4 | 60 epochs, ReduceLROnPlateau, patience 8 | A1 a atteint le plafond de 30 epochs |
# | R5 | Poids de classes équilibrés | Meilleur rappel de disgust et fear, peut-être au prix de la loss |
#
# **Recherche aléatoire.** Six réglages tirés autour du meilleur palier
# (learning rate log-uniforme entre 2e-4 et 2e-3, batch, dropouts, taille de la
# Dense), une seed chacun ; les deux meilleurs sont confirmés sur 3 seeds. Le
# gagnant (seed 42) devient le candidat au modèle final, avec une réévaluation du
# test **déclarée** dans un fichier séparé.
#
# **Exécution.** `python -m src.research --ladder --search 6` reprend les runs
# déjà faits (un JSON par run et par seed dans `training/research/`). Dans le
# notebook, `RUN_RESEARCH=False` relit ces résultats sans entraîner. Sur Colab,
# le même code utilise le GPU s'il est disponible.

# %%
from src.research import SELECTION, run_ladder, run_search, summarize

RUN_RESEARCH = False  # Plusieurs heures sur CPU : lancer plutôt la commande ci-dessus.
RESEARCH_DIR = PROJECT_ROOT / "training/research"
if RUN_RESEARCH:
    research_data = (X_train, y_train, X_val, y_val)
    run_ladder(research_data, out_dir=RESEARCH_DIR, checkpoint_dir=CHECKPOINT_DIR / "research")
    run_search(research_data, 6, out_dir=RESEARCH_DIR, checkpoint_dir=CHECKPOINT_DIR / "research")

research_summary = summarize(RESEARCH_DIR)
if research_summary.empty:
    print("Aucun run d'approfondissement disponible.")
else:
    print("Sélection :", SELECTION)
    display(research_summary.round(4))
    ladder_path = RESEARCH_DIR / "ladder.json"
    if ladder_path.is_file():
        ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
        display(pd.DataFrame(ladder["decisions"]).round(4))

# %%
DEEP_DIVE_WINNER = None
if not research_summary.empty:
    confirmed = research_summary[research_summary["seeds"] == 3]  # Déjà trié par le critère.
    DEEP_DIVE_WINNER = confirmed.iloc[0]["id"]
    print(f"Configuration retenue par le critère : {DEEP_DIVE_WINNER} ; checkpoint de la seed 42.")
    research_records = {
        path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESEARCH_DIR.glob("*_s*.json"))
    }
    for run_id in ("R0", DEEP_DIVE_WINNER):
        for seed in (42, 43, 44):
            record = research_records.get(f"{run_id}_s{seed}")
            if record is not None and seed == 42:
                plot_history(f"{run_id} seed {seed}", record["history"], record["best_epoch"])
    f1_by_run = {
        run_id: pd.DataFrame([research_records[f"{run_id}_s{seed}"]["val_f1"] for seed in (42, 43, 44)]).mean()
        for run_id in ("R0", "R2", "R4", DEEP_DIVE_WINNER)
    }
    f1_research = pd.DataFrame(f1_by_run)
    f1_research.loc["macro"] = f1_research.mean()
    print("F1 de validation par classe, moyenne sur 3 seeds :")
    print(f1_research.round(3).to_string())

# %% [markdown]
# **Résultats du 8 octobre 2026.** 28 runs (12 configurations), soit 6,6 heures sur
# le CPU d'un Mac arm64 (TensorFlow 2.21.0, sans GPU). Le test n'a jamais été
# utilisé.
#
# > *Nos observations :*
# > - **Référence R0.** A1 refait en local obtient une val_loss de 1,1319 ± 0,0097
# >   et une val_accuracy de 0,5733, proches d'A1 sur Colab (1,126252 et
# >   0,577200). L'écart entre seeds est d'environ 0,01 de loss. Un gain plus
# >   petit ne serait donc pas interprétable.
# > - **R1, BatchNorm : rejeté.** La val_loss moyenne monte à 1,2218. Elle varie
# >   fortement d'une epoch à l'autre (seed 42 : 1,244, 1,298, 1,313, 1,288, 1,518,
# >   1,237), et l'arrêt anticipé intervient dès les epochs 12 à 18.
# > - **R2, deux convolutions par bloc : gardé.** La val_loss baisse à
# >   1,0523 ± 0,0076 (−0,080 par rapport à R0), la val_accuracy passe à 0,6095,
# >   et le F1 de disgust monte de 0,169 à 0,306.
# > - **R3, Dropout : rejeté.** 1,0825 ± 0,0104. Les trois seeds retiennent
# >   l'epoch 29 sur 30 : le budget de 30 epochs arrête l'entraînement avant la
# >   convergence.
# > - **R4, 60 epochs et ReduceLROnPlateau : gardé.** 1,0299 ± 0,0035, avec une
# >   val_accuracy de 0,6222 et le meilleur F1 macro (0,569). Les meilleures epochs
# >   vont de 30 à 39, après plusieurs réductions du learning rate.
# > - **R5, poids de classes : rejeté.** 1,1266, et le F1 macro baisse aussi
# >   (0,537 contre 0,569 pour R4) : la pondération n'améliore pas ici les classes
# >   rares.
# > - **Recherche aléatoire.** S5 (learning rate 6,5e-4, batch 32, dropout
# >   0,25 par bloc et 0,3 avant la sortie, Dense 256, 1 468 135 paramètres)
# >   obtient la meilleure val_loss moyenne : 1,0232 ± 0,0121, avec une
# >   val_accuracy de 0,6284 ± 0,0054. S1 suit (1,0395). S4 (learning rate 1,7e-3,
# >   batch 32) échoue : sa val_accuracy reste à 0,251 dès la première epoch, soit
# >   la part de happy en validation (1 082 / 4 307). Le réseau prédit donc
# >   toujours happy.
# > - **Gain total.** De R0 à S5, la val_accuracy moyenne gagne 5,5 points
# >   (0,5733 → 0,6284) et la val_loss baisse de 0,109. Le F1 progresse sur toutes
# >   les classes, par exemple disgust (0,169 → 0,271) et surprise
# >   (0,650 → 0,717).
#
# > *Nos hypothèses :*
# > - Avec BatchNorm, les statistiques des batches d'entraînement augmentés
# >   pourraient différer de celles des images de validation non augmentées,
# >   d'où une validation instable. Ce n'est pas vérifié (il faudrait un run
# >   BatchNorm sans augmentation).
# > - Le Dropout de R3 semble demander plus d'epochs : S5 combine Dropout et
# >   budget de 60 epochs et obtient le meilleur résultat, ce qui va dans ce sens
# >   sans le démontrer.
# > - L'échec de S4 viendrait d'un learning rate trop élevé pour des batches de
# >   32 images. Le réseau reste bloqué sur la classe majoritaire.
#
# > *Notre choix :* selon le critère fixé avant les runs, le modèle retenu est
# > **S5** (checkpoint de la seed 42 : val_loss 1,009580, val_accuracy 0,632691).
# > Son avance sur R4 (0,0067 de loss) est plus petite que son propre écart-type
# > entre seeds (0,0121) : les deux configurations sont presque à égalité. R4 a un
# > meilleur F1 macro (0,569 contre 0,562), notamment sur disgust, et compte
# > 877 287 paramètres seulement. S5 retient encore l'epoch 58 sur 60 pour la seed
# > 42, donc le budget reste une limite. Douze configurations ont été comparées sur
# > la même validation, ce qui rend la meilleure val_loss un peu optimiste : la
# > réévaluation du test, déclarée ci-dessous, donne une estimation indépendante.

# %% [markdown]
# ### Test officiel - évaluations du modèle final
#
# Chaque évaluation du test est unique : `evaluate_test_once` écrit un fichier
# par modèle et refuse de le réécrire. Les exécutions suivantes relisent ces
# fichiers et affichent la même analyse sans recalculer. L'empreinte du test
# enregistrée garantit que les images relues sont les mêmes, dans le même ordre.
#
# **Deux usages du test, déclarés.**
# 1. Le 7 octobre, A1, choisi en phase 7 : `training/logs/test_evaluation.json`.
# 2. Le 8 octobre, S5, choisi sur la validation par l'approfondissement :
#    `training/logs/test_evaluation_deepdive.json`.
#
# Le premier résultat a été vu avant l'approfondissement. Ses constats
# (confusions sad/neutral/fear, disgust → angry) n'ont pas servi à choisir les
# configurations, qui viennent de la validation et de la littérature. Le second
# score reste toutefois moins strictement indépendant qu'une première évaluation.

# %%
from src.evaluate import evaluate_test_once, load_test_evaluation
from src.train import _validation_digest


RUN_TEST = False  # Évaluations faites : A1 le 7 octobre, S5 le 8 octobre. Laisser à False.
FINAL_MODEL_ID = "S5"  # Approfondissement CNN, critère fixé avant les runs.
TEST_RESULTS = {"A1": "test_evaluation.json", "S5": "test_evaluation_deepdive.json"}  # Ordre chronologique.
assert SELECTED_ID in (None, "A1") and DEEP_DIVE_WINNER in (None, FINAL_MODEL_ID)
if RUN_TEST and load_test_evaluation(LOG_DIR, TEST_RESULTS[FINAL_MODEL_ID]) is None:
    evaluate_test_once(FINAL_MODEL_ID, X_test, y_test, log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR,
                       result_name=TEST_RESULTS[FINAL_MODEL_ID])

test_payloads = {}
for run_id, result_name in TEST_RESULTS.items():
    payload = load_test_evaluation(LOG_DIR, result_name)
    if payload is None:
        print(f"{run_id} : test non évalué.")
        continue
    assert payload["run_id"] == run_id and payload["test_sha256"] == _validation_digest(X_test, y_test)
    test_payloads[run_id] = payload
    print(f"Test officiel ({run_id}) : accuracy={payload['accuracy']:.6f}, loss={payload['loss']:.6f}")
    show_analysis(f"{run_id} test", X_test, y_test, payload["y_pred"], payload["confidence"])

# %% [markdown]
# **Résultat du test officiel (évaluation unique du 7 octobre 2026).** A1 a été
# évalué une seule fois sur les 7 178 images du test, après le choix fait sur la
# validation. `test_evaluation.json` conserve ce résultat et les empreintes du
# checkpoint et du test. Une seconde évaluation a bien été refusée.
#
# > *Nos observations :*
# > - **Score global.** L'accuracy test vaut 0,575508 et la loss 1,131685, contre
# >   0,577200 et 1,126252 en validation, soit un écart de 0,17 point d'accuracy
# >   et de 0,005433 de loss. La validation n'a donc pas sensiblement surestimé
# >   A1, malgré la sélection parmi six modèles.
# > - **Classes bien reconnues.** happy reste la mieux reconnue (F1 0,802, recall
# >   0,848), suivie de surprise (0,694) et neutral (0,528).
# > - **Classes difficiles.** disgust a le F1 le plus bas (0,252, recall 0,153).
# >   Le modèle ne prédit disgust que 24 fois pour 111 images, et 43 % des vraies
# >   disgust sont prédites angry. fear suit (F1 0,324, recall 0,263) : ses images
# >   partent vers sad (27 %), angry (15 %) et surprise (13 %).
# > - **Confusions principales.** fear → sad (273 images), sad → neutral (250),
# >   neutral → sad (246), angry → sad (172) et fear → angry (151). On retrouve
# >   le groupe sad/neutral/fear observé en validation, avec angry en plus.
# > - **Équilibre par classe.** Le F1 macro (0,508) reste inférieur au F1 pondéré
# >   (0,565), car les classes fréquentes sont les mieux reconnues. Les F1 par
# >   classe sont proches de ceux d'A1 en validation.
# > - **Erreurs confiantes.** Sept des huit erreurs les plus confiantes
# >   (p ≥ 0,98) sont prédites happy, sur des visages souriants annotés surprise,
# >   neutral, sad ou fear. L'image 5, annotée fear, est prédite surprise avec une
# >   bouche grande ouverte.
#
# > *Nos hypothèses :* comme en validation, plusieurs erreurs confiantes
# > paraissent ambiguës ou mal annotées, par exemple un large sourire annoté sad.
# > Une partie de ces erreurs relèverait donc du bruit d'annotation de FER2013
# > plutôt que du modèle. La confusion disgust → angry pourrait venir d'indices
# > communs (sourcils froncés, bouche crispée) et du faible nombre d'exemples
# > disgust. Ces causes ne sont pas vérifiées séparément.
#
# > *Limites :* ce score décrit un seul run (seed 42) et un seul modèle évalué sur
# > le test. FER2013 contient des images de faible résolution, des annotations
# > ambiguës et des biais de population. Une expression prédite n'est pas une
# > émotion certaine. Les modèles suivants ont été choisis sur la validation
# > seulement ; la seconde évaluation du test (S5), déclarée plus haut, est
# > analysée ci-dessous.

# %% [markdown]
# **Seconde évaluation du test, déclarée (8 octobre 2026) : S5.** Le modèle a été
# choisi sur la validation avant cette évaluation, réalisée une seule fois dans
# `test_evaluation_deepdive.json` (une seconde tentative a été refusée).
#
# > *Nos observations :*
# > - **Score global.** S5 obtient une accuracy test de 0,624408 et une loss de
# >   1,004681, contre 0,575508 et 1,131685 pour A1, soit +4,89 points. Son
# >   checkpoint donnait 0,632691 et 1,009580 en validation : l'écart avec le test
# >   reste faible (−0,83 point d'accuracy).
# > - **Par classe.** Le F1 test progresse pour les sept expressions par rapport à
# >   A1 : angry 0,498 → 0,545, disgust 0,252 → 0,308, fear 0,324 → 0,406,
# >   happy 0,802 → 0,852, neutral 0,528 → 0,588, sad 0,456 → 0,487, surprise
# >   0,694 → 0,749. Le F1 macro passe de 0,508 à 0,562.
# > - **Confusions restantes.** sad → neutral (292 images, 23,4 % des sad),
# >   fear → sad (213), neutral → sad (169), fear → angry (164) et sad → fear
# >   (144). disgust reste la classe la plus difficile : 45 % de ses images sont
# >   prédites angry.
# > - **Erreurs confiantes.** Les huit erreurs les plus confiantes ont p ≥ 0,99.
# >   Cinq sont des visages souriants annotés neutral, sad ou surprise et prédits
# >   happy (images 1, 2, 3, 5, 6). Trois sont des bouches grandes ouvertes
# >   annotées fear ou sad et prédites surprise (images 4, 7, 8). L'image 7 porte
# >   un filigrane de banque d'images.
#
# > *Nos hypothèses :* le gain vient d'une capacité plus grande (deux convolutions
# > par bloc, Dense 256) rendue utilisable par l'augmentation, le Dropout et un
# > entraînement plus long avec réduction du learning rate. Les erreurs confiantes
# > restantes ressemblent à celles d'A1 : ambiguïtés entre sourire et surprise,
# > bouche ouverte entre peur et surprise, et bruit d'annotation de FER2013.
#
# > *Limites :* trois seeds seulement, une validation utilisée pour comparer douze
# > configurations, et un test consulté deux fois, ce qui est déclaré ci-dessus.
# > Le budget de 60 epochs reste atteint (epoch 58 pour la seed 42). Les réseaux
# > publiés autour de 73 % sont plus larges et entraînés bien plus longtemps.

# %% [markdown]
# ## Phase 8 - pipeline final : détection multi-visages et expressions
#
# Cette section assemble le système complet visé par le sujet (Figure 4 du
# sujet) : image → détection des visages → boîtes → extraction de chaque visage →
# `preprocess_face` → modèle final (`FINAL_MODEL_ID`) → softmax → expression et
# probabilité. Les phases 1 à 7 ont construit et choisi le classifieur ; ici, il
# est appliqué à des photos entières qui contiennent plusieurs personnes. Cette
# section sert aussi de **démonstration du modèle final** (livrable 3).
#
# **Choix du détecteur.** Les [classes COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)
# incluent person mais pas face : une boîte de personne ne fournit pas un crop facial.
# Nous retenons [YuNet, publié dans OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
# un détecteur spécialisé pré-entraîné sur des visages, exécuté par FaceDetectorYN.
# Il évite une dépendance YOLO/PyTorch supplémentaire. Ses poids ONNX
# face_detection_yunet_2023mar.onnx (232 589 octets) sont sous
# [licence MIT, Shiqi Yu](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE).
# Le téléchargement officiel et l'empreinte SHA256 attendue sont dans src/detect.py ;
# la licence est conservée à côté des poids, tous deux ignorés par Git.
# Le backend OpenCV est explicite pour accepter les tailles variables avec ce modèle
# à dimensions ONNX fixes, notamment sous OpenCV 5. Aucune nouvelle dépendance.
#
# **Notions à l'oral.**
# - Classification : une classe pour un crop déjà extrait. Détection : localisation
#   et score pour chaque visage présent dans une image entière.
# - Bounding box : rectangle x, y, largeur, hauteur chez YuNet ; notre API renvoie
#   [x1, y1, x2, y2], borné à l'image originale, avec x2/y2 exclusifs pour le crop.
# - Confidence score : score du détecteur pour retenir une proposition de visage.
#   Il est séparé de p, la probabilité softmax du modèle final de l'expression sélectionnée.
#   Aucun des deux scores n'est une certitude sur l'émotion ressentie.
# - IoU : aire d'intersection de deux boîtes divisée par leur aire d'union.
#   NMS : conserve les propositions les mieux scorées et supprime des doublons
#   trop superposés. FaceDetectorYN réalise ce filtrage en interne : seuil de
#   score 0,9, seuil NMS 0,3, au plus 5 000 candidats avant NMS.
# - [YOLO (Redmon et al.)](https://arxiv.org/abs/1506.02640) prédit des boîtes et des
#   classes en un passage du réseau sur l'image. C'est un principe général ;
#   YuNet n'est pas un YOLO. Ici la détection faciale fournit aussi cinq points
#   faciaux, que nous n'utilisons pas pour réaligner les crops.
# - Pré-entraînement : apprendre les poids sur des données antérieures.
#   Fine-tuning : adapter ensuite ces poids à une autre tâche ou d'autres données.
#   Ici YuNet reste figé, et le modèle final est uniquement rechargé, sans fine-tuning.
# - Précision = TP/(TP+FP), rappel = TP/(TP+FN), avec appariement des boîtes
#   prédites aux boîtes annotées à un seuil IoU donné. AP résume la courbe
#   précision/rappel ; mAP moyenne l'AP sur les classes (et parfois plusieurs IoU).
#   Nous n'avons aucune boîte de référence pour ces photos : aucune mesure
#   de précision, rappel ou mAP du détecteur n'est calculée.
#
# **Pipeline réel.** Image BGR uint8 → YuNet (grand côté limité à 1280) → boîtes
# remises à l'échelle originale et bornées → crops BGR convertis en RGB →
# predict_faces → unique preprocess_face (48×48×1, float32, /255) → batch du modèle final →
# résultats structurés → annotation séparée. Zéro visage renvoie une liste vide ;
# les boîtes invalides et les crops vides sont ignorés.
#
# **Images indépendantes de FER2013.** Crédit NASA, domaine public aux États-Unis :
# [Apollo 11](https://commons.wikimedia.org/wiki/File:Apollo_11_Crew.jpg),
# [Apollo 12](https://science.nasa.gov/resource/apollo-12-crew/),
# [Apollo 13](https://commons.wikimedia.org/wiki/File:Apollo_13_Prime_Crew.jpg).
# Usage pédagogique selon les [règles NASA](https://www.nasa.gov/nasa-brand-center/images-and-media/),
# sans soutien implicite de la NASA. Les URL de téléchargement sont dans DEMO_IMAGES.
# Images et annotations restent dans data/demo/, ignoré par Git.
#
# **Reproduction locale ou Colab.** `scripts/colab_bundle.sh` inclut `src/`, les
# checkpoints, YuNet et les photos de démo s'ils sont présents ; les dépendances du
# projet suffisent. Cette section peut être exécutée seule, sans charger FER2013,
# après définition de `PROJECT_ROOT` et `FINAL_MODEL_ID`. Mettre
# DOWNLOAD_FACE_DEMO=True pour télécharger seulement YuNet/licence/photos absents,
# ou utiliser `python -m src.detect --download-demo` depuis la racine. Garder tous
# les flags d'entraînement et `RUN_TEST=False`.

# %%
import cv2
from urllib.error import URLError
from src.detect import (
    DEMO_IMAGES, annotate_faces, detect_expressions, download_demo_assets, load_models,
)

DOWNLOAD_FACE_DEMO = False
FACE_DEMO_DIR = PROJECT_ROOT / "data/demo"
YUNET_PATH = PROJECT_ROOT / "training/checkpoints/face_detection_yunet_2023mar.onnx"
FINAL_MODEL_PATH = PROJECT_ROOT / f"training/checkpoints/{FINAL_MODEL_ID}.keras"
face_demo_models = None
if DOWNLOAD_FACE_DEMO:
    try:
        download_demo_assets(YUNET_PATH, FACE_DEMO_DIR)
    except (OSError, URLError) as error:
        print(f"Téléchargement indisponible : {error}. Transférer les fichiers manuellement.")
if YUNET_PATH.is_file() and FINAL_MODEL_PATH.is_file():
    face_demo_models = load_models(YUNET_PATH, FINAL_MODEL_PATH)  # Une fois avant la boucle.
    print(f"Pipeline final : YuNet + {FINAL_MODEL_ID} ({FINAL_MODEL_PATH.name}).")
else:
    print(f"Démo indisponible : transférer {FINAL_MODEL_PATH.name} et télécharger YuNet "
          "(DOWNLOAD_FACE_DEMO=True). Aucun entraînement ni test officiel lancé.")


def show_face_pipeline(frame_bgr, title, output_path=None):
    """Détecte les visages, classe leurs expressions et affiche l'image annotée."""
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3
    face_detector, expression_model = face_demo_models
    face_results = detect_expressions(frame_bgr, face_detector, expression_model)
    annotated_bgr = annotate_faces(frame_bgr, face_results)
    assert annotated_bgr.shape == frame_bgr.shape
    if output_path is not None:
        assert cv2.imwrite(str(output_path), annotated_bgr)
    print(f"{title} : {len(face_results)} visage(s). "
          "p = softmax de l'expression ; face score = score YuNet.")
    if face_results:
        display(pd.DataFrame(face_results))
    else:
        print("Aucun visage retenu au seuil actuel.")
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))  # Matplotlib attend RGB.
    plt.title(f"{title} - expression prédite, sans certitude émotionnelle")
    plt.axis("off")
    plt.show()
    return face_results

# %%
if face_demo_models is not None:
    face_output_dir = FACE_DEMO_DIR / "annotated"
    face_output_dir.mkdir(parents=True, exist_ok=True)
    for image_name in DEMO_IMAGES:
        image_path = FACE_DEMO_DIR / image_name
        frame_bgr = cv2.imread(str(image_path)) if image_path.is_file() else None
        if frame_bgr is None:
            print(f"Image absente ou illisible : {image_path}. DOWNLOAD_FACE_DEMO=True.")
            continue
        show_face_pipeline(frame_bgr, image_name, face_output_dir / f"{image_path.stem}_annotated.png")

# %% [markdown]
# **Observations locales.** Pipeline exécuté avec OpenCV 5.0.0 et TensorFlow
# 2.21.0 ; trois visages retenus sur chacune des trois photos, avec des scores
# YuNet d'environ 0,93 à 0,95. Les boîtes inspectées entourent les visages visibles.
# Avec S5 (modèle final, 8 octobre) : Apollo 11 donne deux happy (p≈1,00/0,94) et
# un neutral (p≈0,83) ; Apollo 12 trois happy (p≈0,99–1,00) ; Apollo 13 trois happy
# (p≈0,74–1,00). A1 (7 octobre) prédisait les mêmes expressions, avec par exemple
# p≈0,70 pour le neutral d'Apollo 11.
# Les sourires visibles rendent ces sorties plausibles, sans labels d'expression
# de référence. Les textes sont adaptés à la résolution pour rester lisibles.
# Ces portraits posés, essentiellement frontaux, de trois hommes adultes chacun
# ne vérifient ni les petits visages, ni les occlusions, ni la diversité de population.
# Aucun score de qualité du détecteur ne peut en être déduit.
# Les crops n'ont pas l'alignement de FER2013 ; pose, lumière et changement de domaine
# peuvent fausser le classifieur, même quand son softmax est élevé. Pas de calibration des scores.
# Cette extension a été validée localement ; aucune relance Colab n'a été nécessaire.
#
# **Relais vidéo.** Charger detector, classifier = load_models() avant la boucle ;
# appeler detect_expressions(frame_bgr, detector, classifier), puis
# annotate_faces(frame_bgr, results). Les dicts contiennent box_xyxy, detector_score,
# class_id, expression, expression_probability, dans l'ordre des détections.
# Aucun identifiant de suivi temporel : le numéro dessiné dépend de chaque frame.
# Réutiliser ces fonctions sans recharger les modèles et sans réentraîner le classifieur.

# %% [markdown]
# **Démonstration libre.** Pour la soutenance, `RUN_CUSTOM_IMAGE=True` applique le
# même pipeline à une photo choisie : sur Colab, une fenêtre d'upload s'ouvre ; sur
# Mac, renseigner `CUSTOM_IMAGE_PATH`. Le flag reste à False pour qu'« Exécuter
# tout » ne s'arrête pas sur une demande de fichier.

# %%
RUN_CUSTOM_IMAGE = False
CUSTOM_IMAGE_PATH = None  # Mac : par exemple PROJECT_ROOT / "data/demo/ma_photo.jpg".
if RUN_CUSTOM_IMAGE and face_demo_models is not None:
    if IN_COLAB:
        from google.colab import files

        custom_images = {name: cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                         for name, content in files.upload().items()}
    else:
        custom_images = {Path(CUSTOM_IMAGE_PATH).name: cv2.imread(str(CUSTOM_IMAGE_PATH))}
    for name, frame_bgr in custom_images.items():
        if frame_bgr is None:
            print(f"{name} : image illisible.")
            continue
        show_face_pipeline(frame_bgr, name)
