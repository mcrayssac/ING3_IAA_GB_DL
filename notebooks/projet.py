# %% [markdown]
# # Classification d'expressions faciales
#
# **Données.** Nous utilisons FER2013, diffusé sur
# [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013), à partir du jeu de
# données présenté par Goodfellow et al. (2013). La fiche Kaggle indique la licence
# « Database: Open Database, Contents: Database Contents » ; son usage respecte aussi
# les [conditions d'utilisation Kaggle](https://www.kaggle.com/terms).

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
        axis.set(title=f"{run_id} — {title}", xlabel="Epoch", ylabel=metric, xticks=list(epoch_numbers))
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
# **B0 sur Colab.** Suivre le [guide B0](../docs/B0_COLAB.md)
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
# ## Phase 4 — entraînement C0 et comparaison
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
# [section C0 du guide Colab](../docs/B0_COLAB.md#c0--phase-4-sur-colab-gpu).
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
    print(f"C0 — loss train : {c0_history['loss'][0]:.6f} → {c0_history['loss'][-1]:.6f} ; "
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
# ## Phase 5 — évaluation et analyse des erreurs
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
    axis.set(title=f"{title} — matrice normalisée par vraie classe",
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
        show_examples(f"{title} — prédictions {label} les plus confiantes", X, y, y_pred, confidence, indices)
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
# ## Phase 6 — Trois expériences à partir de C0
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
        phase6_rows.append({"id": run_id, "état": "historique absent — run Colab requis"})
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
# **E1 — capacité réduite.** La Dense 64 réduit les paramètres de 43,22 %. L'écart
# d'accuracy train/validation à l'epoch retenue descend à 5,78 points (11,88 pour
# C0), mais l'accuracy validation perd 1,42 point. La loss ne baisse que de 0,003767.
# Réduire la mémorisation ne suffit donc pas à améliorer la reconnaissance ; ce
# candidat compact n'est pas le meilleur selon notre critère de validation.
#
# **E2 — Dropout.** À paramètres constants, la loss baisse de 0,036417 par rapport
# à C0 ; l'accuracy gagne seulement 0,07 point. Le Dropout semble utile ici pour la
# généralisation des probabilités. Son accuracy train est mesurée avec le masquage
# actif, ce qui limite la comparaison directe des écarts train/validation. Le
# surapprentissage persiste : la loss validation remonte à 1,407054 à l'epoch 13,
# après le minimum de 1,195606 à l'epoch 8. En prédiction, le masquage est désactivé.
#
# **E3 — pas d'optimisation plus petit.** La loss baisse de 0,039583 et l'accuracy
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
# ### Test officiel — évaluation unique du modèle final
#
# Le test n'est évalué qu'une fois, après comparaison avec A1 en phase 7 à partir
# de la validation. `evaluate_test_once` refuse une seconde évaluation, car
# `training/logs/test_evaluation.json` existe alors déjà. Les exécutions
# suivantes relisent ce fichier et affichent la même analyse sans recalculer.
# L'empreinte du test enregistrée garantit que les images relues sont les
# mêmes, dans le même ordre.

# %%
from src.evaluate import evaluate_test_once, load_test_evaluation
from src.train import _validation_digest


# Activer une seule fois, après le choix du modèle final ; ensuite, remettre à False
# et versionner training/logs/test_evaluation.json.
RUN_TEST = False
FINAL_MODEL_ID = None  # Choix définitif seulement après la phase 7 (A1).
test_payload = load_test_evaluation(LOG_DIR)
if RUN_TEST and test_payload is not None:
    print("Test officiel déjà évalué : relecture du résultat, aucune nouvelle évaluation.")
elif RUN_TEST:
    assert FINAL_MODEL_ID is not None, "Choisir le modèle final sur la validation avant le test."
    test_payload = evaluate_test_once(FINAL_MODEL_ID, X_test, y_test, log_dir=LOG_DIR, checkpoint_dir=CHECKPOINT_DIR)

if test_payload is None:
    print("Test officiel non évalué : aucune métrique de test disponible.")
else:
    assert test_payload["test_sha256"] == _validation_digest(X_test, y_test)
    print(f"Test officiel ({test_payload['run_id']}) : accuracy={test_payload['accuracy']:.6f}, "
          f"loss={test_payload['loss']:.6f}")
    show_analysis(f"{test_payload['run_id']} test", X_test, y_test,
                  test_payload["y_pred"], test_payload["confidence"])
