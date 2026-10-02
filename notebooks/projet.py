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

# %% [markdown]
# Les images FER2013 sont déjà en 48 x 48 gris. Le redimensionnement conserve
# néanmoins cette taille d'entrée fixe si une image diffère. La normalisation en
# `float32` dans [0, 1] stabilise l'entraînement. Les émotions sont encodées par
# des entiers dans l'ordre de `CLASS_NAMES`, compatible avec la loss sparse. Le
# dossier `test/` officiel est conservé intact : seuls les exemples de `train/`
# sont séparés de façon stratifiée (15 %, seed 42) pour former la validation.

# %%
all_labels = np.concatenate((y_train, y_val))
counts = np.bincount(all_labels, minlength=K)
plt.figure(figsize=(9, 4))
plt.bar(CLASS_NAMES, counts)
plt.title("Répartition par classe dans train + validation")
plt.ylabel("Nombre d'images")
plt.xticks(rotation=30)
plt.show()
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
# expériences. Les courbes et valeurs ci-dessous ne s'affichent qu'après le run.

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
    print("B0 non exécuté dans cette session ; aucun résultat sauvegardé n'est chargé. Guide : docs/B0_COLAB.md.")

# %%
if b0_history is not None:
    epoch_numbers = range(1, len(b0_history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Perte", "Accuracy")):
        axis.plot(epoch_numbers, b0_history[metric], label="Train")
        axis.plot(epoch_numbers, b0_history[f"val_{metric}"], label="Validation")
        axis.set(title=title, xlabel="Epoch", ylabel=metric, xticks=list(epoch_numbers))
        axis.legend()
    plt.tight_layout()
    plt.show()
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
# dans le guide ; les courbes ci-dessus proviennent uniquement du run de cette session.
# La validation globale du notebook Colab demeure une étape distincte du suivi.
