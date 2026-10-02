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

import matplotlib.pyplot as plt
import numpy as np

from src.data import CLASS_NAMES, K, load_dataset


DATA_DIR = Path("data")  # Dossier FER2013 contenant train/ et test/.
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
