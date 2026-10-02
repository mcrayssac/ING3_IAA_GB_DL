# TODO - Projet DL expressions faciales

Statuts : `[ ]` à faire, `[~]` en cours, `[x]` fini, `[!]` bloqué. Une phase par session. Détails techniques : voir `SPEC.md`.
Responsable : A / B (à compléter). Soutenance : **ven. 2026-10-09**.

Règle Git : créer/basculer sur une branche locale par phase avant de coder (`phase-N-nom-court`). Ne faire aucun commit sans demande explicite.

## Planning
| Date | Objectif |
|---|---|
| 10-01 / 10-02 | Phases 1 à 3 |
| 10-03 / 10-04 | Phases 4 à 6 |
| 10-05 | Phase 7 (3 expériences) |
| 10-06 | Socle terminé + explications markdown complètes |
| 10-07 | Phase 8 (optionnelle) |
| 10-08 | Phase 9 : slides, répétition, run complet du notebook sur Colab |

## Phase 1 - Données (3 pts) - resp. :
- [x] `src/data.py` : `CLASS_NAMES`, chargement, split train/val/test, `preprocess_face()`. Fait quand : `X_train.shape == (N,48,48,1)`, valeurs dans [0,1], test intact.
- [x] Notebook : source/licence, nb d'images, classes, répartition par classe (graphique), dimensions, format, déséquilibre, 5 exemples par classe.
- [x] Explications markdown : rôle du redimensionnement, de la normalisation, de l'encodage, du split.
- [x] `tests/test_shapes.py` (shapes et K).

## Phase 2 - Baseline MLP - resp. :
- [ ] `build_mlp()` + entraînement court, val_acc/val_loss enregistrées dans `experiments.csv` (id `B0`).
- [ ] Explications markdown : image -> vecteur, poids/biais, propagation avant, activations, loss, rétropropagation, descente de gradient, softmax.

## Phase 3 - CNN - resp. :
- [ ] `build_cnn()` paramétrable. Fait quand : tableau couche par couche (shape de sortie + nb de paramètres) affiché une fois.
- [ ] Explications markdown : filtres, convolution, feature maps, kernel, stride, padding, ReLU, pooling, Flatten, Dense, sortie ; justification de l'architecture.

## Phase 4 - Entraînement - resp. :
- [ ] `src/train.py` : compile, fit, callbacks (`ModelCheckpoint`, `EarlyStopping`), historique sauvegardé dans `training/logs/`.
- [ ] Run Colab GPU du CNN de départ (id `C0`), courbes loss et accuracy train/val.
- [ ] Explications markdown : choix loss/optimiseur/batch/epochs/métriques ; lecture des courbes, sur/sous-apprentissage.

## Phase 5 - Évaluation - resp. :
- [ ] `src/evaluate.py` : charger le meilleur modèle, évaluer **une seule fois** sur test.
- [ ] Matrice de confusion, precision/recall/F1 par classe.
- [ ] Exemples corrects et incorrects (image, vraie classe, prédiction, probabilité).
- [ ] Analyse d'erreurs rédigée (classes confondues, causes, limites).

## Phase 6 - Expériences (4 pts) - resp. :
- [ ] E1 architecture, E2 régularisation/augmentation, E3 lr/optimiseur/batch (un axe à la fois).
- [ ] `experiments.csv` complet + tableau comparatif dans le notebook (modification, val, observation).
- [ ] Choix du modèle final justifié sur la validation, puis évaluation test finale.

## Phase 7 - Enrichissement léger (si socle fini)
- [ ] Data augmentation avancée OU transfer learning (un seul), comparé au modèle final.

## Phase 8 - Détection de visages (optionnelle)
- [ ] Choisir et justifier le détecteur (YOLO visages ou alternative).
- [ ] Pipeline : détection -> extraction -> `preprocess_face()` -> CNN -> bounding box + expression + score, sur 2-3 images à plusieurs personnes.
- [ ] Bonus : courte vidéo.

## Phase 9 - Livrables - resp. :
- [ ] Notebook Colab : "Exécuter tout" sans erreur sur runtime neuf ; cellule de démo finale.
- [ ] Présentation : démarche, architectures, expériences, résultats, limites.
- [ ] Répétition : chaque membre sait expliquer le code, le modèle, les choix, les erreurs.
