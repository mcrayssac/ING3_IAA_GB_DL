# TODO - Projet DL expressions faciales

Statuts : `[ ]` à faire, `[~]` en cours, `[x]` fini, `[!]` bloqué. Une phase par session. Détails techniques : voir `SPEC.md`.
Binôme : Paul PITIOT / Maxime CRAYSSAC. Soutenance : **ven. 2026-10-09**. Les numéros de phase suivent les parties du sujet.

Règle Git : créer/basculer sur une branche locale par phase avant de coder (`phase-N-nom-court`). Ne faire aucun commit sans demande explicite.

## Barème (sur 20)
| Phases | Points |
|---|---|
| 1 - Données et préparation | 3 |
| 2 + 3 - Modèle de référence + CNN | 5 |
| 4 + 5 - Entraînement et évaluation | 3 |
| 6 - Expérimentations et analyse critique | 4 |
| 7 à 9 - Enrichissement | 2 |
| Soutenance, démonstration et maîtrise du travail | 3 |

Phases 1 à 6 : socle obligatoire. Enrichissement visé : phase 7 (data augmentation). Phases 8 et 9 seulement si tout le reste est terminé.

## Planning
| Date | Objectif |
|---|---|
| 10-01 / 10-02 | Phases 1 et 2 (faites) |
| 10-05 | Phase 3 + compléments phase 1 |
| 10-06 | Phase 4 (C0 sur Colab, comparaison B0) ; phase 5 sur la validation |
| 10-07 | Phase 6 (E1 à E3) et phase 7 (A1) |
| 10-08 | Choix du modèle final, évaluation test unique, notebook Colab "Exécuter tout", démo, slides |
| 10-09 | Répétition, soutenance |

## Phase 1 - Données (3 pts) - resp. :
- [x] `src/data.py` : `CLASS_NAMES`, chargement, split train/val/test, `preprocess_face()`. Fait quand : `X_train.shape == (N,48,48,1)`, valeurs dans [0,1], test intact.
- [x] Notebook : source/licence, nb d'images, classes, répartition par classe (graphique), dimensions, format, déséquilibre, 5 exemples par classe.
- [x] Explications markdown : rôle du redimensionnement, de la normalisation, de l'encodage, du split.
- [x] `tests/test_shapes.py` : formes, classes, stratification, reproductibilité et conservation des exemples (10 tests réussis, Python 3.11).
- [x] Compléments demandés par le sujet : nombre total d'images (35 887 = 28 709 train + 7 178 test), effectifs chiffrés par classe pour train et test, format source (JPEG, niveaux de gris, 48 x 48).

## Phase 2 - Baseline MLP (5 pts avec la phase 3) - resp. :
- [x] `build_mlp()` et préparation de l'entraînement B0 reproductible, sauvegarde de l'historique et des résultats (CSV temporaire : réexécution et refus des fichiers mal formés avant entraînement vérifiés).
- [x] Review locale : 23 tests réussis, smoke FER2013 antérieur (256 train, 64 validation, 1 epoch, sans ligne B0), rechargement des poids et notebook sans B0 vérifiés ; transfert Colab documenté.
- [x] B0 complet : JSON/CSV conformes, modèle rechargé (295943 paramètres), métriques de l'epoch 5 retrouvées sur la validation ; trace epochs/versions fournie et assertion GPU confirmée par l'utilisateur (sortie GPU non conservée). Review détaillée : `docs/B0_COLAB.md`.
- [x] Explications markdown : image -> vecteur, poids/biais, propagation avant, activations, loss, rétropropagation, descente de gradient, softmax ; appels et code des courbes prêts, affichage des résultats après B0.

## Phase 3 - CNN (5 pts avec la phase 2) - resp. : Maxime
- [x] `build_cnn()` paramétrable, au moins deux blocs `Conv2D + ReLU -> MaxPooling`, puis `Flatten -> Dense -> Dense(7, softmax)`. Fait quand : tableau couche par couche (shape de sortie + nb de paramètres) affiché une fois.
- [x] Explications markdown : filtres, convolution, feature maps, kernel, stride, padding, ReLU, pooling, Flatten, Dense, sortie ; évolution des dimensions couche par couche ; justification de l'architecture.

## Phase 4 - Entraînement (3 pts avec la phase 5) - resp. : Paul
- [x] `src/train.py` : compile, fit, callbacks (`ModelCheckpoint`, `EarlyStopping`), historique sauvegardé dans `training/logs/`. Raccordé à `build_cnn` ; vrai smoke CNN 256/64/1 temporaire et relecture des poids vérifiés.
- [x] Run Colab GPU du CNN de départ (id `C0`), courbes loss et accuracy train/val. Tesla T4, split complet 24 402/4 307, 12 epochs ; checkpoint de l'epoch 7 rechargé et validation concordante dans Colab et en local.
- [x] Comparaison B0 / C0 dans le notebook : val_acc, val_loss, nombre de paramètres. Tableau réel vérifié ; B0 dernière epoch 5, C0 meilleure epoch 7 sur val_loss ; métriques de la même epoch et B0 préservé.
- [x] Notebook : avec les flags `RUN_*` à `False`, recharger les historiques sauvegardés (`training/logs/*_history.json`) pour afficher les courbes B0 et C0 sans réentraîner. Affichage réel vérifié dans Colab avec fit interdit ; historique absent géré sans entraînement.
- [x] Explications markdown : choix loss/optimiseur/batch/epochs/métriques ; lecture des courbes, sur/sous-apprentissage. Analyse fondée sur le vrai C0 : surapprentissage après l'epoch 7 et limites de la comparaison B0/C0.

## Phase 5 - Évaluation (3 pts avec la phase 4) - resp. : Maxime
- [x] `src/evaluate.py` : charger un modèle sauvegardé, prédire, matrice de confusion, precision/recall/F1 par classe ; images brutes passées par `preprocess_face()`.
- [x] Analyse sur la **validation** pendant le développement (C0 puis candidats des phases 6 et 7) : classes bien reconnues, classes confondues, classes difficiles et pourquoi.
- [x] Exemples corrects et incorrects (image, vraie classe, prédiction, probabilité) ; plusieurs erreurs commentées.
- [ ] Test officiel : **une seule évaluation**, sur le modèle final, avec la même analyse (matrice, F1, exemples, limites). Mécanisme prêt (`RUN_TEST`, `evaluate_test_once`), à lancer sur le modèle final.

## Phase 6 - Expériences (4 pts) - resp. : Paul
- [ ] Au moins trois expériences, un axe à la fois : E1 architecture (couches, filtres, kernel), E2 régularisation (Dropout ou L2, sans augmentation), E3 learning rate / optimiseur / batch.
- [ ] `experiments.csv` complet + tableau comparatif dans le notebook (modification, résultats validation, observation).
- [ ] Choix du modèle final justifié sur la validation (parmi C0, E1 à E3 et A1), puis évaluation test unique (phase 5).

## Phase 7 - Enrichissement : data augmentation (2 pts avec les phases 8 et 9) - resp. : Maxime
- [ ] Data augmentation (id `A1`) sur le meilleur modèle de la phase 6, comparée sur la validation ; un seul enrichissement.

## Phase 8 - Détection de plusieurs visages (YOLO, extension optionnelle)
- [ ] Choisir et justifier le détecteur : un YOLO pré-entraîné sur COCO détecte des personnes, pas des visages ; utiliser un modèle entraîné sur des visages ou justifier une alternative.
- [ ] Pipeline : détection -> bounding boxes -> extraction -> `preprocess_face()` -> CNN -> bounding box + expression + score, sur 2-3 images à plusieurs personnes.
- [ ] Explications : classification vs détection, bounding box, confidence score, IoU, NMS, principe de YOLO, modèle pré-entraîné, fine-tuning, precision / recall / mAP.

## Phase 9 - Vidéo (extension bonus)
- [ ] Courte vidéo : images successives -> détection -> extraction des visages -> CNN -> prédictions -> affichage.

## Livrables et soutenance (3 pts) - resp. : commun
- [~] Notebook local rendu robuste au dossier de lancement ; validation Colab et présentation à terminer.
- [ ] Notebook Colab autonome : cellule de setup (récupération de `src/` et des données), "Exécuter tout" sans erreur sur runtime neuf ; cellule de démo du modèle final.
- [ ] Présentation (resp. Paul) : démarche, architectures, expériences, résultats, limites.
- [ ] Démonstration du modèle final.
- [ ] Répétition : chaque membre sait expliquer le fonctionnement du système, l'architecture du CNN, les principales parties du code, les choix, les résultats, les erreurs observées et les améliorations.
