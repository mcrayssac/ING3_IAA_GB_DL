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

Phases 1 à 6 : socle obligatoire. Enrichissements réalisés : phase 7 (data augmentation) et phase 8 (pipeline final multi-visages).

## Planning
| Date | Objectif |
|---|---|
| 10-01 / 10-02 | Phases 1 et 2 (faites) |
| 10-05 | Phase 3 + compléments phase 1 (faits) |
| 10-06 | Phase 4 (C0 sur Colab, comparaison B0) ; phase 5 sur la validation (faits) |
| 10-07 | Phases 6, 7 et 8 ; modèle final A1 ; test officiel unique (faits) ; conformité au sujet |
| 10-08 | Approfondissement CNN (plusieurs seeds), notebook Colab "Exécuter tout", slides ; optionnel : YOLO visages |
| 10-09 | Répétition, soutenance |

## Conformité au sujet (audit du 2026-10-07)
Une case par consigne du PDF, avec la section du notebook ou le fichier qui la couvre.

**Avant de commencer - binaire et multiclasse**
- [x] Sigmoïde pour 2 classes, une sortie par classe et softmax pour K classes, formule softmax, `Dense(7, activation="softmax")` : Phase 2.

**Partie 1 - données**
- [x] Source : lien, auteurs, licence d'utilisation : en-tête du notebook.
- [x] Nombre total d'images (35 887), classes, nombre d'images par classe (train et test) : Phase 1.
- [x] Dimensions et format (JPEG, niveaux de gris, 48 x 48), déséquilibre (disgust) : Phase 1.
- [x] Plusieurs exemples par catégorie (5 par classe) : Phase 1.
- [x] Rôle du redimensionnement, de la normalisation, des labels, de l'encodage et du split train/validation/test : Phase 1.
- [x] Test jamais utilisé comme validation : split du train, `evaluate_test_once` (Phase 5, test officiel).

**Partie 2 - modèle de référence**
- [x] Image → Flatten → Dense + activation → Dense → softmax (B0) : Phase 2.
- [x] Explications : image en entrée, poids et biais, propagation avant, activations, loss, rétropropagation, descente de gradient, sortie multiclasse : Phase 2.

**Partie 3 - CNN**
- [x] Plusieurs blocs convolution + pooling, architecture proposée et justifiée : Phase 3.
- [x] Filtres, convolution, feature maps, kernel, stride, padding, ReLU, pooling, Flatten, Dense, couche de sortie : Phase 3.
- [x] Dimensions et nombre de paramètres couche par couche : Phase 3 (summary et tableau de calcul).

**Partie 4 - entraînement**
- [x] Justification de la loss, de l'optimiseur, du batch size, du nombre d'epochs et des métriques : Phase 4.
- [x] Courbes loss et accuracy train/validation, interprétation, surapprentissage : Phases 4, 6 et 7.

**Partie 5 - évaluation et erreurs**
- [x] Matrice de confusion et performances par expression : Phase 5 et test officiel.
- [x] Expressions bien reconnues, confondues, difficiles et pourquoi : Phase 5 et test officiel.
- [x] Exemples image + vraie classe + classe prédite + probabilité, prédictions incorrectes analysées : Phase 5 et test officiel.

**Partie 6 - expériences**
- [x] Au moins trois expériences, un nombre limité de paramètres changé à chaque fois : Phase 6 (E1 à E3).
- [x] Tableau comparatif (modification, résultats validation, observations) : Phases 6 et 7.
- [x] Choix du modèle final justifié : Phase 7 (critère fixé avant les runs).
- [ ] Approfondissement CNN : objectifs tirés de l'analyse, échelle d'améliorations, recherche aléatoire, moyenne sur plusieurs seeds (10-08).

**Partie 7 - enrichissement**
- [x] Data augmentation (A1) : Phase 7.
- [x] Régularisation (Dropout E2, augmentation A1) : Phases 6 et 7.
- [x] Détection d'objets : Phase 8.
- [ ] Transfer learning et réseaux pré-entraînés appliqués au classifieur : non exploré (optionnel). YuNet est déjà un détecteur pré-entraîné.

**Partie 8 - plusieurs visages**
- [x] Image → détection des visages → boîtes → extraction → CNN → expressions : Phase 8, pipeline final.
- [x] Pour chaque visage : boîte, expression prédite et score : Phase 8.
- [x] Détecteur entraîné sur des visages (pas un YOLO COCO), choix justifié : Phase 8 (YuNet).
- [x] Même prétraitement que l'entraînement (`preprocess_face`) : Phase 8, `src/detect.py`.
- [x] Notions : classification vs détection, bounding box, confidence score, IoU, NMS, principe de YOLO, modèle pré-entraîné, fine-tuning, precision, recall, mAP : Phase 8.
- [ ] Optionnel : détecteur YOLO entraîné sur des visages, comparé à YuNet.

**Partie 9 - vidéo (bonus)**
- [ ] Vidéo → images successives → détection → extraction → CNN → affichage : non faite ; l'API de la phase 8 est réutilisable image par image.

**Livrables (soutenance du 9 octobre 2026)**
- [~] Notebook Colab structuré, commenté, exécutable de bout en bout : sommaire et setup automatique Mac/Colab ajoutés ; « Exécuter tout » à valider sur un runtime Colab neuf.
- [ ] Présentation : démarche, architectures, expériences, résultats.
- [x] Démonstration du modèle final : Phase 8 (photos de démo et `RUN_CUSTOM_IMAGE` pour une photo libre).

**Soutenance - chaque membre sait expliquer**
- [ ] Fonctionnement du système, architecture du CNN, principales parties du code, choix réalisés, résultats, erreurs observées, améliorations apportées.

## Phase 1 - Données (3 pts) - resp. :
- [x] `src/data.py` : `CLASS_NAMES`, chargement, split train/val/test, `preprocess_face()`. Fait quand : `X_train.shape == (N,48,48,1)`, valeurs dans [0,1], test intact.
- [x] Notebook : source/licence, nb d'images, classes, répartition par classe (graphique), dimensions, format, déséquilibre, 5 exemples par classe.
- [x] Explications markdown : rôle du redimensionnement, de la normalisation, de l'encodage, du split.
- [x] `tests/test_shapes.py` : formes, classes, stratification, reproductibilité et conservation des exemples (10 tests réussis, Python 3.11).
- [x] Compléments demandés par le sujet : nombre total d'images (35 887 = 28 709 train + 7 178 test), effectifs chiffrés par classe pour train et test, format source (JPEG, niveaux de gris, 48 x 48).

## Phase 2 - Baseline MLP (5 pts avec la phase 3) - resp. :
- [x] `build_mlp()` et préparation de l'entraînement B0 reproductible, sauvegarde de l'historique et des résultats (CSV temporaire : réexécution et refus des fichiers mal formés avant entraînement vérifiés).
- [x] Review locale : 23 tests réussis, smoke FER2013 antérieur (256 train, 64 validation, 1 epoch, sans ligne B0), rechargement des poids et notebook sans B0 vérifiés ; transfert Colab documenté.
- [x] B0 complet : JSON/CSV conformes, modèle rechargé (295943 paramètres), métriques de l'epoch 5 retrouvées sur la validation ; trace epochs/versions fournie et assertion GPU confirmée par l'utilisateur (sortie GPU non conservée). Review détaillée : `docs/COLAB.md`.
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
- [x] Test officiel : **une seule évaluation**, sur le modèle final A1, avec la même analyse (matrice, F1, exemples, limites). Accuracy 0,575508 / loss 1,131685 (validation 0,577200 / 1,126252) ; `test_evaluation.json` versionné, seconde évaluation refusée.

## Phase 6 - Expériences (4 pts) - resp. : Paul
- [x] Trois expériences indépendantes de C0 : E1 Dense 64 (epoch 6/11), E2 Dropout 0,3 sans augmentation (8/13), E3 learning rate 0,0005 (7/12), runs complets Colab Tesla T4 ; JSON/CSV/checkpoints et split vérifiés en local, B0/C0 préservés. Smoke 256/64/1 temporaires, 30 tests réussis et diff propre.
- [x] `experiments.csv` complet, courbes/tableau et analyse réelle dans le notebook ; relecture locale et Colab flags False avec fit interdit, cas sans artefacts vérifié en local. Relais Maxime : E3, val_loss 1,192440 / val_acc 0,559322, checkpoint `training/checkpoints/E3.keras`, configuration `training/logs/E3_history.json` ; critère fixé avant runs : val_loss minimale, puis accuracy maximale en cas d'égalité. Reproduction et limites documentées dans README/notebook.
- [x] Choix définitif parmi C0, E1 à E3 et A1 par le critère fixé avant les runs : **A1** (val_loss 1,126252, val_acc 0,577200), puis évaluation test unique (phase 5). `FINAL_MODEL_ID="A1"`, `RUN_TEST=False`.
- [ ] Approfondissement CNN (resp. Maxime) : objectifs tirés de l'analyse des données et des erreurs, échelle d'améliorations justifiées (BatchNorm, blocs VGG, largeur, dropout, budget + ReduceLROnPlateau, class weights), puis recherche aléatoire ; sélection sur la val_loss moyenne de 3 seeds ; une réévaluation du test déclarée si un modèle plus solide est retenu.

## Phase 7 - Enrichissement : data augmentation (2 pts avec les phases 8 et 9) - resp. : Maxime
- [x] Data augmentation (id `A1`) : E3 + augmentation du train dans `tf.data` (flip, rotation ±18°, translation/zoom ±15 %, contraste ±20 %), poids neufs, même split et budget ; run Colab GPU vérifié en local (epoch 24/29). Par rapport à E3 : val_loss −0,066188, val_acc +1,79 point, surapprentissage retardé ; F1 macro 0,507 → 0,502 (disgust et fear en recul).

## Phase 8 - Pipeline final : détection de plusieurs visages - resp. : Paul (détection), Maxime (intégration)
- [x] Intégration au flux principal : section « Phase 8 - pipeline final » après le test officiel, classifieur suivi par `FINAL_MODEL_ID`, démonstration du modèle final et photo libre (`RUN_CUSTOM_IMAGE`).
- [ ] Optionnel : détecteur YOLO entraîné sur des visages (ultralytics + torch, licence des poids à vérifier), comparé à YuNet sur les photos de démo.
- [x] Choisir et justifier le détecteur : un YOLO pré-entraîné sur COCO détecte des personnes, pas des visages ; utiliser un modèle entraîné sur des visages ou justifier une alternative.
- [x] Pipeline : détection -> bounding boxes -> extraction -> `preprocess_face()` -> CNN -> bounding box + expression + score, sur 2-3 images à plusieurs personnes.
- [x] Explications : classification vs détection, bounding box, confidence score, IoU, NMS, principe de YOLO, modèle pré-entraîné, fine-tuning, precision / recall / mAP.

Vérifié en local : YuNet ONNX officiel (MIT, OpenCV existant) + A1 ; trois photos NASA, trois visages chacune, annotations inspectées ; 40 tests, cellules de démo exécutées, téléchargement depuis zéro contrôlé, notebook converti. README épuré. Pas de relance Colab, d'entraînement ni de nouvelle évaluation du test. API de relais : `load_models()`, `detect_expressions()`, `annotate_faces()` dans `src/detect.py`.

## Phase 9 - Vidéo (extension bonus)
- [ ] Courte vidéo : images successives -> détection -> extraction des visages -> CNN -> prédictions -> affichage.

## Livrables et soutenance (3 pts) - resp. : commun
- [~] Notebook local rendu robuste au dossier de lancement ; validation Colab et présentation à terminer.
- [~] Notebook Colab autonome : cellule de setup automatique Mac/Colab (upload de `dist/projet-code.zip` et `dist/fer2013.zip` produits par `scripts/colab_bundle.sh`) et démo du modèle final faites ; "Exécuter tout" à valider sur un runtime Colab neuf.
- [ ] Présentation (resp. Paul) : démarche, architectures, expériences, résultats, limites.
- [x] Démonstration du modèle final : Phase 8, pipeline final (photos de démo, photo libre avec `RUN_CUSTOM_IMAGE`), classifieur choisi par `FINAL_MODEL_ID`.
- [ ] Répétition : chaque membre sait expliquer le fonctionnement du système, l'architecture du CNN, les principales parties du code, les choix, les résultats, les erreurs observées et les améliorations.
