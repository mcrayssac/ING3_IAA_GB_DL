Fait : phase 1   chargement FER2013, preprocessing centralisé, split train/validation stratifié et notebook documenté.
Reste : relancer le test de shapes dans un environnement où les dépendances de requirements.txt sont installées.

Commande utile : python -m pytest -q
Fait : review phase 1   licence Kaggle explicitée ; test de l'ordre des classes et du type des labels renforcé.
Reste : relancer le test de shapes dans un environnement où les dépendances de requirements.txt sont installées.

Commande utile : python3 -m pytest -q
Fait : test de shapes exécuté dans un environnement temporaire ; 1 test réussi.
Reste : exécuter le notebook complet sur Colab avec le dossier FER2013 (data/ est absent localement).

Commande utile : /private/tmp/fer-phase1-review-venv/bin/python -m pytest -q
Fait : README complet de niveau ING3, avec schéma du pipeline, protocole, état réel, guide et limites.
Reste : installer les dépendances puis valider le README en même temps que les prochains livrables Colab.

Commande utile : git diff --check && python3 -m pytest -q
Fait : phase 9  - notebook robuste depuis la racine ou `notebooks/` ; artefact `.ipynb` ignoré.
Reste : créer un environnement Python du projet, puis tester le notebook avec FER2013 ; run complet sur Colab.

Commande utile : jupyter lab notebooks/projet.ipynb
Fait : phase 1 validée sur données synthétiques : 10 tests ; contrats labels, README et versions des dépendances vérifiés sous Python 3.11.15, imports dont TensorFlow 2.21.0 et kernel contrôlés.
Reste : validation du notebook complet sur Colab avec FER2013, hors de cette revue ; aucun entraînement ni évaluation du test officiel effectué.

Commande utile : .venv/bin/python -m pytest -q && .venv/bin/python -m pip check && git diff --check
Fait : chargement FER2013  - données absentes du dépôt, racine vérifiée depuis racine/notebooks ; contrôle train/test et installation documentés ; 10 tests réussis sous Python 3.11.15.
Reste : installer FER2013 dans data/train et data/test puis vérifier le chargement réel ; kernel actif non observable, aucun entraînement effectué.

Commande utile : .venv/bin/jupytext --to ipynb notebooks/projet.py puis .venv/bin/jupyter lab notebooks/projet.ipynb (synchronisation manuelle, non exécutée ici).
Fait : phase 2  - MLP, protocole B0 et explications prêts ; 11 tests réussis, smoke FER2013 256 train/64 val/1 epoch et artefacts vérifiés, CSV testé en temporaire sans ligne B0. Phase 1 validée localement par l'utilisateur.
Reste : B0 complet sur Colab, récupération/vérification JSON/CSV/poids puis finalisation phase 2 ; validation globale Colab distincte, aucune performance B0 disponible.

Commande utile : .venv/bin/python -m pytest -q && .venv/bin/python -m src.train ; pour B0, suivre README et activer RUN_B0 sur Colab.
Fait : review phase 2  - CSV invalide refusé avant entraînement/sauvegarde, transfert Colab explicite ; 15 tests réussis, smoke FER2013 isolé 256/64/1, rechargement/dernière epoch et notebook sans B0 depuis racine/notebooks vérifiés ; artefacts existants préservés.
Reste : B0 complet sur Colab puis récupérer/vérifier JSON/CSV/modèle et clôturer phase 2 ; installation/GPU et exécution Colab non vérifiés, aucune ligne B0 créée dans le dépôt.

Commande utile : .venv/bin/python -m pytest -q && git diff --check ; suivre README (archives locales, préparation Colab, RUN_B0=True, 5 epochs).
Fait : guide B0 du README reformulé pas à pas, avec explications simples des manipulations et des résultats.
Reste : exécuter B0 complet sur Colab puis récupérer et vérifier ses résultats ; aucun entraînement lancé.

Commande utile : git diff --check ; suivre la section « Faire B0 sur Google Colab, pas à pas » du README.
Fait : préparation B0 Colab corrigée après conflits pip signalés ; versions natives conservées, imports/GPU/versions contrôlés par la cellule, remise à zéro documentée.
Reste : réinitialiser la session Colab altérée puis renvoyer les ZIP ; compatibilité et B0 complet à vérifier sur Colab.

Commande utile : retirer `%pip install -q -r /content/fer2013-project/requirements.txt` de la cellule Colab ; git diff --check.
Fait : phase 2 validée, B0 JSON/CSV/modèle conformes et validation rechargée concordante ; 23 tests, notebook sans B0 et courbes vérifiés ; guide déplacé, CSV renforcé, artefacts préservés.
Reste : phase 9 Colab globale ; GPU confirmé par assertion selon utilisateur, sortie matérielle non conservée ; commit/push/MR à réaliser par utilisateur.
Commande utile : .venv/bin/python -m pytest -q && git diff --check ; reproduction et preuves : docs/B0_COLAB.md.

Fait : phase 3  - `build_cnn()` (3 blocs Conv2D + ReLU -> MaxPooling 32/64/128, Dense 128, sortie softmax 7, 683 527 paramètres) et test dédié ; notebook : summary couche par couche, notions CNN, calcul des paramètres, justification et hypothèse sur C0. Compléments phase 1 : total 35 887 images, effectifs par classe train/test, fichiers JPEG gris 48 x 48. 24 tests réussis, notebook exécuté sans erreur avec RUN_B0=False.
Reste : phase 4  - entraîner C0 sur Colab GPU, courbes et comparaison avec B0 ; performances de C0 encore inconnues.
Commande utile : .venv/bin/python -m pytest -q && .venv/bin/jupytext --to ipynb notebooks/projet.py
Fait : phase 4 indépendante sur phase-4-entrainement, fabrique/callbacks/meilleure epoch ; 15 tests pertinents, smoke MLP 256/64/1 temporaire et relecture conformes, B0/CSV inchangés, test officiel non utilisé.
Reste : raccorder build_cnn après le push de Maxime, exécuter C0 complet sur Colab GPU et rédiger les explications notebook ; tâche technique [~], aucune ligne C0 réelle, aucun commit/push.
Commande utile : .venv/bin/python -m pytest -q tests/test_shapes.py -k 'cnn or mlp or experiment or invalid_csv' ; git diff --check.
Fait : main et phase-4-entrainement à jour avec origin/main (231d373) ; préparation locale restaurée, conflits TODO/JOURNAL résolus ; 26 tests réussis et diff propre ; ancien SPEC local sauvegardé dans /private/tmp/fer2013-phase4-sync-6_1ko0bw/.
Reste : smoke du vrai CNN, intégration notebook, C0 Colab GPU, comparaison B0/C0 et rechargement des historiques ; aucune évaluation test ni commit/push.
Commande utile : .venv/bin/python -m pytest -q ; git diff --check.
Fait : phase 4 terminée ; smoke CNN 256/64/1 temporaire, C0 Colab Tesla T4 complet 24 402/4 307 (12 epochs, meilleure 7), JSON/CSV/checkpoint/trace importés et validation rechargée concordante ; courbes/tableau flags False, analyse réelle, 26 tests et B0 préservé.
Reste : aucune tâche de phase 4 ; aucune évaluation du test officiel, autre phase, dépendance nouvelle, intégration Git, commit ou push ; checkpoint C0 non versionné.
Commande utile : .venv/bin/python -m src.train --model cnn ; .venv/bin/python -m src.train --check-c0 ; .venv/bin/python -m pytest -q ; git diff --check ; preuves/reproduction : docs/B0_COLAB.md.

Fait : phase 5  - `src/evaluate.py` (évaluation, rapport par classe, exemples confiants, `predict_faces`, test unique par création exclusive de `test_evaluation.json`) et test dédié ; C0.keras extrait et contrôlé (`--check-c0`). Notebook : matrice de confusion, paires confondues, rapport par classe, exemples, F1 B0/C0 et analyse rédigée sur la validation (C0 : accuracy 0,556536, F1 macro 0,512 contre 0,244 pour B0). 27 tests réussis, notebook exécuté sans erreur, test officiel non évalué.
Reste : évaluation unique du test sur le modèle final après les phases 6 et 7 (`RUN_TEST=True`, `FINAL_MODEL_ID`).
Commande utile : .venv/bin/python -m pytest -q && .venv/bin/python -m src.train --check-c0
Fait : phase 6 sur phase-6-experiences ; E1/E2/E3 complets Colab Tesla T4, JSON/CSV/checkpoints et split vérifiés, B0/C0 préservés ; smoke temporaires 256/64/1, 30 tests, relecture locale/Colab flags False avec fit interdit, courbes/tableau/analyse et notebook Jupytext généré.
Reste : Maxime, phase 7 A1 depuis la configuration E3 (CNN C0, Adam 0,0005, batch 64, seed 42 ; checkpoint E3.keras epoch 7, val_loss 1,192440 / val_acc 0,559322) ; choix définitif après A1, RUN_TEST=False et FINAL_MODEL_ID=None ; aucun commit/push.
Commande utile : Colab GPU train_experiment("E3", X_train, y_train, X_val, y_val) ; contrôle verify_cnn_run(X_val, y_val, run_id="E3", X_train=X_train, y_train=y_train) ; .venv/bin/jupytext --to ipynb notebooks/projet.py ; git diff --check.

Fait : phase 7, préparation A1  - E1/E2/E3.keras extraits et vérifiés sur la validation (E3 : 0,559322 / 1,192440). A1 = E3 avec poids neufs, même split et budget, augmentation du train seule dans `tf.data` (`_augmentation()` : flip, rotation ±18°, translation/zoom ±15 %, contraste ±20 %) ; `config["augmentation"]` et `verify_cnn_run` suivent ce réglage. Test d'augmentation (forme, plage, reproductibilité) et smoke A1 local réussis ; notebook phase 7 (grille d'augmentation, `RUN_A1`, comparaison et F1 E3/A1) exécuté sans erreur, flags False, test officiel non évalué.
Reste : run A1 sur Colab GPU (`RUN_A1=True` seul), import et vérification des artefacts, analyse, choix du modèle final par le critère fixé, puis évaluation unique du test.
Commande utile : .venv/bin/python -m pytest -q ; Colab : RUN_A1=True dans la section phase 7.

Fait : phase 7 et modèle final  - A1 entraîné sur Colab GPU, artefacts importés (seule la ligne A1 ajoutée au CSV) et vérifiés en local : epoch 24/29, val_loss 1,126252, val_acc 0,577200 (E3 : 1,192440 / 0,559322). Analyse A1 rédigée avant le test : surapprentissage retardé, F1 macro 0,507 → 0,502 (disgust, fear en recul). Modèle final A1 par le critère fixé ; test officiel évalué une seule fois : accuracy 0,575508, loss 1,131685, F1 macro 0,508 ; seconde évaluation refusée. Analyse test rédigée dans le notebook.
Reste : livrables  - notebook Colab « Exécuter tout » sur runtime neuf, cellule de démo du modèle final, slides, répétition.
Commande utile : .venv/bin/python -m pytest -q ; aucun nouveau test officiel (test_evaluation.json fait foi).
