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
