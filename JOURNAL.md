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
Fait : phase 9 — notebook robuste depuis la racine ou `notebooks/` ; artefact `.ipynb` ignoré.
Reste : créer un environnement Python du projet, puis tester le notebook avec FER2013 ; run complet sur Colab.

Commande utile : jupyter lab notebooks/projet.ipynb
Fait : phase 1 validée sur données synthétiques : 10 tests ; contrats labels, README et versions des dépendances vérifiés sous Python 3.11.15, imports dont TensorFlow 2.21.0 et kernel contrôlés.
Reste : validation du notebook complet sur Colab avec FER2013, hors de cette revue ; aucun entraînement ni évaluation du test officiel effectué.

Commande utile : .venv/bin/python -m pytest -q && .venv/bin/python -m pip check && git diff --check

Fait : chargement FER2013 — données absentes du dépôt, racine vérifiée depuis racine/notebooks ; contrôle train/test et installation documentés ; 10 tests réussis sous Python 3.11.15.
Reste : installer FER2013 dans data/train et data/test puis vérifier le chargement réel ; kernel actif non observable, aucun entraînement effectué.
Commande utile : .venv/bin/jupytext --to ipynb notebooks/projet.py puis .venv/bin/jupyter lab notebooks/projet.ipynb (synchronisation manuelle, non exécutée ici).
