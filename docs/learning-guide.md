# Comprendre le projet

Distinguer apprentissage et exploitation : le modèle vient d'une étude figée ; la plateforme ajoute traçabilité, contrat d'entrée et surveillance. Lire data.py puis prepare.py, api.py et monitoring.py. Expliquer pourquoi les stations et dates de test restent exclues de la sélection.

Un modèle doit être produit localement et digne de confiance avant joblib.load. Le hash détecte une modification. MLflow enregistre les essais ; le manifeste JSON définit la version locale chargée.

L'API reçoit les 16 caractéristiques avec leurs unités. Refaire un exemple réel, observer la réponse, puis changer RH à 101 : un rejet 422 doit apparaître. Lire le journal SQLite et comparer les sources manual/historical-replay.

PSI décrit une variation des entrées. Une saison différente peut déclencher ce signal sans démontrer une erreur. Pour mesurer une dérive de performance, il faut collecter les résultats terrain différés. Aucune précision d'irrigation ni économie d'eau n'est connue.

Exercice : expliquer pourquoi les temps de TestClient ne mesurent pas la capacité d'un serveur public. Avant un déploiement, vérifier le conteneur, l'authentification et un test de charge réaliste.
