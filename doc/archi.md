# Architecture, note de cadrage

## Situation initiale

Au départ, le projet fonctionnait entièrement en local, sans conteneurs.

Un seul environnement gérait à la fois :
- les pipelines de données ;
- le scraping ;
- l'entraînement et le réentraînement ;
- le suivi MLflow ;
- le stockage local des artefacts et modèles ;
- l'API.

Le projet fonctionnait, mais tout reposait sur un fonctionnement local.  
Les runs MLflow, les artefacts et les modèles étaient stockés dans `mlruns` sur disque.

## Problème rencontré

Le besoin de restructuration est apparu lorsque l'API a commencé à être dockerisée.

À ce moment-là, un problème est apparu : même si un modèle existait bien dans MLflow, l'API conteneurisée ne parvenait pas à y accéder correctement. Le point bloquant venait du fait que le modèle et les artefacts étaient encore liés à un stockage local, non partagé proprement avec le conteneur API.

Autrement dit, le projet fonctionnait en local, mais cette architecture ne convenait plus dès qu'on séparait les composants.

## Direction retenue

L'objectif est maintenant de migrer vers une architecture plus propre, dans laquelle les services principaux sont dockerisés :

- MLflow Server ;
- MinIO ;
- API ;
- monitoring.

L'entraînement et le réentraînement ne sont pas dockerisés pour l'instant.

Ce choix est volontaire. Le training reste le composant le plus lourd et le plus mouvant du projet, avec :
- beaucoup de fichiers à lire ;
- des données parfois volumineuses ;
- du scraping ;
- des pipelines Kedro ;
- des besoins fréquents de relance et d'itération.

Le dockeriser maintenant ajouterait de la complexité sans résoudre le problème principal.

## Architecture cible actuelle

Le fonctionnement visé est le suivant :

- le training local envoie ses runs vers un MLflow Server centralisé ;
- MLflow stocke les métadonnées dans une base SQLite persistée ;
- MLflow stocke les artefacts, notamment les modèles, dans MinIO ;
- l'API dockerisée ne dépend plus d'aucun chemin local ;
- l'API charge les modèles via MLflow.

## Idée principale

Le point important est de sortir d'un fonctionnement basé sur `mlruns` local comme source de vérité.

La nouvelle source de vérité devient :
- MLflow Server pour le suivi et le registry ;
- SQLite pour les métadonnées ;
- MinIO pour les artefacts.

## Décision actuelle

Pour l'instant :
- on dockerise les services stables ;
- on garde le training hors conteneur ;
- on adapte le code d'entraînement pour qu'il loggue correctement vers MLflow Server ;
- on supprime progressivement la dépendance au stockage local `mlruns`.

Le sujet de la dockerisation du training pourra être revu plus tard, une fois l'architecture MLflow + MinIO stabilisée.