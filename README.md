# edf-forecasting

> **Pipeline MLOps pour la prévision de la consommation électrique en France**

## **Description**

### **Description générale**

Ce projet vise à concevoir et mettre en place un **pipeline complet de Machine Learning** dédié à la **prévision de la consommation électrique en France**, à partir de données de consommation mesurées à un pas de temps infra-journalier (30 minutes).

### **Architecture fonctionnelle**

Le projet s’articule autour de trois briques principales :

#### **Kedro**

Kedro est utilisé pour construire et orchestrer la pipeline Machine Learning, couvrant l’ensemble du cycle de vie du modèle, depuis l’acquisition des données jusqu’à l’évaluation finale.

La pipeline inclut les étapes suivantes :

* Récupération des données de consommation électrique en France

* Nettoyage et pré-traitement des données

* Construction des jeux de données d’entraînement, de validation et de test

* Optimisation des hyperparamètres du modèle

* Entraînement et évaluation du modèle

* Production des artefacts de sortie pour *MLflow*

#### **MLflow**

Les différentes exécutions de pipeline sont suivies via *MLflow* (outil de tracking d’expériences).

* l’enregistrement des paramètres d’entraînement,
* le suivi des métriques de performance,
* la sauvegarde des modèles entraînés,
* la comparaison des différentes versions de modèles.

#### **FastAPI**

*FastAPI* est utilisée pour :

* le chargement du modèle depuis MLflow

* l’exposition du modèle

* la mise à disposition de routes de prédiction et de feedback

## **Installation**

### **Prérequis**

Le projet nécessite à minima les outils suivants :

* **Python** ≥ 3.11
* **Git** ≥ 2.39.5
* **uv** ≥ 0.6.12

### **Installation de `uv`**

Executez la commande ci-dessous:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Rechargez le shell:

```bash
source ~/.bashrc
```

Vérifiez l’installation :

```bash
uv --version
```

**Voir la note d'installation ci-dessous**

> [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### **Récupération du projet**

Clonez le dépôt Git :

```bash
git clone https://github.com/HassaniJawaheer/edf-forecasting.git
cd edf-forecasting
```

### **Installation de l’environnement et des dépendances**

Créer et synchroniser l’environnement :

```bash
uv sync
```
Cette commande crée un environnement virtuel isolé, installe l’ensemble des dépendances du projet.

### **Tests**

Lancez une pipeline simple:

```bash
uv run kedro run --pipeline=hello_mlflow
```

## **Lancement**

Pour exécuter la pipeline complète d’entraînement et d’évaluation du modèle :

```bash
uv run kedro run --pipeline=xgboost_time_series
```

Une fois la pipeline terminée et les artefacts produits, démarrez l’API:

```bash
uv run uvicorn src.edf_forecasting_api.main:app --reload --port 8000
```

Pour tester l’API :

```bash
python3 edf_forecasting_api/evaluate_model.py
```

Ce script exécute des requêtes de prédiction à partir de données de référence et envoie des feedbacks à l’API.
