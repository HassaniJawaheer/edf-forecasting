# edf-forecasting

[![CI EDF Forecasting](https://github.com/HassaniJawaheer/edf-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/HassaniJawaheer/edf-forecasting/actions/workflows/ci.yml)

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

Tested with:
* **Python 3.12**
* **Git 2.39.5**
* **uv 0.6.12**
* **Docker **28.2.1**
* **kubectl 1.35.0**
* **kind 0.32.0 (alpha)**

### Installation de Docker

Installer Docker Engine :

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
```
Ajoutee l'utilisateur au groupe Docker :

```bash
sudo usermod -aG docker $USER
newgrp docker
```
Petite vérification :

```bash
docker run hello-world
```

### **Installation de Kubernetes (kubectl et kind)**

**IMPORTANT** : À installer uniquement après l’installation de Docker.

#### Installer `kubectl` :

```bash
curl -LO https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

Vérification :

```bash
kubectl version --client
```

#### Installer `kind` :

```bash
curl -Lo kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
chmod +x kind
sudo mv kind /usr/local/bin/
```

Vérification :

```bash
kind version
```

#### Créer un cluster Kubernetes local :

```bash
kind create cluster --name mlops-local
```

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

## **Continuous Integration**

Une chaîne d’intégration continue est exécutée à chaque push et pull request.  
Elle valide l’installation de l’environnement, l’exécution d'une pipeline de teste et les tests unitaires.

## **Lancement**

Pour exécuter la pipeline complète d’entraînement et d’évaluation du modèle :

```bash
uv run kedro run --pipeline=xgboost_time_series
```

Enuite, lancez le serveur MLflow :

```bash
uv run mlflow ui --backend-store-uri mlruns
```

Une fois la pipeline terminée et les artefacts produits, démarrez l’API:

```bash
uv run uvicorn src.edf_forecasting_api.main:app --reload --port 8000
```

Pour tester l’API :

```bash
.venv/bin/python src/edf_forecasting_api/evaluate_model.py
```

Ce script exécute des requêtes de prédiction à partir de données de référence et envoie des feedbacks à l’API.
