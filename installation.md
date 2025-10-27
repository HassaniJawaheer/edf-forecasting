# Initialisation de Kedro avec uv

```bash
# Vérifier si uv est installé
uv --version

# (Optionnel) Installer uv si absent
pip install uv

# Créer un projet Kedro vierge
uvx --python 3.12 kedro new
# Répondre :
# - Project name : edf-forecasting
# - Package name : edf_forecasting
# - Tools : 1,2,3,5 (Ruff, PyTest, Logging, Data folder)
# - Example pipeline : Yes (pour tester)

# Aller dans le projet
cd edf-forecasting

# Créer un environnement local
uv venv .venv
source .venv/bin/activate

# Ajouter Kedro au projet
uv add kedro==0.19.12

# Ajouter MLflow et les dépendances stables dans pyproject.toml
# (voir bloc dependencies)

# Générer un lock
uv lock -U

# Synchroniser l’environnement avec le lock
uv sync

# Vérification rapide
uv run kedro --version
uv run python -c "import mlflow, numpy; print(mlflow.__version__, numpy.__version__)"
```

## Workflow quotidien

```bash
# Ajouter une librairie
uv add seaborn==0.13.2

# Supprimer une librairie
uv remove seaborn

# Re-synchroniser sur un poste neuf
uv venv .venv
uv sync
```

## Intégration MLflow

```bash
# Initialiser la configuration MLflow (création conf/local/mlflow.yml)
uv run kedro mlflow init
# → ajoute les fichiers mlflow.yml et credentials.yml

# Vérifier les plugins actifs
uv run kedro info

# Lancer le pipeline (exemple hello)
uv run kedro run

# Lancer l’interface web de MLflow
uv run kedro mlflow ui
uv run mlflow ui --backend-store-uri mlruns

# Corriger la version de MLflow pour compatibilité Python 3.12
uv add -U mlflow==2.12.2
uv sync

# Créer un pipeline de test MLflow
uv run kedro pipeline create hello_mlflow
uv run kedro pipeline create prepare_data_30min
```
# Lancer des pipes
uv run kedro run --pipeline=prepare_data --from-nodes="scrape_data" --to-nodes="prestructure_data"
uv run kedro run --pipeline=prepare_data
uv run kedro run --pipeline=prepare_data --from-nodes="scrape_data"

# Lancer l'api
uv run uvicorn src.edf_forecasting_api.main:app --reload --port 8000

# Test de l'api 
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [
         [
           74494.0, 73481.0, 71506.0, 71505.0, 71134.0, 70856.0, 68840.0, 67315.0,
           65749.0, 64838.0, 64041.0, 64379.0, 64210.0, 64469.0, 64437.0, 64559.0,
           64785.0, 64281.0, 64292.0, 64862.0, 65353.0, 65879.0, 66180.0, 66643.0,
           66901.0, 67719.0, 68547.0, 66745.0, 65090.0, 63891.0, 62228.0, 61554.0,
           61263.0, 61469.0, 62443.0, 65700.0, 68890.0, 70497.0, 71766.0, 72562.0,
           72184.0, 71493.0, 70440.0, 69167.0, 68044.0, 68829.0, 71485.0, 70639.0
         ]
       ],
       "n_predictions": 48
     }'

```


## Gestion des versions MLflow

```bash
# Forcer la mise à jour MLflow → compatibilité Python 3.12
uv add -U mlflow==2.12.2

# Re-synchroniser l’env
uv sync
```

La config suivante a permis de valider le run :

```yaml
server:
  mlflow_tracking_uri: mlruns

tracking:
  experiment:
    name: edf_forecasting
```

# Initialisation projet → push GitHub

```bash
# 1. Initialiser un dépôt Git
git init
# crée un dépôt local vide, première branche = master par défaut

# 2. Renommer master → main
git branch -M main
# -M force le renommage, même si la branche existe déjà

# 3. Vérifier l’état du dépôt
git status
# montre quels fichiers sont suivis, modifiés, non suivis

# 4. Ajouter les fichiers à suivre (staging area)
git add .

# 5. Enregistrer un commit
git commit -m "Init projet EDF Forecasting"
# -m permet d’ajouter un message directement
# si tu lances "git commit" sans git add avant → rien ne sera commité

# 6. Ajouter le lien vers le repo GitHub
git remote add origin git@github.com:HassaniJawaheer/edf-forecasting.git
# "origin" = nom par défaut du dépôt distant

# 7. Pousser la branche main vers GitHub
git push -u origin main
# -u crée un lien entre ta branche locale et distante
```
