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

---

# Workflow quotidien

```bash
# Ajouter une librairie
uv add seaborn==0.13.2

# Supprimer une librairie
uv remove seaborn

# Re-synchroniser sur un poste neuf
uv venv .venv
uv sync
```
