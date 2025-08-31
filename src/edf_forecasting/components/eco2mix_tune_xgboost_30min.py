import os
import yaml
import datetime
import optuna
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)

class XGBoostTuner:
    def __init__(self, n_trials=20, cv=3, timeout=None, seed=42, study_dir="optuna_studies"):
        self.n_trials = n_trials
        self.cv = cv
        self.timeout = timeout
        self.seed = seed

    def run(self, X, y):
        timestamp = datetime.datetime.now().strftime("tuning_%Y-%m-%d_%H-%M")
        base_dir = f"data/07_model_output/eco2mix/xgboost/30min/optuna_study/{timestamp}"
        os.makedirs(base_dir, exist_ok=True)

        study_path = os.path.join(base_dir, "optuna_study.db")
        params_path = os.path.join(base_dir, "best_params.yml")

        study = optuna.create_study(
            direction="minimize",
            study_name="xgb_tuning",
            storage=f"sqlite:///{study_path}",
            load_if_exists=True
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 80),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "n_jobs": -1,
                "random_state": self.seed
            }

            model = XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            return -scores.mean()

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        summary = {
            "study_path": study_path,
            "best_score_rmse": best_score,
            "best_params": best_params,
            "n_trials": self.n_trials,
            "cv": self.cv,
            "seed": self.seed,
            "timestamp": timestamp
        }

        with open(params_path, "w") as f:
            yaml.dump(summary, f)

        logging.info(f"Tuning complete. Params saved to : {params_path}")
        return best_params, study