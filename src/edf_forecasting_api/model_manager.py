import mlflow
import logging
import threading
import numpy as np
from typing import List
import time

class ModelManager:
    def __init__(self, model_name: str, check_interval: int = 300):
        self.model_name = model_name
        self.check_interval = check_interval
        self.model = None
        self.current_version = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def load_model(self):
        client = mlflow.tracking.MlflowClient()
        try:
            versions = client.search_model_versions(f"name='{self.model_name}'")
            production_versions = [
                v for v in versions if getattr(v, "current_stage", None) == "Production"
            ]

            if not production_versions:
                logging.info("No model found in Productions.")
                return 
            
            latest = max(production_versions, key=lambda v: int(v.version))
            version = latest.version

            if version != self.current_version:
                logging.info(f"New model detected : version {version}")
                with self.lock:
                    # logged_model = 'runs:/0de33fa0279642a1b74d039e6536bc6e/model'
                    # logged_model = "models:/timeseries_xgboost_30min/Production"
                    self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/Production")
                    self.current_version = version
                    logging.info(f"Model v{version} successfully loaded")
        except Exception as e:
            logging.error(f"Error while loading model: {e}")
            
    def predict(self, consumptions: List, n_predictions: int):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        with self.lock:
            X = np.array(consumptions)
            #logging.info(f" Initail X size : {X.shape}")

            predictions = []

            for _ in range(n_predictions):
                # Prediction
                y_pred = self.model.predict(X)
                #logging.info(f"y_pred initial shape : {y_pred.shape}")

                y_pred = np.array(y_pred).reshape(-1,1)
                #logging.info(f"y_pred sise : {len(y_pred)}")

                # Prediction update
                predictions.append(y_pred)
                #logging.info(f"Prediction shape : {np.array(predictions).shape}")

                # Auto regression
                X = np.hstack([X[:, 1:], y_pred])
                #logging.info(f"X shape : {X.shape}")
            
            # Convert predictions to a 2D array (N_entries × n_predictions)
            #pred_array = np.squeeze(np.array(predictions)).T
            pred_array = np.concatenate(predictions, axis=1)
            #logging.info(f"pred_array shape : {pred_array.shape}")

            return pred_array.tolist()
    
    def start_watcher(self):
        def watch():
            while not self._stop_event.is_set():
                try:
                    self.load_model()
                except Exception as e:
                    logging.error(f"Model registry check failed: {e}")
                time.sleep(self.check_interval)
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()

    def stop_watcher(self):
        self._stop_event.set()
        logging.info("Watcher stopped.")