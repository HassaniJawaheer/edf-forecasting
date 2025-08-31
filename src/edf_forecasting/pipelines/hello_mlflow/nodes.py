"""
This is a boilerplate pipeline 'hello_mlflow'
generated using Kedro 1.0.0
"""
import time, mlflow

def hello(param_message: str) -> str:
    mlflow.log_param("message", param_message)
    t0 = time.time()
    time.sleep(0.2)
    mlflow.log_metric("duration_sec", float(time.time() - t0))
    return f"Hello: {param_message}"