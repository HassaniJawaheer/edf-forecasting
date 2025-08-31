import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

def generate_prediction_plots(model: BaseEstimator, X_test, y_test, output_dir: str, n_days: int = 3, random_seed: int = None):
    if random_seed is not None:
        np.random.seed(random_seed)

    os.makedirs(output_dir, exist_ok=True)

    indexes = np.random.choice(X_test.shape[0], size=n_days, replace=False)
    y_pred = model.predict(X_test.iloc[indexes])
    y_true = y_test.iloc[indexes]

    for i, idx in enumerate(indexes):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true.iloc[i], label="True", linewidth=2)
        plt.plot(y_pred[i], label="Predicted", linestyle='--')
        plt.title(f"Prediction vs True - Sample {i+1}")
        plt.xlabel("30-minute intervals")
        plt.ylabel("Consumption")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"plot_{i+1}.png")
        plt.savefig(fig_path)
        plt.close()
