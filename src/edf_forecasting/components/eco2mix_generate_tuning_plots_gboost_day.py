import os
import logging
import plotly.io as pio
import optuna

from optuna.visualization import plot_optimization_history, plot_param_importances

logging.basicConfig(level=logging.INFO)

def generate_tuning_plots(study: optuna.Study, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        path1 = os.path.join(output_dir, "optimization_history.png")
        path2 = os.path.join(output_dir, "param_importances.png")

        # Sauvegarde en PNG via Plotly
        pio.write_image(fig1, path1)
        pio.write_image(fig2, path2)

        logging.info("Tuning plots saved successfully.")
        return {
            "optimization_history": path1,
            "param_importances": path2
        }

    except Exception as e:
        logging.warning(f"Failed to generate tuning plots: {e}")
        return {}

