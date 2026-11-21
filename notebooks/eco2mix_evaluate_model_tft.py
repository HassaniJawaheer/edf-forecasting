import os
import yaml
import pickle
import logging
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)

class Eco2mixEvaluateModelTest:
    """
    Load a trained TFT model and a test dataset, perform predictions, compute metrics (MAE, RMSE, R2, coverage),
    and save them to evaluation_scores.yaml in the output directory.
    """

    def __init__(self, model_path, test_dataset_path, output_dir):
        self.model_path = model_path
        self.test_dataset_path = test_dataset_path
        self.output_dir = output_dir

    def run(self):
        with open(self.test_dataset_path, "rb") as f:
            test_dataset = pickle.load(f)

        test_loader = test_dataset.to_dataloader(train=False, batch_size=256, num_workers=4)

        model = TemporalFusionTransformer.load_from_checkpoint(self.model_path)

        raw_predictions = model.predict(test_loader, mode="raw", return_x=True)
        pred_median = raw_predictions.output.prediction[:, :, 3]
        true_values = raw_predictions.x["decoder_target"]

        mae = MAE()(pred_median, true_values)
        rmse = RMSE()(pred_median, true_values)
        r2 = r2_score(true_values.detach().cpu().numpy().flatten(), pred_median.detach().cpu().numpy().flatten())

        pred_5 = raw_predictions.output.prediction[:, :, 0]
        pred_95 = raw_predictions.output.prediction[:, :, 6]
        is_covered = ((true_values >= pred_5) & (true_values <= pred_95)).float()
        coverage = is_covered.mean().item() * 100
        quantile_loss = QuantileLoss()(raw_predictions.output.prediction, true_values)

        scores = {
            "MAE": float(mae.item()),
            "RMSE": float(rmse.item()),
            "R2": float(r2),
            "Coverage": float(coverage),
            "QuantileLoss": float(quantile_loss.item())
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "evaluation_scores.yaml"), "w") as f:
            yaml.dump(scores, f)

        logging.info(f"Test evaluation complete. Scores saved to {self.output_dir}/evaluation_scores.yaml")