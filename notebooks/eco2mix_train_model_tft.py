import os
import yaml
import pickle
import logging
from datetime import datetime
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO)

class Eco2mixTrainModelTFT:
    """
    Train a Temporal Fusion Transformer model using preprocessed datasets and parameters from YAML.
    Saves model, training logs, parameters used, and evaluation scores.
    """

    def __init__(self, training_dataset_path, validation_dataset_path, params_path, output_root):
        self.training_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.params_path = params_path
        self.output_root = output_root

        self.model = None
        self.trainer = None
        self.logger = None

    def run(self):
        with open(self.params_path, "r") as f:
            params = yaml.safe_load(f)

        with open(self.training_dataset_path, "rb") as f:
            training = pickle.load(f)

        with open(self.validation_dataset_path, "rb") as f:
            validation = pickle.load(f)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_root, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        # Model
        self.model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=params.get("learning_rate", 0.001),
            hidden_size=params.get("hidden_size", 128),
            attention_head_size=params.get("attention_head_size", 4),
            dropout=params.get("dropout", 0.1),
            hidden_continuous_size=params.get("hidden_continuous_size", 64),
            output_size=params.get("output_size", 7),
            loss=eval(params.get("loss", "QuantileLoss"))()
        )

        # Logger
        self.logger = TensorBoardLogger(
            save_dir=output_dir,
            name="lightning_logs_pipeline"
        )
        
        # Callbacks list
        callbacks = []

        # EarlyStopping
        if params.get("use_early_stopping", False):
            early_stop = EarlyStopping(
                monitor=params.get("early_stopping_monitor", "val_loss"),
                patience=params.get("early_stopping_patience", 5),
                mode=params.get("early_stopping_mode", "min"),
                verbose=True
            )
            callbacks.append(early_stop)

        # ModelCheckpoint (if activated)
        if params.get("use_checkpoint", False):
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(output_dir, "checkpoints"),
                filename="epoch_{epoch:02d}-val_loss_{val_loss:.2f}",
                save_top_k=-1,  # sauve tous les checkpoints
                monitor=params.get("early_stopping_monitor", "val_loss"),
                mode=params.get("early_stopping_mode", "min"),
                every_n_epochs=1
            )
            callbacks.append(checkpoint_callback)
        
        # Trainer
        self.trainer = Trainer(
            max_epochs=params.get("max_epochs", 20),
            log_every_n_steps=params.get("log_every_n_steps", 1),
            accelerator=params.get("accelerator", "gpu"),
            devices=params.get("devices", 1),
            precision=params.get("precision", "32-true"),
            gradient_clip_val=params.get("gradient_clip_val", 0.1),
            logger=self.logger,
            callbacks=callbacks
        )

        train_loader = training.to_dataloader(train=True, batch_size=params.get("batch_size", 64), num_workers=params.get("train_num_workers", 16))
        val_loader = validation.to_dataloader(train=False, batch_size=params.get("batch_size", 64) * 10, num_workers=params.get("train_num_workers", 16))

        self.trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Save model
        model_path = os.path.join(output_dir, "model.ckpt")
        self.trainer.save_checkpoint(model_path)

        # Copy params
        with open(os.path.join(output_dir, "params_used.yaml"), "w") as f:
            yaml.dump(params, f)

        # Training metadata
        training_info = {
            "model": "TemporalFusionTransformer",
            "input_data": {
                "train": self.training_dataset_path,
                "val": self.validation_dataset_path
            },
            "params_source": self.params_path,
            "runtime": timestamp,
            "seed": params.get("seed", "not_specified")
        }
        with open(os.path.join(output_dir, "training.yml"), "w") as f:
            yaml.dump(training_info, f)

        # Evaluation
        raw_predictions = self.model.predict(val_loader, mode="raw", return_x=True)
        pred_median = raw_predictions.output.prediction[:, :, 3]
        true_values = raw_predictions.x["decoder_target"]

        mae = MAE()(pred_median, true_values)
        rmse = RMSE()(pred_median, true_values)
        r2 = r2_score(true_values.detach().cpu().numpy().flatten(), pred_median.detach().cpu().numpy().flatten())

        pred_5 = raw_predictions.output.prediction[:, :, 0]
        pred_95 = raw_predictions.output.prediction[:, :, 6]
        is_covered = ((true_values >= pred_5) & (true_values <= pred_95)).float()
        coverage = is_covered.mean().item() * 100

        scores = {
            "MAE": float(mae.item()),
            "RMSE": float(rmse.item()),
            "R2": float(r2),
            "Coverage": float(coverage)
        }
        with open(os.path.join(output_dir, "scores.yaml"), "w") as f:
            yaml.dump(scores, f)
        logging.info(f"Training complete. Artifacts saved in {output_dir}")