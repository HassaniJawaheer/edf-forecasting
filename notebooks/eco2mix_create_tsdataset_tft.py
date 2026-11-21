import pandas as pd
import os
import pickle
import logging
from typing import List, Optional
from datetime import timedelta
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

logging.basicConfig(level=logging.INFO)

class Eco2mixCreateTSDatasetsTFT:
    """
    Creates and saves TimeSeriesDataSet objects for training, validation, and test 
    based on test cutoff and validation duration in years (can be float).
    """

    def __init__(
        self,
        df_path: str,
        output_dir: str,
        test_cutoff_years: float,
        val_duration_years: float,
        max_encoder_length: int,
        max_prediction_length: int,
        known_reals: List[str],
        unknown_reals: List[str],
        known_categoricals: Optional[List[str]] = None,
        unknown_categoricals: Optional[List[str]] = None,
        static_reals: Optional[List[str]] = None,
        static_categoricals: Optional[List[str]] = None,
        add_relative_time_idx: bool = True,
        add_target_scales: bool = True,
        add_encoder_length: bool = True,
        target_normalizer=None
    ):
        self.df_path = df_path
        self.output_dir = output_dir
        self.test_cutoff_years = test_cutoff_years
        self.val_duration_years = val_duration_years
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.known_reals = known_reals
        self.unknown_reals = unknown_reals
        self.known_categoricals = known_categoricals or []
        self.unknown_categoricals = unknown_categoricals or []
        self.static_reals = static_reals or []
        self.static_categoricals = static_categoricals or []
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.target_normalizer = target_normalizer or GroupNormalizer(groups=["series_id"])

        self.training = None
        self.validation = None
        self.test = None

    def run(self):
        df = pd.read_parquet(self.df_path)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        latest_date = df["Datetime"].max()

        test_start = latest_date - timedelta(days=round(365 * self.test_cutoff_years))
        val_start = test_start - timedelta(days=round(365 * self.val_duration_years))

        df_train = df[df["Datetime"] < val_start].copy()
        df_val = df[(df["Datetime"] >= val_start) & (df["Datetime"] < test_start)].copy()
        df_test = df[df["Datetime"] >= test_start].copy()

        logging.info(f"Latest date: {latest_date.strftime('%Y-%m-%d')}")
        logging.info(f"Validation starts: {val_start.strftime('%Y-%m-%d')}")
        logging.info(f"Test starts: {test_start.strftime('%Y-%m-%d')}")
        logging.info(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")

        self.training = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target="Consommation",
            group_ids=["series_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=self.known_reals,
            time_varying_unknown_reals=self.unknown_reals,
            time_varying_known_categoricals=self.known_categoricals,
            time_varying_unknown_categoricals=self.unknown_categoricals,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            target_normalizer=self.target_normalizer,
            add_relative_time_idx=self.add_relative_time_idx,
            add_target_scales=self.add_target_scales,
            add_encoder_length=self.add_encoder_length
        )

        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, df_val, predict=True, stop_randomization=True
        )

        self.test = TimeSeriesDataSet.from_dataset(
            self.training, df_test, predict=True, stop_randomization=True
        )

        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, "tft_training_dataset.pkl"), "wb") as f:
            pickle.dump(self.training, f)
        with open(os.path.join(self.output_dir, "tft_validation_dataset.pkl"), "wb") as f:
            pickle.dump(self.validation, f)
        with open(os.path.join(self.output_dir, "tft_test_dataset.pkl"), "wb") as f:
            pickle.dump(self.test, f)

        logging.info(f"All datasets saved to {self.output_dir}")