import pandas as pd
import os
import logging
from typing import List

logging.basicConfig(level=logging.INFO)

class Eco2mixPrepareTFTData:
    """
    Prepares Eco2mix data for TFT: drops columns, adds time_idx and series_id,
    and saves the full dataset to disk.
    """

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        series_id_value: str,
        columns_to_drop: List[str]
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.series_id_value = series_id_value
        self.columns_to_drop = columns_to_drop

    def run(self):
        df = pd.read_csv(self.input_path)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)

        df["time_idx"] = range(len(df))
        df["series_id"] = self.series_id_value

        df.drop(columns=self.columns_to_drop, inplace=True, errors="ignore")

        os.makedirs(self.output_dir, exist_ok=True)
        df.to_parquet(os.path.join(self.output_dir, "df.parquet"))

        logging.info(f"Dataset saved to {self.output_dir} with {len(df)} rows.")
       