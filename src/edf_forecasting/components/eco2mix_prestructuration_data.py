import os
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def read_df(path):
    """Read a data file as CSV with tab separator and latin1 encoding."""
    return pd.read_csv(path, sep="\t", encoding="latin1", index_col=False, low_memory=False)

class Eco2MixDataPreparator:
    """Prepare and merge eco2mix definitive and tempo data into structured CSVs."""

    def __init__(self, raw_dir, output_dir):
        """Initialize preparator with raw and output directories."""
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_consumption_data(self, start_year: int, end_year: int, output_filename="consumption_data.csv"):
        """Load, align, and merge consumption data between start and end years."""
        definitive_dir = self.raw_dir / "definitive"
        year_folders = sorted([p for p in definitive_dir.iterdir() if p.is_dir()])

        baseline_columns = None
        all_dfs = []

        for year_folder in year_folders:
            year = int(year_folder.name)
            if year < start_year or year > end_year:
                continue

            for file in year_folder.glob("*.xls"):
                try:
                    df = read_df(file)
                    df = df.iloc[:-1]
                    if baseline_columns is None:
                        baseline_columns = list(df.columns)
                        logging.info(f"Baseline columns from {file.name} ({len(baseline_columns)} columns)")
                    df = df[[col for col in baseline_columns if col in df.columns]]
                    all_dfs.append(df)
                    logging.info(f"Loaded {file.name} with {df.shape[0]} rows.")
                except Exception as e:
                    logging.error(f"Failed loading {file.name}: {e}")

        if not all_dfs:
            logging.error("No consumption data found.")
            return

        df_all = pd.concat(all_dfs, ignore_index=True)
        if "Date" in df_all.columns and "Heures" in df_all.columns:
            df_all["Datetime"] = pd.to_datetime(df_all["Date"].astype(str) + " " + df_all["Heures"], errors="coerce")
            df_all = df_all.drop(columns=["Date", "Heures"])
            df_all = df_all.sort_values("Datetime").reset_index(drop=True)

        definitive_output_dir = self.output_dir / "definitive"
        definitive_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = definitive_output_dir / output_filename

        # Remove existing file if it exists
        if output_path.exists():
            output_path.unlink()

        df_all.to_csv(output_path, index=False)
        logging.info(f"Consumption data saved to {output_path}.")

    def prepare_tempo_calendar(self, start_year: int, end_year: int, output_filename="tempo_calendar.csv"):
        """Load and merge tempo data into a cleaned CSV between given years."""
        tempo_dir = self.raw_dir / "tempo"
        tempo_folders = sorted([p for p in tempo_dir.iterdir() if p.is_dir()])

        all_dfs = []

        for season_folder in tempo_folders:
            for file in season_folder.glob("*.xls"):
                try:
                    df = read_df(file)
                    df = df.iloc[:-1]
                    if "Date" not in df.columns:
                        logging.warning(f"No Date column in {file.name}, skipping.")
                        continue
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    all_dfs.append(df)
                    logging.info(f"Loaded {file.name} with {df.shape[0]} rows.")
                except Exception as e:
                    logging.error(f"Failed loading {file.name}: {e}")

        if not all_dfs:
            logging.error("No tempo data found.")
            return

        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all = df_all[df_all["Date"].between(f"{start_year}-01-01", f"{end_year}-12-31")]
        df_all = df_all.sort_values("Date").reset_index(drop=True)

        tempo_output_dir = self.output_dir / "tempo"
        tempo_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = tempo_output_dir / output_filename

        # Remove existing file if it exists
        if output_path.exists():
            output_path.unlink()

        df_all.to_csv(output_path, index=False)
        logging.info(f"Tempo calendar saved to {output_path}.")
