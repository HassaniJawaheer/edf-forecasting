import os
import requests
import zipfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Eco2MixScraper:
    def __init__(self, output_dir="data/01_raw/eco2mix"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scrape_definitive_data(self, start_year: int, end_year: int):
        """Download and extract annual definitive electricity data from eco2mix."""
        base_url = "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_{}.zip"
        for year in range(start_year, end_year + 1):
            url = base_url.format(year)
            dest_dir = self.output_dir / "definitive" / str(year)
            if dest_dir.exists():
                logging.info(f"Data for year {year} already exists. Skipping.")
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            zip_path = dest_dir / f"{year}.zip"
            logging.info(f"Downloading {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                logging.info(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_dir)
                zip_path.unlink()
                logging.info(f"Done with year {year}.")
            else:
                logging.warning(f"Failed to download data for year {year}: {response.status_code}")

    def scrape_tempo_data(self, start_year: int, end_year: int):
        """Download and extract tempo tariff calendar data from eco2mix."""
        base_url = "https://eco2mix.rte-france.com/curves/downloadCalendrierTempo?season={}-{}"
        for year in range(start_year, end_year):
            start_suffix = str(year)[2:]
            end_suffix = str(year + 1)[2:]
            url = base_url.format(start_suffix, end_suffix)
            dest_dir = self.output_dir / "tempo" / f"{year}-{year+1}"
            if dest_dir.exists():
                logging.info(f"Tempo data for season {year}-{year+1} already exists. Skipping.")
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            zip_path = dest_dir / f"{year}-{year+1}.zip"
            logging.info(f"Downloading {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dest_dir)
                    logging.info(f"Extracted {zip_path} into {dest_dir}.")
                    zip_path.unlink()
                except zipfile.BadZipFile:
                    logging.warning(f"Downloaded file for {year}-{year+1} is not a zip. Keeping as is.")
            else:
                logging.warning(f"Failed to download tempo data for season {year}-{year+1}: {response.status_code}")
