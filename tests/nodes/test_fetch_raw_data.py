from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
from edf_forecasting.pipelines.fetch_raw_data.nodes import (
    scrape_data,
    prestructure_data,
    clean_data
)

def test_scrape_data_calls_scraper_and_returns_status(tmp_path):
    params = {
        "output_dir": tmp_path,
        "start_year_definitive": 2018,
        "end_year_definitive": 2019,
        "start_year_tempo": 2020,
        "end_year_tempo": 2021
    }

    # Mocking Eco2MixScraper and mlflow
    with patch("edf_forecasting.components.eco2mix_scraper.Eco2MixScraper") as MockScraper, patch("edf_forecasting.pipelines.fetch_raw_data.nodes.mlflow"):
        
        scraper_instance = MagicMock()
        MockScraper.return_value = scraper_instance

        result = scrape_data(params)

        # Test que le scaper est bien instancié
        MockScraper.assert_called_once_with(output_dir=str(Path(tmp_path)))

        # Test les apples les 2 méthodes de la classe
        scraper_instance.scrape_definitive_data.assert_called_once_with(2018, 2019)
        scraper_instance.scrape_tempo_data.assert_called_once_with(2020, 2021)

        # Contrat
        assert result == {"scraping_status": "done"}

def test_prestructure_data_calls_preparator_and_returns_dataframes():

    params = {
        "raw_dir": "/fake/raw",
        "start_year": 2019,
        'end_year': 2020
    }

    fake_df_consumption = pd.DataFrame({"a": [1,2,3]})
    fake_df_tempo = pd.DataFrame({"b": [4,5]})

    with patch("edf_forcasting.components.eco2mix_prestructuration_data.Eco2MixDataPreparator") as MockPreparator, \
        patch("edf_forecasting.pipelines.fetch_raw_data.nodes.mlflow"):
        
        preparator_instance = MagicMock()
        preparator_instance.prepare_consumption_data.return_value = fake_df_consumption
        preparator_instance.prepare_tempo_calendar.return_value = fake_df_tempo
        MockPreparator.return_value = preparator_instance

        df_consumption, df_tempo = prestructure_data(None, params)

        MockPreparator.assert_called_once_with(str("/fake/raw"))

        preparator_instance.prepare_consumption_data.assert_called_once_with(2019, 2020)
        preparator_instance.prepare_tempo_calendar.assert_called_once_with(2019, 2020)

        assert isinstance(df_consumption, pd.DataFrame)
        assert isinstance(df_tempo, pd.DataFrame)

def test_clean_data_calls_cleaner_and_returns_cleaned_dfs():
    params = {
        "columns_to_keep": ["date", "consumption"],
        "tempo_column_name": "tempo",
        "new_tempo_column_name": "tempo_clean",
        "consumption_col": "consumption",
    }

    df_def = pd.DataFrame({"consumption": [1, 2]})
    df_tempo = pd.DataFrame({"tempo": ["A", "B"]})

    df_def_cleaned = pd.DataFrame({"consumption": [1, 2]})
    df_tempo_cleaned = pd.DataFrame({"tempo_clean": ["A", "B"]})

    with patch("edf_forecasting.components.eco2mix_clean_data.Eco2mixCleaner") as MockCleaner, \
        patch("edf_forecasting.pipelines.fetch_raw_data.nodes.mlflow"):

        cleaner_instance = MagicMock()
        cleaner_instance.clean_definitive.return_value = df_def_cleaned
        cleaner_instance.clean_tempo.return_value = df_tempo_cleaned
        MockCleaner.return_value = cleaner_instance

        out_def, out_tempo = clean_data(df_def, df_tempo, params)

        MockCleaner.assert_called_once_with(
            columns_to_keep=["date", "consumption"],
            tempo_column_name="tempo",
            new_tempo_column_name="tempo_clean",
            consumption_col="consumption",
        )

        cleaner_instance.clean_definitive.assert_called_once_with(df_def)
        cleaner_instance.clean_tempo.assert_called_once_with(df_tempo)

        assert isinstance(out_def, pd.DataFrame)
        assert isinstance(out_tempo, pd.DataFrame)
