"""
This is a boilerplate pipeline 'prepare_data'
generated using Kedro 1.0.0
"""
import mlflow
from pathlib import Path

from edf_forecasting.components.eco2mix_scraper import Eco2MixScraper
from edf_forecasting.components.eco2mix_prestructuration_data import Eco2MixDataPreparator
from edf_forecasting.components.eco2mix_clean_data import Eco2mixCleaner


def scrape_data(params):
    out_dir = Path(params["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.log_params({
        "scrape.out_dir": str(out_dir),
        "scrape.def_years": f'{params["start_year_definitive"]}-{params["end_year_definitive"]}',
        "scrape.tempo_years": f'{params["start_year_tempo"]}-{params["end_year_tempo"]}',
    })

    scraper = Eco2MixScraper(output_dir=str(out_dir))
    scraper.scrape_definitive_data(params["start_year_definitive"], params["end_year_definitive"])
    scraper.scrape_tempo_data(params["start_year_tempo"], params["end_year_tempo"])

    return {"scraping_status": "done"}


def prestructure_data(_, params):
    raw_dir = Path(params["raw_dir"])

    mlflow.log_params({
        "pre.raw_dir": str(raw_dir),

        "pre.years": f'{params["start_year"]}-{params["end_year"]}',
    })

    preparator = Eco2MixDataPreparator(str(raw_dir))
    df_consumption = preparator.prepare_consumption_data(params["start_year"], params["end_year"])
    df_tempo = preparator.prepare_tempo_calendar(params["start_year"], params["end_year"])

    mlflow.log_metric("pre.rows_consumption", len(df_consumption))
    mlflow.log_metric("pre.rows_tempo", len(df_tempo))

    return df_consumption, df_tempo


def clean_data(df_definitive, df_tempo, params):
    mlflow.log_params({
        "clean.columns_to_keep": ",".join(params["columns_to_keep"]),
        "clean.tempo_col": params["tempo_column_name"],
        "clean.new_tempo_col": params["new_tempo_column_name"],
        "clean.consumption_col": params["consumption_col"],
    })

    cleaner = Eco2mixCleaner(
        columns_to_keep=params["columns_to_keep"],
        tempo_column_name=params["tempo_column_name"],
        new_tempo_column_name=params["new_tempo_column_name"],
        consumption_col=params["consumption_col"],
    )

    df_def_cleaned = cleaner.clean_definitive(df_definitive)
    df_tempo_cleaned = cleaner.clean_tempo(df_tempo)

    mlflow.log_metric("clean.rows_def", int(len(df_def_cleaned)))
    mlflow.log_metric("clean.rows_tempo", int(len(df_tempo_cleaned)))

    return df_def_cleaned, df_tempo_cleaned
