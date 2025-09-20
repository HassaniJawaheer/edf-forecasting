"""
This is a boilerplate pipeline 'prepare_data_30min'
generated using Kedro 1.0.0
"""
import mlflow
from edf_forecasting.components.eco2mix_add_tempo import Eco2MixAddTempo
from edf_forecasting.components.eco2mix_add_features import Eco2mixFeaturesMinute
import pandas as pd


def add_tempo_min(df_data, df_tempo, params):
    mlflow.log_param("add_tempo.mode", params["mode"])
    adder = Eco2MixAddTempo(mode=params["mode"])
    return adder.add_tempo(df_data, df_tempo)


def add_features_min(df, params):
    include = params.get("include", [])
    mlflow.log_param("features.include", ",".join(include))
    engineer = Eco2mixFeaturesMinute(df)
    df_feat = engineer.run(include=include)
    mlflow.log_metric("features.n_cols", df_feat.shape[1])
    return df_feat


def check_frequency(df_data, params):
    dt_col = params["datetime_col"]
    df = df_data.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).set_index(dt_col)

    inferred_freq = pd.infer_freq(df.index)
    mlflow.log_param("expected_freq", params["freq"])
    mlflow.log_param("inferred_freq", inferred_freq)

    if inferred_freq != params["freq"]:
        raise ValueError(f"Inconsistent time step: expected {params['freq']}, got {inferred_freq}")
    
    return df


def split_train_cal_test(df, params):
    df_train = df[df.index.year < params["cal_year"]]
    df_cal = df[df.index.year == params["cal_year"]]
    df_test = df[df.index.year == params["test_year"]]

    mlflow.log_params({
        "split.cal_year": params["cal_year"],
        "split.test_year": params["test_year"],
    })
    mlflow.log_metrics({
        "rows.train": len(df_train),
        "rows.cal": len(df_cal),
        "rows.test": len(df_test),
    })

    return df_train, df_cal, df_test
