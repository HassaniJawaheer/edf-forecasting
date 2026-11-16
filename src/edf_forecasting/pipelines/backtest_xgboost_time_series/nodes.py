import hashlib
import json
import logging
import platform
import subprocess
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import xgboost

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    "recursive",
    "updated_observations",
    "persistence",
    "previous_day",
    "previous_week",
)


def forecast(model, values, origins, window, horizon):
    initial = np.stack([values[t - window : t] for t in origins])
    current = initial.copy()
    recursive = np.empty((len(origins), horizon))
    updated = np.empty_like(recursive)
    for h in range(horizon):
        recursive[:, h] = np.asarray(model.predict(current)).reshape(-1)
        actual_history = np.stack([values[t + h - window : t + h] for t in origins])
        updated[:, h] = np.asarray(model.predict(actual_history)).reshape(-1)
        current = np.column_stack((current[:, 1:], recursive[:, h]))
    steps = np.arange(horizon)
    return {
        "recursive": recursive,
        "updated_observations": updated,
        "persistence": np.repeat(initial[:, -1:], horizon, axis=1),
        "previous_day": values[origins[:, None] - 48 + steps[None, :] % 48],
        "previous_week": values[origins[:, None] - 336 + steps[None, :] % 336],
    }


def prepare(history, evaluation, params):
    target = params["target"]
    if history.empty or evaluation.empty:
        raise ValueError("Empty history or evaluation data")
    if history.index.max() >= evaluation.index.min():
        raise ValueError("History must precede evaluation")
    combined = pd.concat([history[[target]], evaluation[[target]]])
    if not isinstance(combined.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex required")
    if not combined.index.is_monotonic_increasing or combined.index.has_duplicates:
        raise ValueError("Timestamps must be ordered and unique")
    if (
        not combined.index.to_series()
        .diff()
        .iloc[1:]
        .eq(pd.Timedelta(minutes=30))
        .all()
    ):
        raise ValueError("Expected continuous 30-minute timestamps")
    values = combined[target].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Consumption must contain only finite values")
    if not (evaluation.index.year == params["year"]).all():
        raise ValueError("Evaluation data must match the configured year")
    index = combined.index
    candidates = np.flatnonzero(
        (index.year == params["year"])
        & (index.hour == params["origin_hour"])
        & (index.minute == 0)
    )
    origins = candidates[candidates >= max(params["window"], 336)]
    return index, values, origins


def metrics(frame, groups):
    result = (
        frame.groupby(groups, sort=True)
        .agg(
            n=("error", "size"),
            mae_mw=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
            bias_mw=("error", "mean"),
        )
        .reset_index()
    )
    result["rmse_mw"] = np.sqrt(result.pop("mse"))
    return result


def summarize(frame):
    daily = metrics(frame, ["forecast_length", "method", "origin"])
    distribution = (
        daily.groupby(["forecast_length", "method"])
        .agg(
            origins=("origin", "size"),
            median_daily_mae_mw=("mae_mw", "median"),
            p90_daily_mae_mw=("mae_mw", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )
    overall = metrics(frame, ["forecast_length", "method"]).merge(
        distribution, on=["forecast_length", "method"]
    )
    return {
        "overall": overall,
        "by_horizon": metrics(frame, ["forecast_length", "method", "horizon"]),
        "daily": daily,
        "by_origin_month": metrics(
            frame, ["forecast_length", "method", "origin_month"]
        ),
        "by_origin_weekday": metrics(
            frame, ["forecast_length", "method", "origin_weekday"]
        ),
    }


def plot_results(frame, tables, directory):
    selected = []
    for length in sorted(frame.forecast_length.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        horizon_table = tables["by_horizon"]
        for method in METHODS:
            part = horizon_table[
                (horizon_table.forecast_length == length)
                & (horizon_table.method == method)
            ]
            for axis, metric in zip(axes, ["mae_mw", "rmse_mw"]):
                axis.plot(part.horizon / 2, part[metric], label=method)
                axis.set(xlabel="Horizon (hours)", ylabel=metric)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(directory / f"horizons_{length}.png", dpi=140)
        plt.close(fig)
        daily = tables["daily"]
        daily = daily[(daily.forecast_length == length) & (daily.method == "recursive")]
        for label, quantile in [("median", 0.5), ("p90", 0.9), ("worst", 1.0)]:
            position = (daily.mae_mw - daily.mae_mw.quantile(quantile)).abs().idxmin()
            origin = daily.loc[position, "origin"]
            selected.append(
                {
                    "forecast_length": int(length),
                    "selection": label,
                    "origin": origin,
                    "mae_mw": daily.loc[position, "mae_mw"],
                }
            )
            part = frame[(frame.forecast_length == length) & (frame.origin == origin)]
            truth = part[part.method == "recursive"]
            fig, axis = plt.subplots(figsize=(12, 4))
            axis.plot(
                truth.target_time,
                truth.actual,
                label="actual",
                color="black",
                linewidth=2,
            )
            for method in METHODS:
                series = part[part.method == method]
                axis.plot(
                    series.target_time, series.prediction, label=method, alpha=0.75
                )
            axis.set(title=f"{label} | {origin} | {length} steps", ylabel="MW")
            axis.legend(fontsize=7)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(directory / f"trajectory_{length}_{label}.png", dpi=140)
            plt.close(fig)
    return pd.DataFrame(selected)


def run_backtest(model, history, evaluation, params):
    if params["window"] != model.n_features_in_:
        raise ValueError("Window does not match model feature count")
    if params["n_jobs"] != 1:
        raise ValueError("This backtest requires n_jobs=1")
    if not params["horizons"] or any(h < 1 for h in params["horizons"]):
        raise ValueError("Positive horizons required")
    if params["max_origins"] is not None and params["max_origins"] < 1:
        raise ValueError("max_origins must be positive")
    model.set_params(n_jobs=params["n_jobs"])
    index, values, candidates = prepare(history, evaluation, params)
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("An active Kedro MLflow run is required")
    directory = Path(params["output_dir"]) / run.info.run_id
    directory.mkdir(parents=True, exist_ok=False)
    frames = []
    for length in params["horizons"]:
        origins = candidates[candidates + length <= len(values)]
        if params["max_origins"] is not None:
            origins = origins[: params["max_origins"]]
        if not len(origins):
            raise ValueError("No complete forecast origins")
        logging.info(
            "Backtest: %s origins, %s steps, one CPU thread", len(origins), length
        )
        predictions = forecast(model, values, origins, params["window"], length)
        target_positions = origins[:, None] + np.arange(length)[None, :]
        for method, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise ValueError("Non-finite predictions")
            frame = pd.DataFrame(
                {
                    "forecast_length": length,
                    "method": method,
                    "origin": np.repeat(index[origins], length),
                    "target_time": index[target_positions.ravel()],
                    "horizon": np.tile(np.arange(1, length + 1), len(origins)),
                    "actual": values[target_positions].ravel(),
                    "prediction": prediction.ravel(),
                }
            )
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["error"] = frame.prediction - frame.actual
    frame["absolute_error"] = frame.error.abs()
    frame["squared_error"] = frame.error**2
    frame["origin_month"] = frame.origin.dt.month
    frame["origin_weekday"] = frame.origin.dt.dayofweek
    tables = summarize(frame)
    frame.to_csv(directory / "predictions.csv.gz", index=False, compression="gzip")
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    examples = plot_results(frame, tables, directory)
    examples.to_csv(directory / "selected_trajectories.csv", index=False)
    metadata = {
        "run_id": run.info.run_id,
        "parameters": params,
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "timestamp_convention": "Stored timestamps; origin is the first target; horizon 1 is the next sample after the last observed value",
        "updated_observations": "Diagnostic using observations available after the initial origin",
        "files": {},
    }
    for key in ["model_path", "history_path", "evaluation_path"]:
        path = Path(params[key])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata["files"][key] = {"path": str(path), "sha256": digest}
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    sections = [
        "# Backtest",
        pd.DataFrame(
            [
                {
                    "run_id": run.info.run_id,
                    "year": params["year"],
                    "window": params["window"],
                    "n_jobs": params["n_jobs"],
                }
            ]
        ).to_markdown(index=False),
    ]
    for name in ["overall", "by_origin_month", "by_origin_weekday"]:
        sections.extend(
            [f"## {name}", tables[name].to_markdown(index=False, floatfmt=".3f")]
        )
    sections.extend(["## selected_trajectories", examples.to_markdown(index=False)])
    (directory / "results.md").write_text(
        "\n\n".join(sections) + "\n", encoding="utf-8"
    )
    for row in tables["overall"].itertuples(index=False):
        for metric in [
            "mae_mw",
            "rmse_mw",
            "bias_mw",
            "median_daily_mae_mw",
            "p90_daily_mae_mw",
        ]:
            mlflow.log_metric(
                f"backtest.{row.forecast_length}.{row.method}.{metric}",
                getattr(row, metric),
            )
    for row in tables["by_horizon"].itertuples(index=False):
        for metric in ["mae_mw", "rmse_mw", "bias_mw"]:
            mlflow.log_metric(
                f"horizon.{row.forecast_length}.{row.method}.{metric}",
                getattr(row, metric),
                step=int(row.horizon),
            )
    mlflow.log_artifacts(str(directory), artifact_path="backtest")
    for source in Path(__file__).parent.glob("*.py"):
        mlflow.log_artifact(str(source), artifact_path="backtest_source")
    for config in [
        "conf/base/parameters_backtest.yml",
        "conf/base/catalog_backtest.yml",
        "conf/base/globals.yml",
        "conf/backtest/mlflow.yml",
    ]:
        mlflow.log_artifact(config, artifact_path="backtest_source/config")
    logging.info("Backtest results: %s", directory)
