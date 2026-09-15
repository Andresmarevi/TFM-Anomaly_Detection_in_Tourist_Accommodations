#!/usr/bin/env python
"""Build temporal market and anomaly metrics from archived model results."""

from pathlib import Path
import re
from datetime import date

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
HISTORY_DIR = ROOT_DIR / "data" / "results" / "history"
OUTPUT_FILE = ROOT_DIR / "data" / "results" / "temporal_evolution.csv"


def select_weekly_history_files(history_dir: Path):
    """Keep the latest historical file for each ISO week and horizon."""
    selected = {}
    pattern = re.compile(
        r"model_results_full_(?P<date>\d{4}-\d{2}-\d{2})_(?P<period>1m|3m)\.csv"
    )
    for file in history_dir.glob("model_results_full_*.csv"):
        match = pattern.fullmatch(file.name)
        if not match:
            continue
        extraction_date = date.fromisoformat(match.group("date"))
        period = match.group("period")
        week_key = (*extraction_date.isocalendar()[:2], period)
        current = selected.get(week_key)
        if current is None or extraction_date > current[0]:
            selected[week_key] = (extraction_date, file)
    return [item[1] for item in sorted(selected.values(), key=lambda item: item[0])]


def anomaly_rate(series: pd.Series) -> float:
    numeric_series = pd.to_numeric(series, errors="coerce")
    return float((numeric_series == -1).mean())


def build_temporal_summary() -> pd.DataFrame:
    files = select_weekly_history_files(HISTORY_DIR)
    if not files:
        raise FileNotFoundError(f"No hay resultados históricos en: {HISTORY_DIR}")

    frames = []
    for file in files:
        frame = pd.read_csv(file, sep=";")
        required_columns = {"extraction_date", "lookahead_period"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Faltan columnas {missing_columns} en {file.name}")
        frames.append(frame)

    history = pd.concat(frames, ignore_index=True)
    group_keys = ["extraction_date", "lookahead_period"]
    for date_column in ["checkin_date", "checkout_date"]:
        if date_column in history.columns:
            group_keys.append(date_column)

    grouped = history.groupby(
        group_keys, as_index=False, dropna=False
    )

    summary = grouped.agg(
        total_listings=("id", "size"),
        median_price=("price", "median"),
        p25_price=("price", lambda values: values.quantile(0.25)),
        p75_price=("price", lambda values: values.quantile(0.75)),
        median_price_per_sqm=("price_per_sqm", "median"),
    )

    anomaly_metrics = grouped.apply(
        lambda group: pd.Series(
            {
                "if_anomaly_rate": anomaly_rate(group["if_anomaly"]),
                "lof_anomaly_rate": anomaly_rate(group["lof_anomaly"]),
                "consensus_anomaly_rate": float(
                    (
                        (group["if_anomaly"] == -1)
                        & (group["lof_anomaly"] == -1)
                    ).mean()
                ),
                "business_rule_rate": float(
                    group["any_flag"].fillna(False).astype(bool).mean()
                )
                if "any_flag" in group
                else 0.0,
            }
        ),
        include_groups=False,
    ).reset_index(drop=True)

    anomaly_metrics[group_keys] = (
        history[group_keys]
        .drop_duplicates()
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    summary = summary.merge(
        anomaly_metrics,
        on=group_keys,
        how="left",
    )
    summary["extraction_date"] = pd.to_datetime(summary["extraction_date"])
    summary = summary.sort_values(
        group_keys
    ).reset_index(drop=True)
    summary.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8")
    print(f"Temporal summary saved: {OUTPUT_FILE}")
    return summary


if __name__ == "__main__":
    build_temporal_summary()
