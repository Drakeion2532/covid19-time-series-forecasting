"""Loading and cleaning – follows the paper's preprocessing:
outliers detected with IQR / z-score are treated as missing values, then
filled with forward-fill followed by backward-fill."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "owid_covid_6_countries.csv"
COUNTRIES = ["India", "United States", "Australia", "South Korea", "United Kingdom", "Brazil"]
TARGET = "new_cases_smoothed_per_million"
EXOG = ["stringency_index", "people_vaccinated_per_hundred"]
START, END = "2020-03-01", "2022-12-31"  # daily reporting + stringency coverage


def load(path: str | Path = DATA) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def zscore_outliers(s: pd.Series, z: float = 3.0, window: int = 28) -> pd.Series:
    """Rolling z-score so legitimate wave peaks are not flagged as outliers."""
    m = s.rolling(window, center=True, min_periods=7).mean()
    sd = s.rolling(window, center=True, min_periods=7).std()
    return ((s - m).abs() / sd) > z


def iqr_outliers(s: pd.Series, k: float = 1.5, window: int = 28) -> pd.Series:
    q1 = s.rolling(window, center=True, min_periods=7).quantile(0.25)
    q3 = s.rolling(window, center=True, min_periods=7).quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def country_series(df: pd.DataFrame, country: str, clean: bool = True) -> pd.DataFrame:
    d = (df[df.location == country].set_index("date").sort_index()
         .loc[START:END, [TARGET, *EXOG]].asfreq("D"))
    d[EXOG] = d[EXOG].ffill().bfill().fillna(0)
    y = d[TARGET].clip(lower=0)
    if clean:
        # flag only points that are outliers under BOTH rules (conservative)
        bad = zscore_outliers(y) & iqr_outliers(y)
        y = y.mask(bad)
    d[TARGET] = y.ffill().bfill()
    d.attrs["n_outliers"] = int(bad.sum()) if clean else 0
    return d


def all_countries(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    df = load() if df is None else df
    return {c: country_series(df, c) for c in COUNTRIES}
