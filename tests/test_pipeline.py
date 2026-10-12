import numpy as np
import pandas as pd

from covid_forecast.data import load, country_series, COUNTRIES, TARGET
from covid_forecast.evaluate import metrics
from covid_forecast.models import MODELS


def test_dataset_has_all_countries():
    df = load()
    assert set(COUNTRIES) <= set(df.location.unique())


def test_cleaning_produces_continuous_daily_series():
    d = country_series(load(), "India")
    assert d.index.freq == "D"
    assert d[TARGET].isna().sum() == 0
    assert (d[TARGET] >= 0).all()


def test_metrics():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert np.isclose(m["MSE"], 4 / 3)
    assert np.isclose(m["MAE"], 2 / 3)
    assert np.isclose(m["RMSE"], (4 / 3) ** 0.5)


def test_fast_models_return_horizon_and_no_negatives():
    d = country_series(load(), "South Korea").iloc[-200:]
    for name in ["Naive", "Seasonal-Naive", "SARIMA", "Holt-Winters"]:
        p = MODELS[name](d, 30)
        assert len(p) == 30 and np.all(p >= 0) and np.all(np.isfinite(p)), name
