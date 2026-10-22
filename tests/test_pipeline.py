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


# ---------------------------------------------------------------- planning brief
import numpy as np
import pandas as pd


def test_trend_calls():
    from covid_forecast.planning import trend_call
    w = np.full(7, 100.0)
    assert trend_call(w, w * 1.3) == "rising"
    assert trend_call(w, w * 0.7) == "falling"
    assert trend_call(w, w * 1.1) == "flat"
    assert trend_call(np.full(7, 0.1), np.full(7, 0.3)) == "flat"     # near zero: % change is not a wave


def test_backtest_calls_and_scoring(monkeypatch):
    import covid_forecast.planning as pl
    from covid_forecast.data import all_countries
    monkeypatch.setattr(pl, "ORIGINS", pd.date_range("2022-01-03", periods=3, freq="28D"))
    data = {k: v for k, v in all_countries().items() if k == "India"}
    bt = pl.backtest_calls(data)
    assert len(bt) == 3 and set(bt["actual"]) <= {"rising", "falling", "flat"}
    assert bt["in_80pct_interval_share"].between(0, 1).all()
    s = pl.score_calls(bt, "sarima")
    assert 0 <= s["accuracy"] <= 1 and s["rising_calls"] >= 0
