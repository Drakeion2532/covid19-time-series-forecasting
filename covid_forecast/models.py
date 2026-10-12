"""Forecasting models. Each takes a training frame and a horizon and
returns a numpy array of `horizon` forecasts on the original scale.

Models are fit on log1p(cases per million) – case counts are multiplicative
and strictly positive, so the log scale stabilises variance."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .data import TARGET, EXOG

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def _y(train):
    return np.log1p(train[TARGET].values)


def _inv(x):
    return np.clip(np.expm1(np.asarray(x)), 0, None)


def naive(train, h):
    """Baseline: tomorrow = today. Any useful model must beat this."""
    return np.repeat(train[TARGET].values[-1], h)


def seasonal_naive(train, h):
    last_week = train[TARGET].values[-7:]
    return np.resize(last_week, h)


def arima(train, h):
    import pmdarima as pm
    m = pm.auto_arima(_y(train), seasonal=False, stepwise=True, max_p=5, max_q=5,
                      suppress_warnings=True, error_action="ignore")
    return _inv(m.predict(h))


def sarima(train, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    m = SARIMAX(_y(train), order=(2, 1, 2), seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return _inv(m.forecast(h))


def sarimax(train, h):
    """SARIMA + exogenous policy/vaccination features. Future exogenous values
    are not known at forecast time, so the last observed value is carried
    forward (no look-ahead leakage)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    X = train[EXOG].values
    Xf = np.repeat(X[-1:], h, axis=0)
    m = SARIMAX(_y(train), exog=X, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return _inv(m.forecast(h, exog=Xf))


def holt_winters(train, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    m = ExponentialSmoothing(_y(train), trend="add", damped_trend=True,
                             seasonal="add", seasonal_periods=7).fit()
    return _inv(m.forecast(h))


def prophet(train, h):
    from prophet import Prophet
    train = train.iloc[-120:]  # local trend only: older waves mislead the trend
    df = pd.DataFrame({"ds": train.index, "y": _y(train)})
    # a conservative trend prior: epidemic curves turn quickly, and an
    # aggressive trend extrapolates explosive growth/decay over 30 days
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=0.01, changepoint_range=0.95)
    m.fit(df)
    fut = m.make_future_dataframe(h, include_history=False)
    return _inv(m.predict(fut)["yhat"].values)


def ensemble(train, h):
    """Median of the three strongest classical models – robust to any one
    model mis-reading the trend."""
    preds = np.vstack([sarima(train, h), sarimax(train, h), holt_winters(train, h)])
    return np.median(preds, axis=0)


MODELS = {
    "Naive": naive,
    "Seasonal-Naive": seasonal_naive,
    "ARIMA": arima,
    "SARIMA": sarima,
    "SARIMAX": sarimax,
    "Holt-Winters": holt_winters,
    "Prophet": prophet,
    "Ensemble": ensemble,
}
