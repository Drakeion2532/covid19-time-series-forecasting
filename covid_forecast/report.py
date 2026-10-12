"""Generates all figures + the 30-day forward forecast used in the README."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import all_countries, TARGET, COUNTRIES
from .models import MODELS, _y, _inv

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def eda(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, (c, d) in zip(axes.flat, data.items()):
        ax.plot(d.index, d[TARGET], color="tab:red", lw=1)
        ax.set_title(c); ax.set_ylabel("new cases / million (7-d avg)")
        ax2 = ax.twinx(); ax2.plot(d.index, d["stringency_index"], color="tab:blue", lw=1, alpha=.6)
        ax2.set_ylim(0, 100); ax2.set_ylabel("stringency", color="tab:blue")
    fig.suptitle("Daily cases vs. government stringency (OWID / WHO data)")
    fig.tight_layout(); fig.savefig(FIG / "eda_cases_vs_stringency.png", dpi=110); plt.close(fig)


def heatmap(bt):
    piv = bt.groupby(["country", "model"])["nRMSE"].mean().unstack()
    order = bt.groupby("model")["nRMSE"].mean().sort_values().index
    piv = piv[order].clip(upper=2)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="normalised RMSE (lower = better, capped at 2)")
    ax.set_title("30-day forecast error, mean over 4 rolling origins (2022)")
    fig.tight_layout(); fig.savefig(FIG / "nrmse_heatmap.png", dpi=110); plt.close(fig)


def backtest_plot(data, origin="2022-12-01", h=30, show=("Naive", "SARIMA", "Holt-Winters", "SARIMAX", "Ensemble")):
    o = pd.Timestamp(origin)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (c, d) in zip(axes.flat, data.items()):
        train = d.loc[o - pd.Timedelta(days=365): o - pd.Timedelta(days=1)]
        test = d.loc[o: o + pd.Timedelta(days=h - 1), TARGET]
        ax.plot(train.index[-90:], train[TARGET].iloc[-90:], color="k", lw=1.2, label="history")
        ax.plot(test.index, test.values, color="k", ls="--", lw=1.2, label="actual")
        for m in show:
            ax.plot(test.index, MODELS[m](train, len(test)), lw=1, label=m)
        ax.set_title(c); ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Hold-out forecasts from {origin} (30 days)")
    fig.tight_layout(); fig.savefig(FIG / "holdout_forecasts.png", dpi=110); plt.close(fig)


def forward_forecast(data, h=30):
    """30-day projection beyond the data with 80% prediction intervals (SARIMA)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    out, fig, axes = {}, *plt.subplots(2, 3, figsize=(15, 7))
    for ax, (c, d) in zip(axes.flat, data.items()):
        train = d.iloc[-365:]
        m = SARIMAX(_y(train), order=(2, 1, 2), seasonal_order=(1, 0, 1, 7),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        f = m.get_forecast(h)
        mean, ci = _inv(f.predicted_mean), _inv(f.conf_int(alpha=0.2))
        idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h)
        out[c] = pd.DataFrame({"date": idx.date.astype(str), "forecast": mean.round(2),
                               "lo80": ci[:, 0].round(2), "hi80": ci[:, 1].round(2)}).to_dict("records")
        ax.plot(train.index[-120:], train[TARGET].iloc[-120:], color="k", lw=1)
        ax.plot(idx, mean, color="tab:green"); ax.fill_between(idx, ci[:, 0], ci[:, 1], color="tab:green", alpha=.2)
        ax.set_title(c); ax.tick_params(axis="x", rotation=30)
    fig.suptitle("30-day projection (SARIMA, 80% interval)")
    fig.tight_layout(); fig.savefig(FIG / "forward_forecast.png", dpi=110); plt.close(fig)
    (ROOT / "results" / "forward_forecast_30d.json").write_text(json.dumps(out, indent=1))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = all_countries()
    eda(data)
    heatmap(pd.read_csv(ROOT / "results" / "backtest_all.csv"))
    backtest_plot(data)
    forward_forecast(data)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
