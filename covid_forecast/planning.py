"""Turn forecasts into what a public-health planner asks: "is a wave coming, and can I trust that?"

A curve of predicted cases is hard to act on. Planners decide things like "open the surge
ward / extend testing hours / not yet", so this module:

1. turns each 30-day forecast into a plain **trend call** for week 4 vs the last observed week:
   rising (>= +25%), falling (<= -20%) or flat (also used below 1 case per million per day);
2. backtests those calls from 46 forecast dates (every 2 weeks, Mar 2021 - Nov 2022) in all six
   countries, so each call comes with a **track record** (how often "rising" was right, how many
   real rises were warned about), compared with a no-change baseline;
3. checks whether the **80% prediction interval** is honest, i.e. contains the truth ~80% of the time;
4. writes a one-page **planning brief** per country from the latest data.

    python -m covid_forecast.planning      # -> results/planning_backtest.json, results/planning_brief.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGET, all_countries
from .models import _inv, _y

ROOT = Path(__file__).resolve().parents[1]
RISE, FALL = 1.25, 0.80          # week-4 average vs last observed week
H, TRAIN_DAYS = 30, 365
ORIGINS = pd.date_range("2021-03-01", "2022-11-30", freq="14D")


MIN_LEVEL = 1.0                  # cases per million per day; below this, % changes mean little


def trend_call(last_week: np.ndarray, week4: np.ndarray) -> str:
    if max(np.mean(last_week), np.mean(week4)) < MIN_LEVEL:
        return "flat"            # 0.1 -> 0.2 per million is +100% but not a wave
    ratio = (np.mean(week4) + 1e-9) / (np.mean(last_week) + 1e-9)
    return "rising" if ratio >= RISE else "falling" if ratio <= FALL else "flat"


def sarima_forecast(train: pd.DataFrame, h: int = H):
    """Mean forecast and 80% interval on the original scale (same model as the benchmark)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    m = SARIMAX(_y(train), order=(2, 1, 2), seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    f = m.get_forecast(h)
    ci = _inv(f.conf_int(alpha=0.2))
    return _inv(f.predicted_mean), ci[:, 0], ci[:, 1]


def backtest_calls(data=None) -> pd.DataFrame:
    data = data or all_countries()
    rows = []
    for country, d in data.items():
        for o in ORIGINS:
            train = d.loc[o - pd.Timedelta(days=TRAIN_DAYS): o - pd.Timedelta(days=1)]
            actual = d.loc[o: o + pd.Timedelta(days=H - 1), TARGET].values
            if len(train) < 300 or len(actual) < H:
                continue
            last_week = train[TARGET].values[-7:]
            mean, lo, hi = sarima_forecast(train)
            rows.append({"country": country, "origin": str(o.date()),
                         "actual": trend_call(last_week, actual[-7:]),
                         "sarima": trend_call(last_week, mean[-7:]),
                         "no_change": "flat",
                         "in_80pct_interval_share": float(np.mean((actual >= lo) & (actual <= hi))),
                         "day30_in_interval": bool(lo[-1] <= actual[-1] <= hi[-1])})
    return pd.DataFrame(rows)


def score_calls(bt: pd.DataFrame, col: str) -> dict:
    rising = bt[col] == "rising"
    real_rise = bt["actual"] == "rising"
    return {"accuracy": round(float((bt[col] == bt["actual"]).mean()), 3),
            "rises_warned_about": round(float((rising & real_rise).sum() / max(real_rise.sum(), 1)), 3),
            "rising_calls_that_were_right": round(float((rising & real_rise).sum() / rising.sum()), 3) if rising.any() else None,
            "rising_calls": int(rising.sum())}


def brief(data, summary) -> str:
    s = summary["sarima"]
    lines = ["# 30-day planning brief", "",
             f"Data up to **{max(d.index[-1] for d in data.values()).date()}**. Cases per million per day (7-day average).",
             "", "| Country | Last week | Forecast for week 4 (80% range) | Call |", "|---|---|---|---|"]
    for country, d in data.items():
        train = d.iloc[-TRAIN_DAYS:]
        last = train[TARGET].values[-7:]
        mean, lo, hi = sarima_forecast(train)
        lines.append(f"| {country} | {np.mean(last):.1f} | {np.mean(mean[-7:]):.1f} "
                     f"({np.mean(lo[-7:]):.1f} – {np.mean(hi[-7:]):.1f}) | **{trend_call(last, mean[-7:])}** |")
    lines += ["", "**How far to trust these calls** (backtest: 46 forecast dates × 6 countries, 2021-22):", "",
              f"* Calls were right **{s['accuracy']:.0%}** of the time (always saying \"flat\" scores "
              f"{summary['no_change']['accuracy']:.0%}).",
              f"* Of real rises of 25% or more, **{s['rises_warned_about']:.0%}** were called \"rising\" in advance; "
              f"**{s['rising_calls_that_were_right']:.0%}** of \"rising\" calls turned out right.",
              f"* The 80% range contained the real value on **{summary['interval_coverage_all_days']:.0%}** of days "
              f"(**{summary['interval_coverage_day30']:.0%}** on day 30)."
              + (" The range is too narrow: plan for outcomes beyond it." if summary["interval_coverage_all_days"] < 0.7 else ""),
              "", "_Cases depend on testing. Use alongside hospital and test-positivity data, not instead of it._"]
    return "\n".join(lines) + "\n"


def main():
    data = all_countries()
    bt = backtest_calls(data)
    summary = {"forecast_dates": int(bt["origin"].nunique()), "countries": int(bt["country"].nunique()),
               "calls": len(bt), "real_trend_mix": bt["actual"].value_counts().to_dict(),
               "sarima": score_calls(bt, "sarima"), "no_change": score_calls(bt, "no_change"),
               "interval_coverage_all_days": round(float(bt["in_80pct_interval_share"].mean()), 3),
               "interval_coverage_day30": round(float(bt["day30_in_interval"].mean()), 3)}
    (ROOT / "results" / "planning_backtest.json").write_text(json.dumps(summary, indent=2))
    bt.to_csv(ROOT / "results" / "planning_calls.csv", index=False)
    (ROOT / "results" / "planning_brief.md").write_text(brief(data, summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
