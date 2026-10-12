"""Rolling-origin backtest: for each country and each forecast origin, fit
every model on the preceding `train_days` and score the next `horizon` days.
Metrics: MSE, RMSE, MAE (as in the paper) + MAPE and normalised RMSE."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import all_countries, TARGET
from .models import MODELS

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, p):
    e = y - p
    mse = float(np.mean(e ** 2))
    return {"MSE": mse, "RMSE": mse ** 0.5, "MAE": float(np.mean(np.abs(e))),
            "MAPE": float(np.mean(np.abs(e) / np.maximum(y, 1e-6)) * 100),
            "nRMSE": mse ** 0.5 / max(float(np.mean(y)), 1e-6)}


def backtest(horizon=30, train_days=365, origins=("2022-03-01", "2022-06-01", "2022-09-01", "2022-12-01"),
             models=None):
    data = all_countries()
    models = models or list(MODELS)
    rows = []
    for country, d in data.items():
        for o in origins:
            o = pd.Timestamp(o)
            train = d.loc[o - pd.Timedelta(days=train_days): o - pd.Timedelta(days=1)]
            test = d.loc[o: o + pd.Timedelta(days=horizon - 1), TARGET].values
            for name in models:
                t0 = time.time()
                try:
                    pred = MODELS[name](train, len(test))
                    m = metrics(test, pred)
                except Exception as ex:  # keep the benchmark running
                    print(f"  ! {country} {o.date()} {name}: {ex}")
                    continue
                rows.append({"country": country, "origin": str(o.date()), "model": name,
                             **m, "fit_s": round(time.time() - t0, 2)})
            print(f"done {country} {o.date()}")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    by_cm = df.groupby(["country", "model"])[["RMSE", "MAE", "MSE", "MAPE", "nRMSE"]].mean()
    best = by_cm["RMSE"].groupby(level=0).idxmin().map(lambda t: t[1]).to_dict()
    overall = df.groupby("model")[["nRMSE", "MAPE"]].mean().sort_values("nRMSE")
    naive = overall.loc["Naive", "nRMSE"]
    overall["nRMSE_improvement_vs_naive_%"] = (1 - overall["nRMSE"] / naive) * 100
    return {"best_model_per_country": best,
            "overall_ranking": overall.round(3).reset_index().to_dict(orient="records"),
            "per_country_model": by_cm.round(3).reset_index().to_dict(orient="records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args()
    df = backtest(a.horizon, models=a.models)
    out = ROOT / "results"
    df.to_csv(out / "backtest_all.csv", index=False)
    s = summarise(df)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: s[k] for k in ("best_model_per_country", "overall_ranking")}, indent=2))


if __name__ == "__main__":
    main()
