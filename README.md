# Predicting COVID-19 Trends using Time-Series Models

[![tests](https://github.com/Drakeion2532/covid19-time-series-forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/Drakeion2532/covid19-time-series-forecasting/actions/workflows/tests.yml)

30-day forecasting of daily COVID-19 cases across **6 countries** (India, United States, Australia, South Korea, United Kingdom, Brazil). The benchmark compares ARIMA, SARIMA, SARIMAX (with policy and vaccination features), Holt-Winters, Prophet and an ensemble against naive baselines, using a rolling-origin backtest.

Reference implementation of the research paper **"Predicting COVID-19 Trends using Time Series Models"**: H. T. Chaudhari, A. G. Jadhav, I. S. Pathare, **S. S. Punde**. The work was done during the research internship at the **University of Nevada, Las Vegas (UNLV)**, Summer 2024.

![holdout](results/figures/holdout_forecasts.png)

## Why it matters

Hospitals and public-health teams need to know **how many cases to expect over the next weeks** to plan beds, staff and testing. This project shows which classical forecasting models hold up across countries with very different epidemic curves and policies. It also shows where they fail.

## Pipeline

1. **Data**: Our World in Data COVID-19 dataset (aggregates WHO + JHU). Subset at `data/raw/owid_covid_6_countries.csv`.
2. **Cleaning** (as in the paper): outliers flagged by both a rolling z-score (|z| > 3) and a rolling IQR rule are treated as missing, then forward-filled and back-filled.
3. **Target**: `new_cases_smoothed_per_million`, modelled on a log scale.
4. **Exogenous features** (SARIMAX): Oxford stringency index and % of people vaccinated. At forecast time only the last observed value is used, so there is **no look-ahead leakage**.
5. **Evaluation**: rolling-origin backtest with 4 forecast origins in 2022 × 6 countries × 30-day horizon. Each model is refit on the preceding 365 days. Metrics: **MSE, RMSE, MAE** (the paper's metrics) plus MAPE and normalised RMSE.

## Results

Mean over 6 countries × 4 origins (lower is better). The improvement column is measured against the naive "tomorrow = today" forecast.

| Model | nRMSE | MAPE | vs. Naive |
|---|---|---|---|
| **Ensemble** (median of SARIMA, SARIMAX, Holt-Winters) | **0.404** | 36.6% | **+25.0%** |
| SARIMA (2,1,2)(1,0,1)₇ | 0.406 | 36.7% | +24.6% |
| Holt-Winters (damped, weekly) | 0.407 | **34.0%** | +24.5% |
| SARIMAX (+ stringency, vaccination) | 0.475 | 43.8% | +11.9% |
| Naive | 0.539 | 53.9% | – |
| ARIMA (auto order) | 0.628 | 61.1% | −16.6% |
| Seasonal-Naive (same day last week) | 0.738 | 78.6% | −36.9% |
| Prophet | 2.131 | 245.6% | −295% |

**Best model per country (RMSE):** India: SARIMA · South Korea: SARIMA · United States: SARIMAX · Brazil: SARIMAX · Australia: Holt-Winters · United Kingdom: Seasonal-Naive

![heatmap](results/figures/nrmse_heatmap.png)

### Findings

* **Weekly seasonality matters.** SARIMA beats ARIMA by about 35% nRMSE because reporting follows a 7-day cycle. (In the paper SARIMA was the best model for the United States; in this backtest SARIMAX edges it out there, as described below.)
* **Exogenous features help where policy was still changing.** SARIMAX wins for the US and Brazil, but by late 2022 most stringency indices were flat, so the extra features added little elsewhere.
* **Prophet is a poor fit for 30-day epidemic forecasting.** Its piecewise-linear trend extrapolates the slope of the last wave and overshoots badly. This is a known limitation worth knowing before using it in production.
* **The ensemble is the safest choice.** It is never the worst model and is best on average, though only just ahead of SARIMA (0.404 vs 0.406). Its three members were chosen after seeing this backtest, so its lead is slightly optimistic; a stricter test would pick members on 2021 origins and score them on 2022.
* Full per-country, per-origin numbers are in `results/backtest_all.csv` and `results/summary.json`.

### From forecast to planning decision (`covid_forecast/planning.py`)

A curve of predicted cases is hard to act on. A public-health planner asks: **is a wave coming in the next month, and how far can I trust that call?** `python -m covid_forecast.planning` answers that:

1. **The call:** each 30-day SARIMA forecast becomes **rising** (week 4 at least 25% above the last observed week), **falling** (at least 20% below) or **flat**. Levels under 1 case per million per day are always "flat", because a move from 0.1 to 0.2 isn't a wave.
2. **The track record:** those calls were backtested from **46 forecast dates** (every 2 weeks, Mar 2021 – Nov 2022) in all six countries, 276 calls in total:

| | SARIMA calls | Always "no change" |
|---|---|---|
| Calls that were right | **59%** | 24% |
| Real rises (≥ 25%) warned about in advance | **56%** | 0% |
| "Rising" calls that turned out right | **69%** | – |

3. **The interval check:** an 80% range should contain the truth about 80% of the time. Here it did on **86%** of days, and on **76%** of day-30 values. The ranges are honest, if slightly wide early on.
4. **The brief:** a one-page planning brief per country from the latest data goes to `results/planning_brief.md`, with each call shown next to its track record.

In plain terms: when the model warns of a rise, it's right about 2 times in 3, and it misses roughly 2 rises in 5. That's useful as an early-warning input, not as the only one. The brief says so, and recommends using it alongside hospital and test-positivity data.

### 30-day projection

`results/forward_forecast_30d.json` holds a 30-day projection beyond the end of the daily data, with 80% prediction intervals.

![forward](results/figures/forward_forecast.png)

![eda](results/figures/eda_cases_vs_stringency.png)

## Run it

```bash
pip install -r requirements.txt
python -m covid_forecast.evaluate   # backtest -> results/backtest_all.csv, summary.json (~1-2 min)
python -m covid_forecast.report     # figures + 30-day forward forecast
pytest -q
```

To refresh the data from the full OWID file:

```bash
curl -L -o owid.csv https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
```

Then filter it to the countries you want (see `covid_forecast/data.py`).

## Notes and limitations

* Daily data with stringency coverage runs from **2020-03-01 to 2022-12-31**. After 2022 several countries switched to weekly reporting, which is visible as the step shape in late-2022 curves.
* Case counts depend on testing volume. A drop in reported cases may reflect less testing rather than less infection.

## Structure

```
covid_forecast/
  data.py       loading, outlier handling (z-score + IQR), ffill/bfill
  models.py     Naive, Seasonal-Naive, ARIMA, SARIMA, SARIMAX, Holt-Winters, Prophet, Ensemble
  evaluate.py   rolling-origin backtest + metrics
  report.py     figures + forward forecast
  planning.py   trend calls, their backtested track record, interval check, planning brief
data/raw/       6-country dataset (OWID, CC-BY 4.0)
results/        metrics, forecasts, figures
tests/
```

Data: Mathieu, E. et al. *Coronavirus Pandemic (COVID-19)*, Our World in Data (CC BY 4.0).
