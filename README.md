# DeepS&P

S&P 500 forecasting dashboard: a three-layer PyTorch LSTM trained on ninety years of
daily closes, a Monte Carlo engine for forward path simulation, and market analytics.

Version 2 is a Next.js app. The model ran once in PyTorch and its predictions ship as
static JSON, so the site needs no Python server and every interaction runs in the browser.

---

## Run it locally

```bash
npm install
npm run dev        # http://localhost:3200
```

Production build: `npm run build && npm start`. Tests: `npm test`. Types: `npm run typecheck`.

Deploy: import the repo in Vercel. No environment variables are needed.

---

## What's in it

| Tab | What it does |
|---|---|
| **Prediction** | Pick any session from 1928 to 2020. The model saw only the closes before it; you get its prediction, the actual close and the error. Price history on a log axis with 50/200-session averages, range presets and the model line on the 5Y/1Y views. |
| **Monte Carlo** | Seeded geometric random walk on daily returns drawn from the historical mean and standard deviation. Up to 2,000 paths × 365 sessions, fan chart with a 5–95% band, terminal distribution, share of paths ending below the start. |
| **Analytics** | All-time high/low, average and annualised return, 30-session volatility, max drawdown, best/worst day, drawdown and volatility charts, and a data explorer with CSV export that includes the model's prediction per session. |
| **Model** | Architecture, parameter count, measured accuracy and the limits of the model. |

---

## The model

| | |
|---|---|
| Type | Stacked LSTM, 3 layers × 256 hidden, dropout 0.20 |
| Input | 1 feature (close), min-max scaled |
| Head | Linear → 1 |
| Parameters | 1,318,145 |
| Data | 23,323 sessions, Dec 1927 – Nov 2020 |
| Framework | PyTorch |

Measured on 2016–2020 (`model/export_predictions.py` prints these):

| Metric | Value |
|---|---|
| MAPE | 0.83% |
| Median absolute % error | 0.53% |
| Direction hit rate | 50.2% |

The low error mostly tracks the previous close; direction is a coin flip. It is a
univariate, single-step model trained through 2020. Research and education only,
not investment advice.

---

## Regenerate the predictions

```bash
pip install torch joblib scikit-learn pandas numpy
python model/export_predictions.py     # writes public/data/spx.json
```

The network is causal, so one forward pass over the whole series yields its output after
every prefix, which is the same number the per-date pass in v1 produced.

---

## Layout

```
DeepSP/
├── model/
│   ├── assets/                  SPX.csv, trained weights, fitted scaler
│   ├── export_predictions.py    PyTorch → public/data/spx.json
│   └── sp_model.ipynb           Training notebook
├── public/data/spx.json         Closes, volume and predictions (0.8 MB)
├── src/app/                     Next.js App Router
├── src/components/              Dashboard and the SVG chart
└── src/lib/                     Series maths, Monte Carlo, tests
```
