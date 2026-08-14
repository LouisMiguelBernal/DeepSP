# DeepS&P

S&P 500 forecasting platform — a three-layer LSTM trained on ninety years of daily
closes, paired with a Monte Carlo engine for forward path simulation.

---

## Run it locally

Double-click `run.bat`, or from a terminal in the project folder:

```bash
run.bat
```

First run creates a virtual environment and installs PyTorch (CPU build, ~200 MB),
so give it a few minutes. After that it launches straight to
**http://localhost:8502**.

Prefer to drive it yourself:

```bash
python -m venv .venv && .venv\Scripts\activate && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && streamlit run sp500.py
```

The first page load takes roughly a minute — importing PyTorch and unpickling the
scaler is slow, and both are cached afterwards.

---

## What's in it

**Prediction** — pick any session between 1930 and 2020. The model sees only the
closes that preceded it and predicts that day's close, so the error is a genuine
out-of-sample read on a single bar. Shown against the full price history on a log
axis, with the selected session marked.

**Monte Carlo** — draws daily returns from the historical mean and standard
deviation and compounds them forward. Fan chart with a 5–95% band and a median
path, plus the terminal price distribution. Seeded, so a run is reproducible.

**Analytics** — all-time highs and lows, average and annualised returns, rolling
30-session volatility, peak-to-trough drawdown, and a filterable data explorer
with CSV export.

**Model** — architecture, parameter count, training details, and an honest list of
what the model cannot do.

---

## The model

| | |
|---|---|
| Type | Stacked LSTM |
| Hidden size | 256 |
| Layers | 3 |
| Dropout | 0.20 |
| Input | 1 feature (close price) |
| Head | Linear → 1 |
| Scaling | MinMax |
| Training period | 1930 – 2020 |
| Framework | PyTorch |

It is univariate and single-step: close price in, one bar ahead out. It does not
produce a path, and chaining its own outputs would compound error quickly. It was
trained through 2020, so the distribution it learned may not describe the market
you are looking at.

Research and educational use only. Nothing here is investment advice.

---

## Layout

```
DeepSP/
├── .streamlit/config.toml       Base theme
├── assets/
│   ├── SPX.csv                  Daily OHLCV, 1930–2020
│   ├── lstm_spx_gpu_safe.pth    Trained weights
│   └── scaler_spx_gpu_safe.save Fitted MinMax scaler
├── sp500.py                     The application
├── sp_model.ipynb               Training notebook
├── theme.py                     Shared design system
├── requirements.txt
└── run.bat                      One-command local launch
```

`theme.py` is shared verbatim with [QuantMaven](https://github.com/LouisMiguelBernal/QuantMaven)
and [GiftxAI](https://github.com/LouisMiguelBernal/GiftxAI) — one visual language,
one accent hue per project.

---

## Known wrinkle

The scaler was pickled under scikit-learn 1.5 and will emit an
`InconsistentVersionWarning` on newer versions. `MinMaxScaler` stores only the
fitted min and scale arrays and transforms identically across these versions, so
the warning is safe to ignore. Re-fit and re-save it from `sp_model.ipynb` if you
would rather it went away.
