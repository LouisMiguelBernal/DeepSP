"""
Export the LSTM's one-step-ahead predictions for every session to JSON.

The network is causal, so one forward pass over the whole close series gives the
output after each prefix: out[t] is the model's view after seeing closes[0..t],
which is its prediction for session t+1. That is identical to the Streamlit app's
per-date pass, computed once instead of on every click.

Run from the repo root:  python model/export_predictions.py
Writes public/data/spx.json for the Next.js app.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUT = ROOT.parent / "public" / "data" / "spx.json"


class StockLSTM(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)  # every step, not only the last


def main() -> None:
    df = pd.read_csv(ASSETS / "SPX.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    scaler = joblib.load(ASSETS / "scaler_spx_gpu_safe.save")
    model = StockLSTM()
    model.load_state_dict(torch.load(ASSETS / "lstm_spx_gpu_safe.pth", map_location="cpu"))
    model.eval()

    closes = df["Close"].to_numpy().reshape(-1, 1)
    x = torch.tensor(scaler.transform(closes), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x).squeeze(0).numpy()  # (n, 1)
    nxt = scaler.inverse_transform(out).ravel()  # nxt[t] = prediction for t+1

    pred = np.full(len(df), np.nan)
    pred[1:] = nxt[:-1]

    params = sum(p.numel() for p in model.parameters())
    payload = {
        "meta": {
            "index": "S&P 500 (SPX)",
            "first": df["Date"].iloc[0].strftime("%Y-%m-%d"),
            "last": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "sessions": int(len(df)),
            "parameters": int(params),
            "torch": torch.__version__.split("+")[0],
        },
        "d": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "c": [round(float(v), 2) for v in df["Close"]],
        "v": [int(v) for v in df["Volume"]],
        "p": [None if np.isnan(v) else round(float(v), 2) for v in pred],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))

    # Quick read-out on the test-like tail so the numbers on the site are measured.
    tail = df.index[df["Date"] >= "2016-01-01"]
    err = np.abs(pred[tail] - df["Close"].to_numpy()[tail]) / df["Close"].to_numpy()[tail]
    real = np.sign(np.diff(df["Close"].to_numpy()[tail[0] - 1 : tail[-1] + 1]))
    guess = np.sign(pred[tail] - df["Close"].to_numpy()[tail - 1])
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB), params={params:,}")
    print(f"2016-2020: MAPE={err.mean()*100:.2f}%  median APE={np.median(err)*100:.2f}%  direction={np.mean(real==guess)*100:.1f}%")
    allerr = np.abs(pred[1:] - df["Close"].to_numpy()[1:]) / df["Close"].to_numpy()[1:]
    print(f"all: MAPE={np.nanmean(allerr)*100:.2f}%")


if __name__ == "__main__":
    main()
