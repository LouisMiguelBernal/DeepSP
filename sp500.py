import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="📈 DeepS&P ", layout="wide")
st.title("📈 DeepS&P LSTM Forecast Dashboard")

# ---------------------------------------------------------
# Load SPX CSV
# ---------------------------------------------------------
df = pd.read_csv("assets/SPX.csv", parse_dates=["Date"], index_col="Date").sort_index()
df_chart = df.loc["1930-01-01":"2020-12-31"].copy()
df_chart["SMA50"] = df_chart["Close"].rolling(50).mean()
df_chart["SMA200"] = df_chart["Close"].rolling(200).mean()

# ---------------------------------------------------------
# Load Scaler + Model
# ---------------------------------------------------------
scaler = joblib.load("assets/scaler_spx_gpu_safe.save")

class StockLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockLSTM(hidden_size=256, num_layers=3, dropout=0.2).to(device)
model.load_state_dict(torch.load("assets/lstm_spx_gpu_safe.pth", map_location=device))
model.eval()

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 LSTM Prediction",
    "📉 Monte Carlo Simulation",
    "📄 Raw Data",
    "ℹ️ Model Information"
])

# ---------------------------------------------------------
# TAB 1 • LSTM Prediction
# ---------------------------------------------------------
with tab1:
    st.header("🔮 Predict S&P 500 Close Price")

    year = st.selectbox("Year", list(range(1930, 2021)))
    month = st.selectbox("Month", list(range(1, 13)))
    day = st.selectbox("Day", list(range(1, 32)))

    try:
        selected_date = pd.Timestamp(datetime(year, month, day))
    except:
        st.error("Invalid date selected.")
        st.stop()

    if selected_date not in df.index:
        dates_sorted = df.index.values
        pos = np.searchsorted(dates_sorted, np.datetime64(selected_date))
        if pos == 0:
            st.error("No data before this date.")
            st.stop()
        selected_date = df.index[pos - 1]
        st.info(f"Date not found. Using previous date: {selected_date.strftime('%Y-%m-%d')}")

    idx = df.index.get_loc(selected_date)
    seq_len = idx

    col1, col2 = st.columns([1, 2])

    with col1:
        if seq_len <= 0:
            st.error("Not enough historical data.")
        else:
            seq_data = df["Close"].values[:idx].reshape(-1, 1)
            seq_scaled = scaler.transform(seq_data)
            seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_scaled = model(seq_tensor).item()

            predicted_price = scaler.inverse_transform([[pred_scaled]])[0][0]
            actual_price = df.loc[selected_date, "Close"]

            st.subheader("📌 Prediction Results")
            st.metric("Actual Close Price", f"${actual_price:.2f}")
            st.metric("Predicted Close Price", f"${predicted_price:.2f}")

    with col2:
        st.subheader("📈 S&P 500 Historical Chart")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["Close"], mode="lines",
            name="Close", line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["SMA50"], mode="lines",
            name="SMA50", line=dict(color="orange", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["SMA200"], mode="lines",
            name="SMA200", line=dict(color="green", dash="dot")
        ))

        if selected_date in df_chart.index:
            fig.add_trace(go.Scatter(
                x=[selected_date], y=[df_chart.loc[selected_date, "Close"]],
                mode="markers+text", name="Selected Date",
                marker=dict(color="red", size=12),
                text=[f"${actual_price:.2f}"],
                textposition="top center"
            ))

        fig.update_layout(
            title="S&P 500 (1930–2020) with SMA50/200",
            height=600,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 • Monte Carlo Simulation
# ---------------------------------------------------------
with tab2:
    st.header("📉 Monte Carlo Simulation: Future SPX Price Paths")

    num_paths = st.slider("Number of Simulations", 100, 2000, 500)
    num_days = st.slider("Forecast Days", 30, 365, 180)

    last_price = df_chart["Close"].iloc[-1]

    returns = df_chart["Close"].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()

    paths = np.zeros((num_days, num_paths))
    paths[0] = last_price

    for t in range(1, num_days):
        shock = np.random.normal(mu, sigma, num_paths)
        paths[t] = paths[t - 1] * (1 + shock)

    fig_mc = go.Figure()
    for i in range(min(num_paths, 50)):  # show only 50 for clarity
        fig_mc.add_trace(go.Scatter(
            y=paths[:, i], mode="lines", line=dict(width=1), opacity=0.4,
            name=f"Path {i+1}"
        ))

    fig_mc.update_layout(
        title="Monte Carlo Simulation of Future S&P 500 Prices",
        xaxis_title="Days Ahead",
        yaxis_title="Simulated Price",
        height=600,
        template="plotly_dark"
    )

    st.plotly_chart(fig_mc, use_container_width=True)

# ---------------------------------------------------------
# TAB 3 • Raw Data Viewer
# ---------------------------------------------------------
with tab3:
    st.header("📄 Raw Dataset")
    st.dataframe(df_chart.tail(500))

# ---------------------------------------------------------
# TAB 4 • Model Information
# ---------------------------------------------------------
with tab4:
    st.header("ℹ️ Model Information")
    st.write("""
    **Model Type:** LSTM  
    **Hidden Size:** 256  
    **Layers:** 3  
    **Dropout:** 0.2  
    **Training Window:** 1930–2020  
    **Scaler:** MinMax  
    **Forecast Target:** Next-Day Close Price  
    """)
