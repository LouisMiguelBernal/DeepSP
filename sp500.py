"""
DeepS&P — LSTM forecasting and Monte Carlo simulation on 90+ years of S&P 500 data.

Run:  streamlit run sp500.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from plotly.subplots import make_subplots

import theme

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(
    page_title="DeepS&P — LSTM Market Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

P = theme.apply("deepsp")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(
        ASSETS / "SPX.csv", parse_dates=["Date"], index_col="Date"
    ).sort_index()

    chart = df.loc["1930-01-01":"2020-12-31"].copy()
    chart["SMA50"] = chart["Close"].rolling(50).mean()
    chart["SMA200"] = chart["Close"].rolling(200).mean()
    chart["Daily_Return"] = chart["Close"].pct_change()
    chart["Volatility_30D"] = chart["Daily_Return"].rolling(30).std() * np.sqrt(252)
    # Peak-to-trough drawdown, used by the analytics tab.
    chart["Drawdown"] = chart["Close"] / chart["Close"].cummax() - 1.0
    return df, chart


try:
    df, df_chart = load_data()
except Exception as exc:  # missing or malformed CSV
    st.error(f"Could not load `assets/SPX.csv` — {exc}")
    st.stop()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StockLSTM(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


@st.cache_resource(show_spinner=False)
def load_model():
    scaler = joblib.load(ASSETS / "scaler_spx_gpu_safe.save")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM().to(device)
    model.load_state_dict(
        torch.load(ASSETS / "lstm_spx_gpu_safe.pth", map_location=device)
    )
    model.eval()
    return scaler, model, device


try:
    scaler, model, device = load_model()
except Exception as exc:
    st.error(f"Could not load the model from `assets/` — {exc}")
    st.stop()


@st.cache_data(show_spinner=False)
def predict_at(idx: int) -> float:
    """Run the LSTM over every close up to (not including) `idx`.

    Cached on the index because a full-history forward pass costs a second or
    two on CPU and the page reruns on every widget interaction.
    """
    seq = df["Close"].values[:idx].reshape(-1, 1)
    seq_scaled = scaler.transform(seq)
    tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(tensor).item()
    return float(scaler.inverse_transform([[pred_scaled]])[0][0])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

last_close = float(df_chart["Close"].iloc[-1])
prev_close = float(df_chart["Close"].iloc[-2])
change = last_close - prev_close
change_pct = change / prev_close * 100
device_name = "CUDA" if torch.cuda.is_available() else "CPU"

with st.sidebar:
    st.markdown(
        '<div class="tk-eyebrow">DeepS&amp;P</div>'
        '<div style="font-size:1.05rem;font-weight:700;letter-spacing:-.02em;'
        'margin:.3rem 0 1.2rem;">LSTM Market Forecasting</div>',
        unsafe_allow_html=True,
    )

    theme.kv_panel(
        "Series",
        [
            ("Index", "S&P 500 (SPX)"),
            ("From", df_chart.index[0].strftime("%d %b %Y")),
            ("To", df_chart.index[-1].strftime("%d %b %Y")),
            ("Sessions", f"{len(df_chart):,}"),
            ("Avg volume", f"{df_chart['Volume'].mean():,.0f}"),
        ],
    )

    theme.kv_panel(
        "Final bar",
        [
            ("Close", f"${last_close:,.2f}"),
            ("Change", f"{change:+,.2f}"),
            ("Change %", f"{change_pct:+.2f}%"),
        ],
    )

    st.markdown(
        theme.badge("Model loaded", "pos", dot=True)
        + " "
        + theme.badge(f"Compute · {device_name}", "accent"),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="color:var(--faint);font-size:.75rem;line-height:1.6;'
        'margin-top:1.4rem;">Research and education only. Nothing here is '
        "investment advice.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

theme.hero(
    "Deep<em>S&amp;P</em>",
    "A three-layer LSTM trained on ninety years of S&P 500 closes, paired with "
    "a Monte Carlo engine for forward path simulation.",
    eyebrow="S&P 500 · Neural Forecasting",
    meta=[
        theme.badge("LSTM · 3 × 256", "accent"),
        theme.badge(f"{len(df_chart):,} sessions"),
        theme.badge("1930 – 2020"),
        theme.badge(device_name, "pos", dot=True),
    ],
)

tab_pred, tab_mc, tab_analytics, tab_model = st.tabs(
    ["Prediction", "Monte Carlo", "Analytics", "Model"]
)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

with tab_pred:
    theme.section(
        "Point-in-time backtest",
        "Pick any historical session. The model sees only the closes that "
        "preceded it, then predicts that day's close — so the error below is "
        "an out-of-sample read on a single bar.",
    )

    col_date, col_run = st.columns([2, 1])
    with col_date:
        selected = st.date_input(
            "Session",
            value=df_chart.index[-1].date(),
            min_value=df.index[1].date(),
            max_value=df_chart.index[-1].date(),
        )
    with col_run:
        st.markdown('<div style="height:1.72rem"></div>', unsafe_allow_html=True)
        st.caption("Non-trading dates snap back to the previous session.")

    selected_date = pd.Timestamp(selected)
    if selected_date not in df.index:
        pos = np.searchsorted(df.index.values, np.datetime64(selected_date))
        if pos == 0:
            st.warning("No history available before that date.")
            st.stop()
        selected_date = df.index[pos - 1]
        st.caption(f"Snapped to {selected_date.strftime('%d %B %Y')}.")

    idx = df.index.get_loc(selected_date)
    if idx <= 0:
        st.warning("Not enough history before that date to run the model.")
        st.stop()

    with st.spinner("Running the sequence through the network…"):
        predicted = predict_at(idx)

    actual = float(df.loc[selected_date, "Close"])
    err = predicted - actual
    err_pct = abs(err) / actual * 100

    theme.stat_row(
        [
            {
                "label": "Actual close",
                "value": f"${actual:,.2f}",
                "delta": selected_date.strftime("%d %b %Y"),
            },
            {
                "label": "LSTM prediction",
                "value": f"${predicted:,.2f}",
                "delta": "Sequence-to-point forecast",
                "tone": "accent",
            },
            {
                "label": "Absolute error",
                "value": f"${abs(err):,.2f}",
                "delta": f"{err:+,.2f} vs actual",
            },
            {
                "label": "Error",
                "value": f"{err_pct:.2f}%",
                "delta": "Relative to actual close",
                "tone": "pos" if err_pct < 2 else "warn" if err_pct < 5 else "neg",
            },
        ]
    )

    theme.section("Price history", "Close with 50 and 200-session moving averages.")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
    )
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["Close"],
            name="Close",
            line=dict(color=P["accent"], width=1.6),
            hovertemplate="$%{y:,.2f}<extra>Close</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["SMA50"],
            name="SMA 50",
            line=dict(color=P["warn"], width=1, dash="dot"),
            hovertemplate="$%{y:,.2f}<extra>SMA 50</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["SMA200"],
            name="SMA 200",
            line=dict(color=P["accent_2"], width=1, dash="dash"),
            hovertemplate="$%{y:,.2f}<extra>SMA 200</extra>",
        ),
        row=1,
        col=1,
    )
    if selected_date in df_chart.index:
        fig.add_trace(
            go.Scatter(
                x=[selected_date],
                y=[actual],
                name="Selected",
                mode="markers",
                marker=dict(
                    color=P["bg"],
                    size=11,
                    line=dict(color=P["text"], width=2),
                ),
                hovertemplate="$%{y:,.2f}<extra>Selected session</extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=df_chart.index,
            y=df_chart["Volume"],
            name="Volume",
            marker_color="rgba(138,145,158,0.4)",
            hovertemplate="%{y:,.0f}<extra>Volume</extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    theme.style_fig(fig, height=620)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Price axis is logarithmic — over ninety years a linear axis flattens "
        "everything before 1990 into a single line."
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def simulate(last_price: float, mu: float, sigma: float, days: int, paths: int, seed: int):
    """Geometric random walk on daily returns drawn from the historical
    distribution. Vectorised — one cumulative product, no Python loop."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, size=(days - 1, paths))
    walk = np.vstack([np.ones((1, paths)), np.cumprod(1 + shocks, axis=0)])
    return last_price * walk


with tab_mc:
    theme.section(
        "Forward path simulation",
        "Draws daily returns from the historical mean and standard deviation, "
        "then compounds them forward. It models dispersion, not direction.",
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        num_paths = st.slider("Paths", 100, 2000, 500, 50)
    with c2:
        num_days = st.slider("Horizon (sessions)", 30, 365, 180, 15)
    with c3:
        seed = st.number_input("Seed", value=42, step=1)

    if st.button("Run simulation", type="primary"):
        st.session_state["mc"] = (int(num_paths), int(num_days), int(seed))

    if "mc" in st.session_state:
        n_paths, n_days, mc_seed = st.session_state["mc"]
        returns = df_chart["Close"].pct_change().dropna()
        mu, sigma = float(returns.mean()), float(returns.std())

        with st.spinner("Simulating…"):
            paths = simulate(last_close, mu, sigma, n_days, n_paths, mc_seed)

        final = paths[-1]
        p5, p95 = np.percentile(final, [5, 95])
        median = float(np.median(final))
        total_ret = median / last_close - 1

        theme.stat_row(
            [
                {"label": "Start", "value": f"${last_close:,.2f}", "delta": "Final close"},
                {
                    "label": "Median outcome",
                    "value": f"${median:,.2f}",
                    "delta": f"{total_ret:+.1%} over {n_days} sessions",
                    "tone": "accent",
                },
                {"label": "5th percentile", "value": f"${p5:,.2f}", "tone": "neg"},
                {"label": "95th percentile", "value": f"${p95:,.2f}", "tone": "pos"},
                {
                    "label": "Dispersion",
                    "value": f"${np.std(final):,.2f}",
                    "delta": "Std. dev. of final price",
                },
            ]
        )

        # Fan chart. Every sampled path goes into a single trace separated by
        # NaNs — 300 individual traces would make the figure unusable.
        sample = min(120, n_paths)
        xs, ys = [], []
        steps = np.arange(n_days, dtype=float)
        for i in range(sample):
            xs.extend(steps.tolist() + [None])
            ys.extend(paths[:, i].tolist() + [None])

        p5_path = np.percentile(paths, 5, axis=1)
        p95_path = np.percentile(paths, 95, axis=1)
        median_path = np.median(paths, axis=1)

        fig_mc = go.Figure()
        fig_mc.add_trace(
            go.Scatter(
                x=steps, y=p95_path, name="95th percentile",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            )
        )
        fig_mc.add_trace(
            go.Scatter(
                x=steps, y=p5_path, name="5–95% band",
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty", fillcolor=P["accent_soft"], hoverinfo="skip",
            )
        )
        fig_mc.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", name=f"{sample} sampled paths",
                line=dict(color="rgba(138,145,158,0.20)", width=0.8),
                hoverinfo="skip", connectgaps=False,
            )
        )
        fig_mc.add_trace(
            go.Scatter(
                x=steps, y=median_path, name="Median path",
                line=dict(color=P["accent"], width=2.4),
                hovertemplate="$%{y:,.2f}<extra>Median</extra>",
            )
        )
        fig_mc.update_xaxes(title_text="Sessions ahead")
        fig_mc.update_yaxes(title_text="Price")
        theme.style_fig(fig_mc, height=520)
        st.plotly_chart(fig_mc, use_container_width=True)

        theme.section("Terminal distribution")
        fig_h = go.Figure(
            go.Histogram(
                x=final, nbinsx=60, marker_color=P["accent"],
                marker_line=dict(width=0), opacity=0.85,
                hovertemplate="$%{x:,.0f} · %{y} paths<extra></extra>",
            )
        )
        for value, label, color in (
            (p5, "5th", P["neg"]),
            (median, "median", P["text"]),
            (p95, "95th", P["pos"]),
        ):
            fig_h.add_vline(
                x=value, line_dash="dot", line_color=color, line_width=1.5,
                annotation_text=f"{label} ${value:,.0f}",
                annotation_font_color=color, annotation_font_size=11,
            )
        fig_h.update_xaxes(title_text="Price after horizon")
        fig_h.update_yaxes(title_text="Paths")
        theme.style_fig(fig_h, height=330, legend=False)
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        theme.empty_state(
            "No simulation yet",
            "Set the number of paths and the horizon, then run the simulation. "
            "The seed makes each run reproducible.",
        )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

with tab_analytics:
    ann_vol = float(df_chart["Volatility_30D"].iloc[-1] * 100)
    max_dd = float(df_chart["Drawdown"].min() * 100)
    dd_date = df_chart["Drawdown"].idxmin()
    best = df_chart["Daily_Return"].max() * 100
    worst = df_chart["Daily_Return"].min() * 100

    theme.section("Historical profile", "Whole-series statistics, 1930 to 2020.")
    theme.stat_row(
        [
            {"label": "All-time high", "value": f"${df_chart['Close'].max():,.2f}"},
            {"label": "All-time low", "value": f"${df_chart['Close'].min():,.2f}"},
            {
                "label": "Avg daily return",
                "value": f"{df_chart['Daily_Return'].mean() * 100:.3f}%",
                "delta": f"≈ {df_chart['Daily_Return'].mean() * 252 * 100:.1f}% annualised",
                "tone": "pos",
            },
            {"label": "30d volatility", "value": f"{ann_vol:.1f}%", "delta": "Annualised"},
            {
                "label": "Max drawdown",
                "value": f"{max_dd:.1f}%",
                "delta": f"Trough {dd_date.strftime('%b %Y')}",
                "tone": "neg",
            },
            {
                "label": "Best / worst day",
                "value": f"{best:+.1f}% / {worst:.1f}%",
            },
        ]
    )

    theme.section("Drawdown", "Distance below the running all-time high.")
    fig_dd = go.Figure(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["Drawdown"] * 100,
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.16)",
            line=dict(color=P["neg"], width=1.1),
            name="Drawdown",
            hovertemplate="%{y:.1f}%<extra>Drawdown</extra>",
        )
    )
    fig_dd.update_yaxes(title_text="%", ticksuffix="%")
    theme.style_fig(fig_dd, height=280, legend=False)
    st.plotly_chart(fig_dd, use_container_width=True)

    theme.section("Rolling 30-session volatility", "Annualised.")
    fig_vol = go.Figure(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["Volatility_30D"] * 100,
            line=dict(color=P["accent_2"], width=1.1),
            name="Volatility",
            hovertemplate="%{y:.1f}%<extra>Volatility</extra>",
        )
    )
    fig_vol.update_yaxes(title_text="%", ticksuffix="%")
    theme.style_fig(fig_vol, height=260, legend=False)
    st.plotly_chart(fig_vol, use_container_width=True)

    theme.section("Data explorer")
    years = list(range(int(df_chart.index[0].year), int(df_chart.index[-1].year) + 1))
    c1, c2 = st.columns(2)
    with c1:
        y0 = st.selectbox("From year", years, index=max(0, len(years) - 21))
    with c2:
        y1 = st.selectbox("To year", years, index=len(years) - 1)
    if y1 < y0:
        y0, y1 = y1, y0

    subset = df_chart.loc[f"{y0}-01-01":f"{y1}-12-31"]
    st.dataframe(
        subset[["Open", "High", "Low", "Close", "Volume", "SMA50", "SMA200"]],
        use_container_width=True,
        height=380,
        column_config={
            col: st.column_config.NumberColumn(col, format="$%.2f")
            for col in ("Open", "High", "Low", "Close", "SMA50", "SMA200")
        }
        | {"Volume": st.column_config.NumberColumn("Volume", format="%d")},
    )
    st.download_button(
        f"Download {y0}–{y1} as CSV",
        data=subset.to_csv().encode(),
        file_name=f"spx_{y0}_{y1}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

with tab_model:
    theme.section("Architecture and training")

    c1, c2 = st.columns(2)
    with c1:
        theme.kv_panel(
            "Network",
            [
                ("Type", "Stacked LSTM"),
                ("Hidden size", "256"),
                ("Layers", "3"),
                ("Dropout", "0.20"),
                ("Input features", "1 (close)"),
                ("Head", "Linear → 1"),
                ("Parameters", f"{sum(p.numel() for p in model.parameters()):,}"),
            ],
        )
        theme.panel(
            "How a prediction is made",
            "<p>The full sequence of closes preceding the selected session is "
            "min-max scaled, pushed through the three LSTM layers, and the final "
            "hidden state is mapped to a single value by the dense head. That "
            "value is inverse-transformed back into dollars.</p>"
            "<p>Because the model never sees the target bar, the error shown on "
            "the Prediction tab is genuinely out-of-sample for that date.</p>",
        )
    with c2:
        theme.kv_panel(
            "Training",
            [
                ("Period", "1930 – 2020"),
                ("Observations", f"{len(df):,}"),
                ("Scaling", "MinMax"),
                ("Framework", f"PyTorch {torch.__version__.split('+')[0]}"),
                ("Inference device", device_name),
            ],
        )
        theme.panel(
            "Limits worth knowing",
            theme.bullets(
                [
                    "**Univariate. **Close price only — no volume, macro, or "
                    "cross-asset inputs.",
                    "**Single step. **It predicts one bar ahead, not a path. "
                    "Chaining its own outputs would compound error quickly.",
                    "**Regime-bound. **Trained through 2020; the distribution it "
                    "learned may not describe the market you trade.",
                    "**Not advice. **This is a research artefact. Do not size "
                    "positions with it.",
                ]
            ),
        )

    theme.section("Forward pass")
    layers = ["Sequence", "LSTM ×256", "LSTM ×256", "LSTM ×256", "Dense", "Close"]
    colors = [P["faint"], P["accent"], P["accent"], P["accent"], P["accent_2"], P["pos"]]
    fig_arch = go.Figure()
    fig_arch.add_trace(
        go.Scatter(
            x=list(range(len(layers))),
            y=[0] * len(layers),
            mode="markers+text",
            marker=dict(size=64, color=P["surface"], line=dict(color=colors, width=2)),
            text=layers,
            textposition="middle center",
            textfont=dict(color=P["text"], size=10, family="Inter"),
            hoverinfo="skip",
        )
    )
    for i in range(len(layers) - 1):
        fig_arch.add_annotation(
            x=i + 1, y=0, ax=i, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor=P["faint"], standoff=36, startstandoff=36,
        )
    fig_arch.update_xaxes(visible=False, range=[-0.6, len(layers) - 0.4])
    fig_arch.update_yaxes(visible=False, range=[-1, 1])
    theme.style_fig(fig_arch, height=170, legend=False)
    st.plotly_chart(fig_arch, use_container_width=True)


theme.footer(
    '<b>DeepS&amp;P</b> · LSTM + Monte Carlo on 90 years of S&amp;P 500 data',
    "PyTorch · Streamlit · Plotly — research use only",
)
