import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="DeepS&P | AI-Powered Market Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for Professional Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #2ecc71;
        --accent-color: #e74c3c;
        --bg-dark: #0e1117;
        --bg-card: #1a1d29;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .metric-card h3 {
        color: white;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card .delta {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* Info cards */
    .info-card {
        background: #1a1d29;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .info-card h4 {
        color: #667eea;
        margin-top: 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1d29;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: transparent;
        border-radius: 8px;
        color: #a0a0a0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #1a1d29;
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-success {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
    }
    
    .status-warning {
        background-color: rgba(241, 196, 15, 0.2);
        color: #f1c40f;
    }
    
    .status-info {
        background-color: rgba(52, 152, 219, 0.2);
        color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data with Error Handling
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("assets/SPX.csv", parse_dates=["Date"], index_col="Date").sort_index()
        df_chart = df.loc["1930-01-01":"2020-12-31"].copy()
        df_chart["SMA50"] = df_chart["Close"].rolling(50).mean()
        df_chart["SMA200"] = df_chart["Close"].rolling(200).mean()
        df_chart["Daily_Return"] = df_chart["Close"].pct_change()
        df_chart["Volatility_30D"] = df_chart["Daily_Return"].rolling(30).std() * np.sqrt(252)
        return df, df_chart
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

df, df_chart = load_data()

if df is None or df_chart is None:
    st.stop()

# ---------------------------------------------------------
# Load Model with Error Handling
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
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
        
        return scaler, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

scaler, model, device = load_model_and_scaler()

if scaler is None or model is None:
    st.stop()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 2rem; margin: 0;'>📈</h1>
        <h2 style='font-size: 1.3rem; margin: 0.5rem 0;'>DeepS&P</h2>
        <p style='color: #888; font-size: 0.9rem;'>AI-Powered Market Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Market Status
    st.markdown("### 📊 Market Status")
    last_close = df_chart["Close"].iloc[-1]
    prev_close = df_chart["Close"].iloc[-2]
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Last Close", f"${last_close:.2f}", f"{change:.2f}")
    with col2:
        st.metric("Change %", f"{change_pct:.2f}%")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📈 Quick Stats")
    st.markdown(f"""
    <div class='info-card'>
        <strong>Data Range:</strong><br/>
        {df_chart.index[0].strftime('%Y-%m-%d')} to {df_chart.index[-1].strftime('%Y-%m-%d')}
    </div>
    <div class='info-card'>
        <strong>Total Trading Days:</strong><br/>
        {len(df_chart):,}
    </div>
    <div class='info-card'>
        <strong>Average Daily Volume:</strong><br/>
        {df_chart['Volume'].mean():,.0f} shares
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Status
    st.markdown("### 🤖 Model Status")
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    <span class='status-badge status-success'>● Active</span>
    <span class='status-badge status-info'>{device_name}</span>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown("""
<div class='main-header'>
    <h1>📈 DeepS&P LSTM Forecast Dashboard</h1>
    <p>Advanced AI-powered S&P 500 price prediction and market analysis platform</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 LSTM Prediction",
    "📉 Monte Carlo Simulation",
    "📊 Market Analytics",
    "ℹ️ Model Information"
])

# ---------------------------------------------------------
# TAB 1 • LSTM Prediction
# ---------------------------------------------------------
with tab1:
    st.markdown("## 🔮 AI-Powered Price Prediction")
    st.markdown("Select a historical date to see the LSTM model's prediction accuracy")
    
    # Date Selection in columns
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown("### 📅 Select Date")
    with col2:
        year = st.selectbox("Year", list(range(1930, 2021)), index=90)
    with col3:
        month = st.selectbox("Month", list(range(1, 13)), index=0)
    with col4:
        day = st.selectbox("Day", list(range(1, 32)), index=0)

    try:
        selected_date = pd.Timestamp(datetime(year, month, day))
    except:
        st.error("⚠️ Invalid date selected. Please choose a valid date.")
        st.stop()

    if selected_date not in df.index:
        dates_sorted = df.index.values
        pos = np.searchsorted(dates_sorted, np.datetime64(selected_date))
        if pos == 0:
            st.error("⚠️ No historical data available before this date.")
            st.stop()
        selected_date = df.index[pos - 1]
        st.info(f"📍 Date adjusted to nearest trading day: **{selected_date.strftime('%B %d, %Y')}**")

    idx = df.index.get_loc(selected_date)
    seq_len = idx

    st.markdown("---")

    # Prediction Section
    if seq_len <= 0:
        st.error("⚠️ Insufficient historical data for prediction.")
    else:
        # Make prediction
        seq_data = df["Close"].values[:idx].reshape(-1, 1)
        seq_scaled = scaler.transform(seq_data)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(seq_tensor).item()

        predicted_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        actual_price = df.loc[selected_date, "Close"]
        error = abs(predicted_price - actual_price)
        error_pct = (error / actual_price) * 100
        
        # Results Display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3>Actual Close Price</h3>
                <div class='value'>${:,.2f}</div>
                <div class='delta'>Market Close on {}</div>
            </div>
            """.format(actual_price, selected_date.strftime('%b %d, %Y')), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card' style='background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);'>
                <h3>LSTM Prediction</h3>
                <div class='value'>${:,.2f}</div>
                <div class='delta'>AI Model Forecast</div>
            </div>
            """.format(predicted_price), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card' style='background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);'>
                <h3>Prediction Error</h3>
                <div class='value'>{:.2f}%</div>
                <div class='delta'>${:,.2f} difference</div>
            </div>
            """.format(error_pct, error), unsafe_allow_html=True)

        st.markdown("---")

        # Chart Section
        st.markdown("### 📈 Interactive Price Chart")
        
        # Create subplot with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('S&P 500 Price & Moving Averages', 'Trading Volume')
        )

        # Price and MAs
        fig.add_trace(
            go.Scatter(
                x=df_chart.index, y=df_chart["Close"], 
                mode="lines", name="Close Price",
                line=dict(color="#3498db", width=2),
                hovertemplate='<b>Date</b>: %{x}<br><b>Close</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_chart.index, y=df_chart["SMA50"],
                mode="lines", name="SMA 50",
                line=dict(color="#f39c12", width=1.5, dash="dash"),
                hovertemplate='<b>SMA50</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_chart.index, y=df_chart["SMA200"],
                mode="lines", name="SMA 200",
                line=dict(color="#2ecc71", width=1.5, dash="dot"),
                hovertemplate='<b>SMA200</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Highlight selected date
        if selected_date in df_chart.index:
            fig.add_trace(
                go.Scatter(
                    x=[selected_date], y=[df_chart.loc[selected_date, "Close"]],
                    mode="markers+text", name="Selected Date",
                    marker=dict(color="#e74c3c", size=15, symbol="diamond"),
                    text=[f"${actual_price:.2f}"],
                    textposition="top center",
                    textfont=dict(color="white", size=12),
                    hovertemplate='<b>Selected</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Volume
        fig.add_trace(
            go.Bar(
                x=df_chart.index, y=df_chart["Volume"],
                name="Volume", marker_color="#95a5a6",
                hovertemplate='<b>Volume</b>: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=700,
            template="plotly_dark",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        fig.update_xaxes(showgrid=False, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 • Monte Carlo Simulation
# ---------------------------------------------------------
with tab2:
    st.markdown("## 📉 Monte Carlo Simulation")
    st.markdown("Simulate thousands of potential future price paths using historical volatility")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_paths = st.slider("Number of Simulation Paths", 100, 2000, 500, 50)
    with col2:
        num_days = st.slider("Forecast Horizon (Days)", 30, 365, 180, 30)

    if st.button("🚀 Run Simulation", use_container_width=True):
        with st.spinner("Running Monte Carlo simulation..."):
            last_price = df_chart["Close"].iloc[-1]
            returns = df_chart["Close"].pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std()

            paths = np.zeros((num_days, num_paths))
            paths[0] = last_price

            for t in range(1, num_days):
                shock = np.random.normal(mu, sigma, num_paths)
                paths[t] = paths[t - 1] * (1 + shock)

            # Statistics
            final_prices = paths[-1]
            mean_price = np.mean(final_prices)
            median_price = np.median(final_prices)
            std_price = np.std(final_prices)
            percentile_5 = np.percentile(final_prices, 5)
            percentile_95 = np.percentile(final_prices, 95)

            # Display statistics
            st.markdown("### 📊 Simulation Results")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Mean Price", f"${mean_price:,.2f}")
            with col2:
                st.metric("Median Price", f"${median_price:,.2f}")
            with col3:
                st.metric("Std Dev", f"${std_price:,.2f}")
            with col4:
                st.metric("5th Percentile", f"${percentile_5:,.2f}")
            with col5:
                st.metric("95th Percentile", f"${percentile_95:,.2f}")

            st.markdown("---")

            # Create visualization
            fig_mc = go.Figure()

            # Plot subset of paths
            sample_paths = min(100, num_paths)
            for i in range(sample_paths):
                fig_mc.add_trace(
                    go.Scatter(
                        y=paths[:, i],
                        mode="lines",
                        line=dict(color='rgba(52, 152, 219, 0.3)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

            # Add mean path
            mean_path = np.mean(paths, axis=1)
            fig_mc.add_trace(
                go.Scatter(
                    y=mean_path,
                    mode="lines",
                    name="Mean Path",
                    line=dict(color="#e74c3c", width=3),
                    hovertemplate='<b>Day</b>: %{x}<br><b>Mean Price</b>: $%{y:,.2f}<extra></extra>'
                )
            )

            # Add percentile bands
            p5_path = np.percentile(paths, 5, axis=1)
            p95_path = np.percentile(paths, 95, axis=1)
            
            fig_mc.add_trace(
                go.Scatter(
                    y=p95_path,
                    mode="lines",
                    name="95th Percentile",
                    line=dict(color="#2ecc71", width=2, dash="dash"),
                    hovertemplate='<b>95th Percentile</b>: $%{y:,.2f}<extra></extra>'
                )
            )
            
            fig_mc.add_trace(
                go.Scatter(
                    y=p5_path,
                    mode="lines",
                    name="5th Percentile",
                    line=dict(color="#f39c12", width=2, dash="dash"),
                    fill='tonexty',
                    fillcolor='rgba(52, 152, 219, 0.1)',
                    hovertemplate='<b>5th Percentile</b>: $%{y:,.2f}<extra></extra>'
                )
            )

            fig_mc.update_layout(
                title=f"Monte Carlo Simulation: {num_paths} Paths over {num_days} Days",
                xaxis_title="Days Ahead",
                yaxis_title="Simulated Price ($)",
                height=600,
                template="plotly_dark",
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            fig_mc.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_mc.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

            st.plotly_chart(fig_mc, use_container_width=True)

            # Distribution of final prices
            st.markdown("### 📊 Distribution of Final Prices")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(
                    x=final_prices,
                    nbinsx=50,
                    name="Final Price Distribution",
                    marker_color="#3498db",
                    hovertemplate='<b>Price Range</b>: $%{x:,.2f}<br><b>Count</b>: %{y}<extra></extra>'
                )
            )
            
            fig_hist.add_vline(
                x=mean_price, line_dash="dash", line_color="#e74c3c",
                annotation_text=f"Mean: ${mean_price:,.2f}",
                annotation_position="top right"
            )
            
            fig_hist.update_layout(
                title="Distribution of Final Simulated Prices",
                xaxis_title="Final Price ($)",
                yaxis_title="Frequency",
                height=400,
                template="plotly_dark",
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )

            st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# TAB 3 • Market Analytics
# ---------------------------------------------------------
with tab3:
    st.markdown("## 📊 Market Analytics & Historical Data")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        all_time_high = df_chart["Close"].max()
        st.metric("All-Time High", f"${all_time_high:,.2f}")
    
    with col2:
        all_time_low = df_chart["Close"].min()
        st.metric("All-Time Low", f"${all_time_low:,.2f}")
    
    with col3:
        avg_daily_return = df_chart["Daily_Return"].mean() * 100
        st.metric("Avg Daily Return", f"{avg_daily_return:.3f}%")
    
    with col4:
        current_volatility = df_chart["Volatility_30D"].iloc[-1] * 100
        st.metric("30D Volatility", f"{current_volatility:.2f}%")
    
    st.markdown("---")
    
    # Data table with filters
    st.markdown("### 📋 Historical Data Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("From Year", list(range(1930, 2021)), index=0)
    with col2:
        end_year = st.selectbox("To Year", list(range(1930, 2021)), index=90)
    
    filtered_data = df_chart.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
    
    st.dataframe(
        filtered_data[["Open", "High", "Low", "Close", "Volume", "SMA50", "SMA200"]].style.format({
            "Open": "${:,.2f}",
            "High": "${:,.2f}",
            "Low": "${:,.2f}",
            "Close": "${:,.2f}",
            "Volume": "{:,.0f}",
            "SMA50": "${:,.2f}",
            "SMA200": "${:,.2f}"
        }),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_data.to_csv()
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"spx_data_{start_year}_{end_year}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------------------------------------
# TAB 4 • Model Information
# ---------------------------------------------------------
with tab4:
    st.markdown("## ℹ️ Model Architecture & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>🧠 Neural Network Architecture</h4>
            <ul>
                <li><strong>Model Type:</strong> LSTM (Long Short-Term Memory)</li>
                <li><strong>Hidden Size:</strong> 256 units</li>
                <li><strong>Number of Layers:</strong> 3 stacked LSTM layers</li>
                <li><strong>Dropout Rate:</strong> 0.2 (20%)</li>
                <li><strong>Input Features:</strong> 1 (Close Price)</li>
                <li><strong>Output:</strong> Next-day close price prediction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h4>📊 Training Details</h4>
            <ul>
                <li><strong>Training Period:</strong> 1930–2020</li>
                <li><strong>Data Points:</strong> 90+ years of daily data</li>
                <li><strong>Normalization:</strong> MinMax Scaler</li>
                <li><strong>Framework:</strong> PyTorch</li>
                <li><strong>Device:</strong> {} capable</li>
            </ul>
        </div>
        """.format("GPU (CUDA)" if torch.cuda.is_available() else "CPU"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>🎯 Model Capabilities</h4>
            <ul>
                <li>Sequence-to-point prediction</li>
                <li>Temporal pattern recognition</li>
                <li>Long-term dependency learning</li>
                <li>Market trend analysis</li>
                <li>Volatility-aware forecasting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h4>⚠️ Disclaimer</h4>
            <p style='font-size: 0.9rem; color: #bbb;'>
            This model is for educational and research purposes only. 
            Past performance does not guarantee future results. Always consult 
            with a qualified financial advisor before making investment decisions. 
            This tool should not be used as the sole basis for any financial decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Architecture Diagram
    st.markdown("### 🏗️ LSTM Architecture Visualization")
    
    fig_arch = go.Figure()
    
    # Simple architecture visualization
    layers = ["Input\n(Sequence)", "LSTM\nLayer 1", "LSTM\nLayer 2", "LSTM\nLayer 3", "Dense\nLayer", "Output\n(Price)"]
    x_pos = list(range(len(layers)))
    
    fig_arch.add_trace(go.Scatter(
        x=x_pos,
        y=[0]*len(layers),
        mode='markers+text',
        marker=dict(size=60, color=['#3498db', '#9b59b6', '#9b59b6', '#9b59b6', '#e74c3c', '#2ecc71']),
        text=layers,
        textposition="middle center",
        textfont=dict(color='white', size=10),
        hoverinfo='skip'
    ))
    
    # Add arrows
    for i in range(len(layers)-1):
        fig_arch.add_annotation(
            x=x_pos[i+1], y=0,
            ax=x_pos[i], ay=0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#7f8c8d'
        )
    
    fig_arch.update_layout(
        height=200,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig_arch, use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p>DeepS&P LSTM Forecast Dashboard | Powered by PyTorch & Streamlit</p>
    <p style='font-size: 0.8rem;'>© 2024 Advanced Financial Analytics Platform</p>
</div>
""", unsafe_allow_html=True)
