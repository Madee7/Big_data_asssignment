"""
dashboard/app.py
----------------
Streamlit Dashboard — Stock Portfolio Analyzer

Pages
-----
  📊 Portfolio Overview    cluster scatter, KPIs, ticker table
  🔍 Stock Deep Dive       price chart, moving averages, peer compare
  🤖 ML Insights           model metrics, silhouette, feature importances
  🧮 Return Simulator      live classification prediction with PROBABILITIES
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ── resolve project root ──────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Portfolio Analyzer",
    page_icon=cfg.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp, .main              { background-color: #0d1117; }

.kpi-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 22px 16px;
    text-align: center;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.kpi-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e6edf3;
    border-left: 4px solid #58a6ff;
    padding-left: 10px;
    margin: 28px 0 14px;
}
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _load_cluster_results() -> pd.DataFrame | None:
    p = os.path.join(cfg.MODELS_DIR, "cluster_results.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data(ttl=3600)
def _load_feature_importances() -> pd.DataFrame | None:
    p = os.path.join(cfg.MODELS_DIR, "feature_importances.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data(ttl=3600)
def _load_pipeline_summary() -> dict | None:
    p = os.path.join(cfg.PROCESSED_DIR, "pipeline_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def _load_raw_stock(ticker: str) -> pd.DataFrame | None:
    p = os.path.join(cfg.BASE_DIR, "data", "raw", f"{ticker}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, parse_dates=["date"])

@st.cache_resource
def _load_ml_models():
    model = joblib.load(os.path.join(cfg.MODELS_DIR, "rf_return_model.joblib"))
    scaler = joblib.load(os.path.join(cfg.MODELS_DIR, "rf_scaler.joblib"))
    return model, scaler

def _predict(feature_row: dict) -> tuple:
    try:
        model, scaler = _load_ml_models()
        X = np.array([[feature_row.get(f, 0.0) for f in cfg.PREDICTOR_FEATURES]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        # Extract the probability of Class 1 (Profit)
        prob = model.predict_proba(X_scaled)[0][1]
        return float(pred), float(prob)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0, 0.0

def _no_data():
    st.info("🚀 **Pipeline output not found.**\n\nRun the pipeline first:\n`python run_pipeline.py`", icon="ℹ️")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Portfolio Analyzer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Portfolio Overview", "🔍 Stock Deep Dive", "🤖 ML Insights", "🧮 Return Simulator"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    summary = _load_pipeline_summary()
    if summary:
        st.markdown("**Pipeline status** ✅")
        st.caption(f"Tickers : {summary.get('tickers_processed', '—')}")
        st.caption(f"Duration: {summary.get('pipeline_duration_seconds', '—')}s")
    else:
        st.warning("Pipeline not run yet.\n\n`python run_pipeline.py`")


# ═════════════════════════════════════════════════════════════════
# PAGE 1 — Portfolio Overview
# ═════════════════════════════════════════════════════════════════
if page == "📊 Portfolio Overview":
    st.markdown("# 📊 Portfolio Overview")
    
    clusters = _load_cluster_results()
    if clusters is None: _no_data()

    order = ["Conservative", "Balanced", "Growth", "Aggressive"]
    cols  = st.columns(4)
    for i, label in enumerate(order):
        n     = len(clusters[clusters["cluster_label"] == label])
        color = cfg.CLUSTER_COLORS.get(label, "#888")
        cols[i].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{n}</div><div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-title">Risk vs Return by Cluster</div>', unsafe_allow_html=True)
    
    # FIX: Clip negative Sharpe ratios so Plotly doesn't crash on size
    clusters["plot_size"] = clusters["sharpe_ratio"].clip(lower=0.1)

    fig = px.scatter(
        clusters, x="volatility", y="annualised_return", color="cluster_label",
        color_discrete_map=cfg.CLUSTER_COLORS, text="ticker", size="plot_size", size_max=30,
        hover_data={"beta": ":.2f", "sharpe_ratio": ":.2f", "plot_size": False},
        template="plotly_dark", title="Portfolio Clustering — Risk / Return Space",
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3", height=520)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Avg Sharpe by Profile</div>', unsafe_allow_html=True)
        df_s = clusters.groupby("cluster_label")["sharpe_ratio"].mean().reindex(order).reset_index()
        fig_s = px.bar(df_s, x="cluster_label", y="sharpe_ratio", color="cluster_label", color_discrete_map=cfg.CLUSTER_COLORS, template="plotly_dark")
        fig_s.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3", showlegend=False, height=300)
        st.plotly_chart(fig_s, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Avg Beta by Profile</div>', unsafe_allow_html=True)
        df_b = clusters.groupby("cluster_label")["beta"].mean().reindex(order).reset_index()
        fig_b = px.bar(df_b, x="cluster_label", y="beta", color="cluster_label", color_discrete_map=cfg.CLUSTER_COLORS, template="plotly_dark")
        fig_b.add_hline(y=1.0, line_dash="dash", line_color="#58a6ff")
        fig_b.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3", showlegend=False, height=300)
        st.plotly_chart(fig_b, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — Stock Deep Dive
# ═════════════════════════════════════════════════════════════════
elif page == "🔍 Stock Deep Dive":
    st.markdown("# 🔍 Stock Deep Dive")
    clusters = _load_cluster_results()
    if clusters is None: _no_data()

    ticker = st.selectbox("Select Ticker", sorted(clusters["ticker"].tolist()))
    row    = clusters[clusters["ticker"] == ticker].iloc[0]
    label  = row["cluster_label"]
    color  = cfg.CLUSTER_COLORS.get(label, "#888")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ann. Return",  f"{row['annualised_return']:.1%}")
    m2.metric("Volatility",   f"{row['volatility']:.1%}")
    m3.metric("Sharpe Ratio", f"{row['sharpe_ratio']:.2f}")
    m4.metric("Beta (MSFT)",  f"{row['beta']:.2f}")
    st.markdown("---")

    raw = _load_raw_stock(ticker)
    if raw is not None:
        raw = raw.sort_values("date")
        raw["ma7"]  = raw["close"].rolling(7).mean()
        raw["ma30"] = raw["close"].rolling(30).mean()

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=raw["date"], y=raw["close"], name="Close", line=dict(color="#58a6ff", width=1.5)))
        fig_p.add_trace(go.Scatter(x=raw["date"], y=raw["ma7"], name="MA7", line=dict(color="#f59e0b", width=1, dash="dot")))
        fig_p.add_trace(go.Scatter(x=raw["date"], y=raw["ma30"], name="MA30", line=dict(color="#10b981", width=1, dash="dash")))
        fig_p.update_layout(title=f"{ticker} — Price", template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3", height=380)
        st.plotly_chart(fig_p, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — ML Insights
# ═════════════════════════════════════════════════════════════════
elif page == "🤖 ML Insights":
    st.markdown("# 🤖 ML Insights")
    summary = _load_pipeline_summary()
    fi      = _load_feature_importances()
    if summary is None: _no_data()

    st.markdown('<div class="section-title">Classifier Model Performance</div>', unsafe_allow_html=True)
    rp = summary.get("return_prediction", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Winning Model", rp.get("winning_model", "Classifier").replace(" Classifier", ""))
    m2.metric("Test Accuracy", f"{rp.get('accuracy', 0)*100:.1f}%")
    m3.metric("F1-Score",      f"{rp.get('f1_score', 0):.3f}")
    m4.metric("Forecast (days)", str(rp.get("forward_days", cfg.FORWARD_DAYS)))

    if fi is not None:
        st.markdown('<div class="section-title">Feature Importances</div>', unsafe_allow_html=True)
        fig_fi = px.bar(
            fi.sort_values("importance"), x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Blues", template="plotly_dark",
        )
        fig_fi.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3", height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 4 — Return Simulator
# ═════════════════════════════════════════════════════════════════
elif page == "🧮 Return Simulator":
    st.markdown(f"# 🧮 {cfg.FORWARD_DAYS}-Day Return Simulator")
    st.caption("Watch the 'Profit Probability' shift in real-time as you tune the indicators!")
    st.markdown("---")

    col_l, col_m, col_r = st.columns(3)

    with col_l:
        st.markdown("**Market Context**")
        daily_return = st.slider("Daily Return (%)", -5.0, 5.0, 0.0, 0.1) / 100
        bench_return = st.slider("Bench Return (%)", -5.0, 5.0, 0.0, 0.1) / 100
        volatility   = st.slider("Volatility (ann.)", 0.05, 1.0, 0.25, 0.01)
        beta         = st.slider("Beta", 0.0,  3.0, 1.0, 0.05)
        day_of_week  = st.selectbox("Day of Week", [1, 2, 3, 4, 5], format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][x-1])

    with col_m:
        st.markdown("**Momentum / Alpha**")
        momentum_14d = st.slider("14-Day Momentum", -0.20, 0.20, 0.0, 0.01)
        drawdown_30d = st.slider("30-Day Drawdown", -0.30, 0.0, 0.0, 0.01)
        ma_spread    = st.slider("MA Spread (Short vs Long)", -0.15, 0.15, 0.0, 0.01)
        volume_ratio = st.slider("Volume Spike Ratio", 0.1, 5.0, 1.0, 0.1)
        sharpe_ratio = st.slider("Sharpe Ratio", -2.0, 4.0, 0.8, 0.1)

    with col_r:
        st.markdown("**Technical & Lags (The Heavy Hitters)**")
        return_lag_1  = st.slider("Lag Return (1 Day Ago) %", -5.0, 5.0, 0.0, 0.1) / 100
        return_lag_5  = st.slider("Lag Return (5 Days Ago) %", -5.0, 5.0, 0.0, 0.1) / 100
        return_lag_10 = st.slider("Lag Return (10 Days Ago) %", -5.0, 5.0, 0.0, 0.1) / 100
        rsi_binned   = st.radio("RSI State", [0.0, 1.0, 2.0], format_func=lambda x: {0.0: "Oversold (<30)", 1.0: "Neutral", 2.0: "Overbought (>70)"}[x])
        bb_pct_b     = st.slider("Bollinger %B", 0.0, 1.0, 0.5, 0.01)

    # Build the exact dictionary the classifier is expecting
    feature_row = {
        "daily_return":    daily_return,
        "ma7":             150.0, 
        "ma30":            150.0, 
        "ma_cross_signal": 1.0 if ma_spread > 0 else -1.0,
        "rsi":             50.0,  
        "bb_pct_b":        bb_pct_b,
        "volatility":      volatility,
        "sharpe_ratio":    sharpe_ratio,
        "beta":            beta,
        "bench_return":    bench_return,
        "volume_ma20":     1000000, 
        "volume_ratio":    volume_ratio,        
        "return_lag_1":    return_lag_1,   # UNLOCKED!
        "return_lag_5":    return_lag_5,   # UNLOCKED!
        "return_lag_10":   return_lag_10,  # UNLOCKED!
        "momentum_14d":    momentum_14d,
        "drawdown_30d":    drawdown_30d,
        "day_of_week":     float(day_of_week),
        "ma_spread":       ma_spread,
        "rsi_binned":      rsi_binned,
    }

    st.markdown("---")
    
    # We predict live on every slider change!
    pred, prob = _predict(feature_row)
    
    color = "#10b981" if pred == 1.0 else "#ef4444"
    text  = "PROFIT (UP) 🚀" if pred == 1.0 else "LOSS (DOWN) 📉"
    
    # Render UI
    st.markdown(
        f"<div style='text-align:center;padding:40px;"
        f"background:linear-gradient(135deg,#161b22,#21262d);"
        f"border:2px solid {color};border-radius:16px;margin-top:20px'>"
        f"<div style='color:#8b949e;font-size:1.1rem;margin-bottom:8px'>"
        f"PREDICTED DIRECTION ({cfg.FORWARD_DAYS} DAYS)</div>"
        f"<div style='color:{color};font-size:3.5rem;font-weight:800;"
        f"font-family:JetBrains Mono,monospace'>{text}</div>"
        f"<div style='color:#e6edf3;font-size:1.2rem;margin-top:15px;'>"
        f"Confidence Probability: <b>{prob*100:.1f}%</b> Profit</div>"
        f"</div>",
        unsafe_allow_html=True,
    )