"""
config.py
---------
Central configuration for the Stock Portfolio Pipeline.
All tunable parameters live here — edit this file only.

CHANGES FROM ORIGINAL:
  - TICKERS updated to match actual Kaggle data (14 tickers, not 26)
  - START_DATE / END_DATE aligned to actual data range (2013-2018)
  - BENCHMARK_TICKER changed from AAPL to MSFT (AAPL still in universe)
  - FORWARD_DAYS reduced from 30 → 10  (less noisy, more predictable)
  - N_ESTIMATORS increased from 200 → 500 (better generalisation)
  - MAX_DEPTH reduced from 8 → 6  (less overfitting)
  - PREDICTOR_FEATURES expanded with MACD, volume, lag returns
  - RETURN_CLIP added to winsorise extreme daily returns
"""

import os

# ─────────────────────────────────────────────
# Stock Universe
# ─────────────────────────────────────────────
# FIX: trimmed to the 14 tickers actually present in the Kaggle dataset.
# Removed: META, NVDA, PFE, QQQ, SPY, TSLA, UNH, WFC, WMT, XOM
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMD", "INTC",
    # Finance
    "JPM", "BAC", "GS", "BLK",
    # Healthcare
    "JNJ", "ABBV",
    # Consumer
    "AMZN", "COST",
    # Energy
    "CVX",
]

# FIX: SPY and QQQ are not in the dataset — use MSFT as a stable benchmark
BENCHMARK_TICKER = "MSFT"

# ─────────────────────────────────────────────
# Date Range
# ─────────────────────────────────────────────
# FIX: updated to match actual data range in the Kaggle parquet files
START_DATE = "2020-01-01"
END_DATE   = "2025-01-01"

# ─────────────────────────────────────────────
# Paths  (all relative to this file's directory)
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

for _d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────
# Data Cleaning
# ─────────────────────────────────────────────
# FIX: winsorise daily returns to reduce impact of earnings shocks
# (e.g. AMD -24.2% on 2017-05-02 is real but hurts model training)
RETURN_CLIP_LOWER = -0.15   # –15 %
RETURN_CLIP_UPPER =  0.15   # +15 %

# ─────────────────────────────────────────────
# Feature Engineering Windows
# ─────────────────────────────────────────────
SHORT_WINDOW  = 7    # short moving-average window (days)
LONG_WINDOW   = 30   # long  moving-average window (days)
RSI_PERIOD    = 14   # RSI look-back (days)
BOLLINGER_WIN = 20   # Bollinger Band window (days)
BOLLINGER_STD = 2    # Bollinger Band std-deviation multiplier

# MACD windows
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# Lag return windows (days)
RETURN_LAGS = [1, 5, 10]

# Volume moving average window
VOLUME_MA_WIN = 20

# ─────────────────────────────────────────────
# ML — Clustering
# ─────────────────────────────────────────────
N_CLUSTERS     = 4
CLUSTER_LABELS = {
    0: "Conservative",
    1: "Balanced",
    2: "Growth",
    3: "Aggressive",
}
CLUSTER_COLORS = {
    "Conservative": "#3B82F6",
    "Balanced":     "#10B981",
    "Growth":       "#F59E0B",
    "Aggressive":   "#EF4444",
}

# ─────────────────────────────────────────────
# ML — Return Prediction
# ─────────────────────────────────────────────
# FIX: 10-day forward return is far less noisy than 30-day
FORWARD_DAYS = 10

# FIX: more trees (500) + shallower depth (6) = better bias-variance balance
N_ESTIMATORS = 500
MAX_DEPTH    = 6
RANDOM_STATE = 42
TEST_SIZE    = 0.2   # NOTE: use walk-forward split in ml/return_predictor.py

# FIX: expanded feature set — adds MACD, volume ratio, lag returns
PREDICTOR_FEATURES = [
    # Price & trend
    "daily_return",
    "ma7",
    "ma30",
    "ma_cross_signal",
    "ma_spread",           # relative distance MA7 vs MA30
    # Momentum & oscillators
    "rsi",
    "rsi_binned",          # 0=oversold, 1=neutral, 2=overbought
    "bb_pct_b",
    "momentum_14d",        # 14-day rate of change
    # Risk & drawdown
    "volatility",
    "sharpe_ratio",
    "beta",
    "drawdown_30d",        # distance from 30-day high
    # Seasonality
    "day_of_week",
    # Volume
    "volume_ma20",
    "volume_ratio",
    # Lag returns
    "return_lag_1",
    "return_lag_5",
    "return_lag_10",
]

# ─────────────────────────────────────────────
# Streamlit
# ─────────────────────────────────────────────
APP_TITLE = "📈 Stock Portfolio Analyzer"
APP_ICON  = "📈"
