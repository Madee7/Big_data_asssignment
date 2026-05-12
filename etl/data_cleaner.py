"""
etl/data_cleaner.py
-------------------
Applies all data-quality fixes identified in the Kaggle dataset audit
and engineers the expanded feature set needed for better R² / accuracy.

Issues fixed
------------
1. Drop 14 null daily_return rows (first trading day per ticker — no prior close).
2. Winsorise extreme daily returns at ±15% (e.g. AMD –24.2% earnings shock).
3. Adds new ML features: MACD, volume ratio, lag returns.

Usage
-----
    from etl.data_cleaner import load_and_clean, engineer_features

    df = load_and_clean()          # cleaned base DataFrame
    df = engineer_features(df)     # with all ML features added
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── resolve paths: this file is in etl/, project root is one level up ──
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)                          # big_data/
_PROC    = os.path.join(_ROOT, "data", "processed", "clean_stocks")

try:
    import config as cfg
    _CLIP_LO = cfg.RETURN_CLIP_LOWER
    _CLIP_HI = cfg.RETURN_CLIP_UPPER
    _LAGS    = cfg.RETURN_LAGS
    _VOL_WIN = cfg.VOLUME_MA_WIN
    _MACD_F  = cfg.MACD_FAST
    _MACD_S  = cfg.MACD_SLOW
    _MACD_SG = cfg.MACD_SIGNAL
except ImportError:
    # Fallback defaults so the module works standalone
    _CLIP_LO = -0.15
    _CLIP_HI =  0.15
    _LAGS    = [1, 5, 10]
    _VOL_WIN = 20
    _MACD_F  = 12
    _MACD_S  = 26
    _MACD_SG = 9


# ─────────────────────────────────────────────────────────────────
# 1. Load & clean
# ─────────────────────────────────────────────────────────────────

def load_and_clean(processed_dir: str = _PROC) -> pd.DataFrame:
    """
    Load all parquet files from the processed clean_stocks directory,
    apply the three data-quality fixes, and return a tidy DataFrame.

    Parameters
    ----------
    processed_dir : str
        Path to the partitioned parquet directory
        (default: <project>/data/processed/clean_stocks).

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, volume, daily_return, ticker
        Sorted by ticker then date. No nulls. Returns clipped to ±15 %.
    """
    # ── load all parquet partitions ───────────────────────────────
    pattern = os.path.join(processed_dir, "**", "*.parquet")
    files   = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(
            f"No parquet files found under: {processed_dir}\n"
            "Run the pipeline first:  python run_pipeline.py"
        )

    dfs = []
    for f in files:
        try:
            df     = pd.read_parquet(f)
            ticker = f.split("ticker=")[1].split(os.sep)[0]
            df["ticker"] = ticker
            dfs.append(df)
        except Exception as exc:
            print(f"[data_cleaner] WARNING — skipped {f}: {exc}")

    df = pd.concat(dfs, ignore_index=True)

    # ── parse dates ───────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])

    # ── FIX 1: drop first-row nulls (no prior close on day 1) ────
    n_null = df["daily_return"].isnull().sum()
    if n_null:
        print(f"[data_cleaner] FIX 1 — dropping {n_null} null daily_return rows "
              f"(first trading day per ticker, expected)")
        df = df.dropna(subset=["daily_return"])

    # ── FIX 2: winsorise extreme returns ─────────────────────────
    n_clipped = ((df["daily_return"] < _CLIP_LO) |
                 (df["daily_return"] > _CLIP_HI)).sum()
    if n_clipped:
        print(f"[data_cleaner] FIX 2 — clipping {n_clipped} extreme daily_return "
              f"values to [{_CLIP_LO:.0%}, {_CLIP_HI:.0%}]")
        df["daily_return"] = df["daily_return"].clip(_CLIP_LO, _CLIP_HI)

    # ── sort ──────────────────────────────────────────────────────
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"[data_cleaner] Loaded {len(df):,} rows | "
          f"{df['ticker'].nunique()} tickers | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    return df


# ─────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────

def _macd(series: pd.Series, fast: int, slow: int, signal: int):
    """Return (macd_line, signal_line, histogram) for a price series."""
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD, volume ratio, and lag-return features to the cleaned DataFrame.
    All calculations are done per-ticker to avoid look-ahead leakage.

    New columns added
    -----------------
    macd              : MACD line  (EMA12 – EMA26)
    macd_signal_line  : 9-day EMA of MACD line
    macd_histogram    : macd – macd_signal_line
    volume_ma20       : 20-day rolling mean of volume
    volume_ratio      : volume / volume_ma20  (>1 = above-average activity)
    return_lag_1      : daily_return shifted 1 day
    return_lag_5      : daily_return shifted 5 days
    return_lag_10     : daily_return shifted 10 days

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_and_clean().

    Returns
    -------
    pd.DataFrame
        Original columns + new feature columns.
        Rows with NaN features (warm-up period) are dropped.
    """
    results = []

    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").copy()

        # ── MACD ─────────────────────────────────────────────────
        grp["macd"], grp["macd_signal_line"], grp["macd_histogram"] = _macd(
            grp["close"], _MACD_F, _MACD_S, _MACD_SG
        )

        # ── Volume ratio ─────────────────────────────────────────
        grp["volume_ma20"] = (
            grp["volume"].rolling(_VOL_WIN, min_periods=1).mean()
        )
        grp["volume_ratio"] = grp["volume"] / grp["volume_ma20"]

        # ── Lag returns ──────────────────────────────────────────
        for lag in _LAGS:
            grp[f"return_lag_{lag}"] = grp["daily_return"].shift(lag)

        results.append(grp)

    df_feat = pd.concat(results, ignore_index=True)

    # Drop warm-up rows where lag features are NaN
    lag_cols = [f"return_lag_{lag}" for lag in _LAGS]
    n_before = len(df_feat)
    df_feat  = df_feat.dropna(subset=lag_cols + ["macd"])
    n_dropped = n_before - len(df_feat)
    if n_dropped:
        print(f"[data_cleaner] engineer_features — dropped {n_dropped} warm-up rows "
              f"(MACD/lag NaN period)")

    df_feat = df_feat.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"[data_cleaner] Feature engineering complete — "
          f"{len(df_feat):,} rows, {len(df_feat.columns)} columns")

    return df_feat


# ─────────────────────────────────────────────────────────────────
# 3. Quick audit report (run standalone to verify)
# ─────────────────────────────────────────────────────────────────

def audit(df: pd.DataFrame) -> None:
    """Print a concise data-quality summary to stdout."""
    print("\n" + "=" * 55)
    print("DATA QUALITY AUDIT")
    print("=" * 55)
    print(f"  Rows          : {len(df):,}")
    print(f"  Tickers       : {df['ticker'].nunique()}  "
          f"({', '.join(sorted(df['ticker'].unique()))})")
    print(f"  Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Null values   : {df.isnull().sum().sum()}")
    print(f"  Duplicates    : {df.duplicated(subset=['ticker','date']).sum()}")

    ret = df["daily_return"]
    print(f"\n  Daily return  : mean={ret.mean():.4f}  std={ret.std():.4f}")
    print(f"                  min={ret.min():.4f}  max={ret.max():.4f}")

    bad_ohlc = (
        (df["high"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"])
    ).sum()
    print(f"  Bad OHLC rows : {bad_ohlc}")
    print(f"  Zero-vol rows : {(df['volume'] == 0).sum()}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────
# Entry point — run as script for a quick sanity check
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean()
    audit(df)
    df = engineer_features(df)
    print(f"\nNew feature columns added:\n"
          f"  {[c for c in df.columns if c not in ['date','open','high','low','close','volume','daily_return','ticker']]}")
    print(f"\nSample row:\n{df.iloc[30].to_string()}")