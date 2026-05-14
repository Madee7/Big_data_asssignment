"""
etl/ingest.py
-------------
Stage 1 — Data Ingestion (yfinance).

Downloads historical stock data from Yahoo Finance via yfinance,
then loads it into a PySpark DataFrame.

FIX vs original
---------------
  The original code called yfinance inside Spark worker tasks, causing
  "Python worker exited unexpectedly (EOFException)" crashes.

  The fix: ALL downloading happens in the driver process (plain pandas).
  Spark is only used at the end to create the DataFrame from the finished
  pandas result. No Spark UDFs, no worker serialisation, no crashes.
"""

import os
import sys
import logging
import glob
from time import sleep

import pandas as pd
import yfinance as yf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Download from yfinance  (driver-only, no Spark workers)
# -----------------------------------------------------------------

def _download_yfinance() -> pd.DataFrame:
    """
    Download all tickers from yfinance sequentially in the driver process.
    Returns a clean pandas DataFrame ready to be handed to Spark.
    """
    log.info(f"Downloading {len(cfg.TICKERS)} tickers from yfinance "
             f"({cfg.START_DATE} -> {cfg.END_DATE}) ...")

    dfs = []
    for i, ticker in enumerate(cfg.TICKERS, 1):
        try:
            pdf = yf.download(
                ticker,
                start=cfg.START_DATE,
                end=cfg.END_DATE,
                progress=False,
                auto_adjust=True,
                ignore_tz=True,
            )

            if pdf.empty:
                log.warning(f"  [{i}/{len(cfg.TICKERS)}] {ticker}: no data returned")
                continue

            if isinstance(pdf.columns, pd.MultiIndex):
                pdf.columns = pdf.columns.get_level_values(0)

            pdf = pdf.reset_index()
            pdf.columns = [c.lower().strip().replace(" ", "_") for c in pdf.columns]
            pdf["ticker"] = ticker

            keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
            pdf  = pdf[[c for c in keep if c in pdf.columns]]

            dfs.append(pdf)
            log.info(f"  [{i}/{len(cfg.TICKERS)}] {ticker}: {len(pdf)} rows")

            if i < len(cfg.TICKERS):
                sleep(0.5)

        except Exception as exc:
            log.error(f"  [{i}/{len(cfg.TICKERS)}] {ticker}: download failed -- {exc}")

    if not dfs:
        raise ValueError(
            "No data downloaded from yfinance.\n"
            "Check your internet connection, ticker list, and date range in config.py."
        )

    master = pd.concat(dfs, ignore_index=True)

    master["date"]   = pd.to_datetime(master["date"], errors="coerce")
    master["ticker"] = master["ticker"].astype(str).str.upper().str.strip()

    for col in ["open", "high", "low", "close"]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")

    if "volume" in master.columns:
        master["volume"] = (
            pd.to_numeric(master["volume"], errors="coerce")
            .fillna(0).astype("int64")
        )

    master = master.dropna(subset=["close", "date"]).reset_index(drop=True)

    os.makedirs(cfg.RAW_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.RAW_DIR, "yfinance_cache.parquet")

    # Remove stale cache so Spark always reads a fresh schema
    if os.path.exists(cache_path):
        os.remove(cache_path)

    master.to_parquet(cache_path, index=False, coerce_timestamps="ms", allow_truncated_timestamps=True)
    log.info(f"  Cached {len(master):,} rows -> {cache_path}")

    return master


# -----------------------------------------------------------------
# Load local CSV / cache fallback  (--skip-download)
# -----------------------------------------------------------------

def _load_local_csv() -> pd.DataFrame:
    cache_path = os.path.join(cfg.RAW_DIR, "yfinance_cache.parquet")
    if os.path.exists(cache_path):
        log.info(f"Loading cached yfinance data from {cache_path}")
        return pd.read_parquet(cache_path)

    csv_files = sorted(glob.glob(os.path.join(cfg.RAW_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV or cached parquet found in {cfg.RAW_DIR}.\n"
            "Run without --skip-download to fetch fresh data."
        )

    log.info(f"Loading local CSVs from {cfg.RAW_DIR}/*.csv")
    dfs = []
    for fpath in csv_files:
        pdf = pd.read_csv(fpath, low_memory=False)
        pdf.columns = [c.lower().strip().replace(" ", "_") for c in pdf.columns]
        if "name" in pdf.columns and "ticker" not in pdf.columns:
            pdf = pdf.rename(columns={"name": "ticker"})
        if "ticker" not in pdf.columns:
            pdf["ticker"] = os.path.splitext(os.path.basename(fpath))[0].upper()
        dfs.append(pdf)

    master = pd.concat(dfs, ignore_index=True)
    master["date"]   = pd.to_datetime(master["date"], errors="coerce")
    master["ticker"] = master["ticker"].astype(str).str.upper().str.strip()
    master = master[master["ticker"].isin(cfg.TICKERS)]

    start = pd.to_datetime(cfg.START_DATE)
    end   = pd.to_datetime(cfg.END_DATE)
    master = master[(master["date"] >= start) & (master["date"] <= end)]

    return master.dropna(subset=["close", "date"]).reset_index(drop=True)


# -----------------------------------------------------------------
# pandas -> Spark
# -----------------------------------------------------------------

def _to_spark(spark: SparkSession, master: pd.DataFrame) -> DataFrame:
    # FIX: createDataFrame() crashes on Windows with datetime columns (Arrow bug).
    # Write to parquet first (already cached), then read back with spark.read.parquet()
    # which bypasses the Arrow serialisation issue entirely.
    cache_path = os.path.join(cfg.RAW_DIR, "yfinance_cache.parquet")

    # Ensure cache exists (write if not already there)
    if not os.path.exists(cache_path):
        os.makedirs(cfg.RAW_DIR, exist_ok=True)
        master.to_parquet(cache_path, index=False, coerce_timestamps="ms", allow_truncated_timestamps=True)

    sdf = spark.read.parquet(cache_path)
    sdf = (
        sdf
        .withColumn("date",   F.to_date(F.col("date")))
        .withColumn("open",   F.col("open").cast(DoubleType()))
        .withColumn("high",   F.col("high").cast(DoubleType()))
        .withColumn("low",    F.col("low").cast(DoubleType()))
        .withColumn("close",  F.col("close").cast(DoubleType()))
        .withColumn("volume", F.col("volume").cast(LongType()))
    )
    log.info(f"Loaded {sdf.count():,} rows into Spark.")
    return sdf


# -----------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------

def load_raw_to_spark(spark: SparkSession) -> DataFrame:
    """Load from local files (--skip-download path)."""
    master = _load_local_csv()
    return _to_spark(spark, master)


def run_ingestion(spark: SparkSession, skip_download: bool = False) -> DataFrame:
    log.info("=" * 60)
    log.info("STAGE 1 -- DATA INGESTION (yfinance)")
    log.info("=" * 60)

    if skip_download:
        log.info("--skip-download: loading from local/cached files ...")
        master = _load_local_csv()
    else:
        master = _download_yfinance()

    sdf = _to_spark(spark, master)
    log.info("Ingestion complete.\n")
    return sdf


if __name__ == "__main__":
    from utils.spark_session import get_spark_session
    spark = get_spark_session()
    sdf   = run_ingestion(spark)
    sdf.printSchema()
    sdf.show(5)
    spark.stop()