"""
etl/feature_engineering.py
---------------------------
Stage 3 — Feature Engineering (PySpark).

Indicators computed
-------------------
Price / trend
    MA7, MA30              — simple moving averages
    ma_cross_signal        — 1 (bullish) / -1 (bearish)

Momentum & Alpha (NEW)
    RSI (14-day)           — relative strength index
    bb_pct_b               — Bollinger %B position
    MACD / signal / hist   — Disabled to save memory
    momentum_14d           — 14-day rate of change
    drawdown_30d           — Distance from the 30-day high
    day_of_week            — Seasonality identifier

ETL ML Improvements
    ma_spread              — Distance between MA7 and MA30
    rsi_binned             — 0 (Oversold), 1 (Neutral), 2 (Overbought)

Volume
    volume_ma20            — 20-day rolling mean volume
    volume_ratio           — volume / volume_ma20

Lag returns
    return_lag_1/5/10      — daily_return shifted N days

Risk
    volatility             — annualised rolling std of daily returns
    sharpe_ratio           — (ann. return − risk-free) / ann. volatility
    beta                   — covariance with benchmark / benchmark variance

ML target
    forward_return         — N-day (default 10) forward price return
"""

import os
import logging
import sys

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURES_PARQUET = os.path.join(cfg.PROCESSED_DIR, "features_stocks")


# ─────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────

def _add_moving_averages(sdf: DataFrame) -> DataFrame:
    s  = cfg.SHORT_WINDOW
    l  = cfg.LONG_WINDOW
    ws = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(s - 1), 0)
    wl = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(l - 1), 0)
    return (
        sdf
        .withColumn(f"ma{s}", F.avg("close").over(ws))
        .withColumn(f"ma{l}", F.avg("close").over(wl))
        .withColumn(
            "ma_cross_signal",
            F.when(F.col(f"ma{s}") > F.col(f"ma{l}"), 1.0).otherwise(-1.0),
        )
    )

def _add_rsi(sdf: DataFrame) -> DataFrame:
    period = cfg.RSI_PERIOD
    sdf = (
        sdf
        .withColumn("gain", F.when(F.col("daily_return") > 0, F.col("daily_return")).otherwise(0.0))
        .withColumn("loss", F.when(F.col("daily_return") < 0, -F.col("daily_return")).otherwise(0.0))
    )
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(period - 1), 0)
    return (
        sdf
        .withColumn("avg_gain", F.avg("gain").over(w))
        .withColumn("avg_loss", F.avg("loss").over(w))
        .withColumn("rs",  F.col("avg_gain") / (F.col("avg_loss") + 1e-10))
        .withColumn("rsi", 100.0 - 100.0 / (1.0 + F.col("rs")))
        .drop("gain", "loss", "avg_gain", "avg_loss", "rs")
    )

def _add_bollinger_bands(sdf: DataFrame) -> DataFrame:
    win = cfg.BOLLINGER_WIN
    std = cfg.BOLLINGER_STD
    w   = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(win - 1), 0)
    return (
        sdf
        .withColumn("bb_mid",   F.avg("close").over(w))
        .withColumn("_bb_std",  F.stddev("close").over(w))
        .withColumn("bb_upper", F.col("bb_mid") + std * F.col("_bb_std"))
        .withColumn("bb_lower", F.col("bb_mid") - std * F.col("_bb_std"))
        .withColumn(
            "bb_pct_b",
            (F.col("close") - F.col("bb_lower"))
            / (F.col("bb_upper") - F.col("bb_lower") + 1e-10),
        )
        .drop("_bb_std")
    )

def _add_volatility(sdf: DataFrame) -> DataFrame:
    w = Window.partitionBy("ticker").orderBy("date") \
              .rowsBetween(-(cfg.LONG_WINDOW - 1), 0)
    return sdf.withColumn(
        "volatility",
        F.stddev("daily_return").over(w) * F.sqrt(F.lit(252.0)),
    )

def _add_sharpe_ratio(sdf: DataFrame, risk_free: float = 0.04) -> DataFrame:
    stats = (
        sdf.groupBy("ticker")
        .agg(
            F.avg("daily_return").alias("_avg_ret"),
            F.stddev("daily_return").alias("_std_ret"),
        )
        .withColumn(
            "sharpe_ratio",
            (F.col("_avg_ret") * 252.0 - risk_free)
            / (F.col("_std_ret") * F.sqrt(F.lit(252.0)) + 1e-10),
        )
        .select("ticker", "sharpe_ratio")
    )
    return sdf.join(stats, on="ticker", how="left")

def _add_beta(sdf: DataFrame) -> DataFrame:
    bench = (
        sdf.filter(F.col("ticker") == cfg.BENCHMARK_TICKER)
           .select("date", F.col("daily_return").alias("bench_return"))
    )
    betas = (
        sdf.join(bench, on="date", how="left")
           .groupBy("ticker")
           .agg(
               F.covar_samp("daily_return", "bench_return").alias("_cov"),
               F.variance("bench_return").alias("_var"),
           )
           .withColumn("beta", F.col("_cov") / (F.col("_var") + 1e-10))
           .select("ticker", "beta")
    )
    # Join bench_return back to the main dataframe
    sdf = sdf.join(bench, on="date", how="left")
    return sdf.join(betas, on="ticker", how="left")

def _add_new_alpha_features(sdf: DataFrame) -> DataFrame:
    """Adds Momentum, Drawdown, and Seasonality features."""
    # 1. 14-day Momentum
    w_lag = Window.partitionBy("ticker").orderBy("date")
    sdf = sdf.withColumn("_close_lag_14", F.lag("close", 14).over(w_lag))
    sdf = sdf.withColumn("momentum_14d", (F.col("close") - F.col("_close_lag_14")) / (F.col("_close_lag_14") + 1e-10))
    sdf = sdf.drop("_close_lag_14")

    # 2. 30-day Maximum Drawdown (Distance from recent high)
    w_30 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-29, 0)
    sdf = sdf.withColumn("_max_30d", F.max("close").over(w_30))
    sdf = sdf.withColumn("drawdown_30d", (F.col("close") - F.col("_max_30d")) / (F.col("_max_30d") + 1e-10))
    sdf = sdf.drop("_max_30d")

    # 3. Day of Week (Seasonality)
    sdf = sdf.withColumn("day_of_week", F.dayofweek("date").cast(DoubleType()))

    return sdf

def _add_etl_improvements(sdf: DataFrame) -> DataFrame:
    """Converts absolute indicators into relative/binned signals for better ML performance."""
    # 1. MA Spread (Relative distance instead of absolute price)
    sdf = sdf.withColumn("ma_spread", (F.col("ma7") - F.col("ma30")) / (F.col("ma30") + 1e-10))
    
    # 2. RSI Binning (0: Oversold, 1: Neutral, 2: Overbought)
    sdf = sdf.withColumn(
        "rsi_binned",
        F.when(F.col("rsi") < 30, 0.0)
         .when(F.col("rsi") > 70, 2.0)
         .otherwise(1.0)
    )
    return sdf

def _add_volume_features(sdf: DataFrame) -> DataFrame:
    win = cfg.VOLUME_MA_WIN   # 20
    w   = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(win - 1), 0)
    return (
        sdf
        .withColumn("volume_ma20",  F.avg(F.col("volume").cast(DoubleType())).over(w))
        .withColumn("volume_ratio", F.col("volume") / (F.col("volume_ma20") + 1e-10))
    )

def _add_lag_returns(sdf: DataFrame) -> DataFrame:
    w = Window.partitionBy("ticker").orderBy("date")
    for lag in cfg.RETURN_LAGS:   # [1, 5, 10]
        sdf = sdf.withColumn(f"return_lag_{lag}", F.lag("daily_return", lag).over(w))
    return sdf

def _add_forward_return(sdf: DataFrame) -> DataFrame:
    days = cfg.FORWARD_DAYS
    w    = Window.partitionBy("ticker").orderBy("date")
    return (
        sdf
        .withColumn("_future_close", F.lead("close", days).over(w))
        .withColumn(
            "forward_return",
            F.when(
                F.col("_future_close").isNotNull(),
                (F.col("_future_close") - F.col("close")) / F.col("close"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
        .drop("_future_close")
    )


# ─────────────────────────────────────────────────────────────────
# Parquet I/O
# ─────────────────────────────────────────────────────────────────

def write_features_parquet(sdf: DataFrame, path: str = FEATURES_PARQUET) -> None:
    log.info(f"  Writing features → {path}")
    sdf.write.mode("overwrite").partitionBy("ticker").parquet(path)

def read_features_parquet(spark: SparkSession, path: str = FEATURES_PARQUET) -> DataFrame:
    return spark.read.parquet(path)


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────

def run_feature_engineering(spark: SparkSession, clean_sdf: DataFrame) -> DataFrame:
    log.info("=" * 60)
    log.info("STAGE 3 — FEATURE ENGINEERING")
    log.info("=" * 60)

    sdf = clean_sdf

    log.info("  Moving averages ...")
    sdf = _add_moving_averages(sdf)

    log.info("  RSI ...")
    sdf = _add_rsi(sdf)

    log.info("  Bollinger Bands ...")
    sdf = _add_bollinger_bands(sdf)

    log.info("  Volatility ...")
    sdf = _add_volatility(sdf)

    log.info("  Sharpe Ratio ...")
    sdf = _add_sharpe_ratio(sdf)

    log.info("  Beta vs benchmark ...")
    sdf = _add_beta(sdf)

    log.info("  New Alpha features (Momentum, Drawdown, Seasonality) ...")
    sdf = _add_new_alpha_features(sdf)

    log.info("  ETL Classifications (Spread & Binning) ...")
    sdf = _add_etl_improvements(sdf)

    log.info("  Volume features ...")
    sdf = _add_volume_features(sdf)

    log.info("  Lag returns ...")
    sdf = _add_lag_returns(sdf)

    log.info("  Forward return (ML target) ...")
    sdf = _add_forward_return(sdf)

    write_features_parquet(sdf)
    log.info("Feature engineering complete.\n")
    return sdf


if __name__ == "__main__":
    from utils.spark_session import get_spark_session
    from etl.ingest import run_ingestion
    from etl.transform import run_transform

    spark    = get_spark_session()
    raw      = run_ingestion(spark)
    clean    = run_transform(spark, raw)
    features = run_feature_engineering(spark, clean)
    features.printSchema()
    features.show(3)
    spark.stop()