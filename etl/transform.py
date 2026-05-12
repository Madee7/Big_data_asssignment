"""
etl/transform.py
----------------
Stage 2 — Data Cleaning & Transformation (PySpark).

Steps
-----
1. Cast numeric columns to DoubleType
2. Drop rows where close price or date is null
3. Remove duplicate (date, ticker) pairs
4. Compute daily percentage return per ticker
5. Remove extreme return outliers (data errors / un-adjusted splits)
6. Sort and repartition for efficient downstream reads
7. Write clean data to Parquet (partitioned by ticker)
"""

import os
import logging

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Parquet output path
CLEAN_PARQUET = os.path.join(cfg.PROCESSED_DIR, "clean_stocks")


# ─────────────────────────────────────────────────────────────────
# Cleaning steps
# ─────────────────────────────────────────────────────────────────
def _cast_numerics(sdf: DataFrame) -> DataFrame:
    for col in ["open", "high", "low", "close"]:
        sdf = sdf.withColumn(col, F.col(col).cast(DoubleType()))
    return sdf.withColumn("volume", F.col("volume").cast("long"))


def _remove_nulls(sdf: DataFrame) -> DataFrame:
    before = sdf.count()
    sdf    = sdf.dropna(subset=["date", "ticker", "close"])
    log.info(f"  Null removal: {before - sdf.count():,} rows dropped")
    return sdf


def _remove_duplicates(sdf: DataFrame) -> DataFrame:
    before = sdf.count()
    sdf    = sdf.dropDuplicates(["date", "ticker"])
    log.info(f"  Deduplication: {before - sdf.count():,} duplicates removed")
    return sdf


def _add_daily_return(sdf: DataFrame) -> DataFrame:
    """Per-ticker daily return: (close_t - close_t-1) / close_t-1."""
    w = Window.partitionBy("ticker").orderBy("date")
    return (
        sdf
        .withColumn("prev_close", F.lag("close", 1).over(w))
        .withColumn(
            "daily_return",
            F.when(
                F.col("prev_close").isNotNull() & (F.col("prev_close") != 0),
                (F.col("close") - F.col("prev_close")) / F.col("prev_close"),
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        .drop("prev_close")
    )


def _remove_return_outliers(
    sdf: DataFrame,
    lower: float = -0.5,
    upper: float = 0.5,
) -> DataFrame:
    """Drop rows where |daily_return| > 50% (likely data errors)."""
    before = sdf.count()
    sdf = sdf.filter(
        F.col("daily_return").isNull() |
        F.col("daily_return").between(lower, upper)
    )
    log.info(f"  Outlier removal: {before - sdf.count():,} extreme returns removed")
    return sdf


def _sort_and_repartition(sdf: DataFrame) -> DataFrame:
    return sdf.repartition("ticker").sortWithinPartitions("date")


# ─────────────────────────────────────────────────────────────────
# Parquet I/O
# ─────────────────────────────────────────────────────────────────
def write_clean_parquet(sdf: DataFrame, path: str = CLEAN_PARQUET) -> None:
    log.info(f"  Writing clean data → {path}")
    sdf.write.mode("overwrite").partitionBy("ticker").parquet(path)


def read_clean_parquet(spark: SparkSession, path: str = CLEAN_PARQUET) -> DataFrame:
    return spark.read.parquet(path)


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────
def run_transform(spark: SparkSession, raw_sdf: DataFrame) -> DataFrame:
    """
    Full transformation stage.

    Parameters
    ----------
    spark   : SparkSession
    raw_sdf : Raw ingested PySpark DataFrame

    Returns
    -------
    DataFrame
        Cleaned PySpark DataFrame written to Parquet.
    """
    log.info("=" * 60)
    log.info("STAGE 2 — TRANSFORMATION & CLEANING")
    log.info("=" * 60)

    sdf = raw_sdf
    sdf = _cast_numerics(sdf)
    sdf = _remove_nulls(sdf)
    sdf = _remove_duplicates(sdf)
    sdf = _add_daily_return(sdf)
    sdf = _remove_return_outliers(sdf)
    sdf = _sort_and_repartition(sdf)

    write_clean_parquet(sdf)
    log.info("Transformation complete.\n")
    return sdf


if __name__ == "__main__":
    from utils.spark_session import get_spark_session
    from etl.ingest import run_ingestion

    spark = get_spark_session()
    raw   = run_ingestion(spark)
    clean = run_transform(spark, raw)
    clean.printSchema()
    clean.show(5)
    spark.stop()
