"""
etl/ingest.py
-------------
Stage 1 — Data Ingestion (Local File Mode).

Reads historical stock data from local CSV files (e.g., downloaded from Kaggle),
standardizes the column names, extracts ticker symbols, and loads them 
into a PySpark DataFrame.
"""

import os
import logging
import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType

# ── resolve project root so imports work globally ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_raw_to_spark(spark: SparkSession) -> DataFrame:
    """
    Reads all CSVs in the raw data directory and standardizes them 
    into a clean PySpark DataFrame.
    """
    csv_glob = os.path.join(cfg.RAW_DIR, "*.csv")
    log.info(f"Loading local CSVs from {csv_glob}")
    
    # 1. Read all CSVs in the raw directory
    try:
        sdf = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(csv_glob)
        )
    except Exception as e:
        log.error(f"Failed to read CSV files: {e}")
        raise FileNotFoundError(f"Ensure your Kaggle CSV files are inside {cfg.RAW_DIR}")

    # 2. Standardize column names (lowercase, replace spaces with underscores)
    for c in sdf.columns:
        sdf = sdf.withColumnRenamed(c, c.lower().strip().replace(" ", "_"))

    # 3. Use Adjusted Close if the Kaggle dataset provides it
    if "adj_close" in sdf.columns:
        if "close" in sdf.columns:
            sdf = sdf.drop("close")
        sdf = sdf.withColumnRenamed("adj_close", "close")

    # 4. Handle missing Ticker Column (Very common in Kaggle datasets)
    if "ticker" not in sdf.columns:
        if "symbol" in sdf.columns:
            sdf = sdf.withColumnRenamed("symbol", "ticker")
        elif "name" in sdf.columns:
            sdf = sdf.withColumnRenamed("name", "ticker")
        else:
            # If no ticker column exists, extract it from the filename (e.g., "AAPL.csv" -> "AAPL")
            sdf = sdf.withColumn("ticker", F.regexp_extract(F.input_file_name(), r"([^/\\]+)\.csv$", 1))

    # 5. Filter to ONLY process the 24 tickers defined in config.py
    sdf = sdf.filter(F.col("ticker").isin(cfg.TICKERS))

    # 6. Format dates and strictly cast data types 
    # (This prevents the PySpark crash you experienced earlier where Volume was a float)
    sdf = sdf.withColumn("date", F.to_date("date"))
    
    for col_name in ["open", "high", "low", "close"]:
        if col_name in sdf.columns:
            sdf = sdf.withColumn(col_name, F.col(col_name).cast(DoubleType()))
            
    if "volume" in sdf.columns:
        # Force volume to be a whole number (LongType)
        sdf = sdf.withColumn("volume", F.col("volume").cast(LongType()))

    # 7. Select only the required columns
    keep_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    existing_cols = [c for c in keep_cols if c in sdf.columns]
    sdf = sdf.select(*existing_cols)

    log.info(f"Loaded {sdf.count():,} rows into Spark.")
    return sdf


def run_ingestion(spark: SparkSession) -> DataFrame:
    """
    Full ingestion stage entry point.
    """
    log.info("=" * 60)
    log.info("STAGE 1 — DATA INGESTION (LOCAL KAGGLE FILES)")
    log.info("=" * 60)

    # FIXED: The function call name now matches the definition above
    sdf = load_raw_to_spark(spark)

    log.info("Ingestion complete.\n")
    return sdf


# ─────────────────────────────────────────────────────────────────
# Standalone smoke-test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.spark_session import get_spark_session
    spark = get_spark_session()
    sdf   = run_ingestion(spark)
    sdf.printSchema()
    sdf.show(5)
    spark.stop()