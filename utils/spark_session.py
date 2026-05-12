"""
utils/spark_session.py
----------------------
Factory for creating and reusing a PySpark SparkSession.
"""

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "StockPortfolioPipeline") -> SparkSession:
    """
    Create or retrieve an existing SparkSession.

    Args:
        app_name: Name shown in the Spark UI.

    Returns:
        Active SparkSession.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")        # tuned for local dev
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "true")        # Adaptive Query Execution
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def stop_spark(spark: SparkSession) -> None:
    """Gracefully stop the SparkSession."""
    if spark:
        spark.stop()
        print("[Spark] Session stopped.")
