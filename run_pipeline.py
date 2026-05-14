"""
run_pipeline.py
---------------
Master script — Stock Portfolio Analysis end-to-end pipeline.

Stages
------
  1  Data Ingestion       etl/ingest.py
  2  Transformation       etl/transform.py
  3  Feature Engineering  etl/feature_engineering.py
  4a Portfolio Clustering ml/clustering.py
  4b Return Prediction    ml/return_predictor.py

Usage
-----
  # Full run (downloads fresh data)
  python run_pipeline.py

  # Skip download — reuse existing raw CSVs
  python run_pipeline.py --skip-download

FIXES vs original
-----------------
  FIX 1 — removed stray top-level import of engineer_features
           (it was imported before sys.path was set up, causing ImportError)
  FIX 2 — data_cleaner lives in etl/, so import is  etl.data_cleaner
           (was incorrectly imported as bare  data_cleaner)
  FIX 3 — data_cleaner call moved INSIDE run_full_pipeline(), after
           all lazy imports, so Spark is ready before any data is loaded
"""
import os
os.environ["PYSPARK_PYTHON"] = r"C:\Users\Mo7am\AppData\Local\Programs\Python\Python312\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\Mo7am\AppData\Local\Programs\Python\Python312\python.exe"

import argparse
import json
import logging
import os
import sys
import time

# ── make sure the project root is always on sys.path ─────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def run_full_pipeline(skip_download: bool = False) -> dict:
    # Lazy imports so PySpark initialises only when needed
    import config as cfg
    from utils.spark_session import get_spark_session, stop_spark
    from etl.ingest import run_ingestion, load_raw_to_spark
    from etl.transform import run_transform
    from etl.feature_engineering import run_feature_engineering
    from ml.clustering import run_clustering
    from ml.return_predictor import run_return_predictor

    # # FIX 2: correct import path — data_cleaner.py lives inside etl/
    # from etl.data_cleaner import load_and_clean, engineer_features

    # # FIX 3: data loading now happens here, after sys.path is set up
    # df = engineer_features(load_and_clean())

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   STOCK PORTFOLIO ANALYSIS — END-TO-END PIPELINE     ║")
    log.info("╚══════════════════════════════════════════════════════╝\n")

    t_total = time.time()
    spark   = get_spark_session()

    # ── Stage 1 ───────────────────────────────────────────────────
    t0 = time.time()
    if skip_download:
        log.info("--skip-download: loading existing raw CSVs ...")
        raw_sdf = load_raw_to_spark(spark)
    else:
        raw_sdf = run_ingestion(spark)
    log.info(f"Stage 1 finished in {time.time() - t0:.1f}s\n")

    # ── Stage 2 ───────────────────────────────────────────────────
    t0 = time.time()
    clean_sdf = run_transform(spark, raw_sdf)
    log.info(f"Stage 2 finished in {time.time() - t0:.1f}s\n")

    # ── Stage 3 ───────────────────────────────────────────────────
    t0 = time.time()
    features_sdf = run_feature_engineering(spark, clean_sdf)
    log.info(f"Stage 3 finished in {time.time() - t0:.1f}s\n")

    # ── Stage 4a ──────────────────────────────────────────────────
    t0 = time.time()
    ticker_profile, cluster_stats = run_clustering(features_sdf)
    log.info(f"Stage 4a finished in {time.time() - t0:.1f}s\n")

    # ── Stage 4b ──────────────────────────────────────────────────
    t0 = time.time()
    rf_metrics = run_return_predictor(features_sdf)
    log.info(f"Stage 4b finished in {time.time() - t0:.1f}s\n")

    # ── Summary ───────────────────────────────────────────────────
    # ── Summary ───────────────────────────────────────────────────
    summary = {
        "pipeline_duration_seconds": round(time.time() - t_total, 1),
        "tickers_processed":         len(ticker_profile),
        "clustering": {
            "n_clusters":       cluster_stats["n_clusters"],
            "silhouette_score": cluster_stats["silhouette_score"],
            "distribution":     cluster_stats["cluster_sizes"],
        },
        "return_prediction": {  # <-- MATCHES DASHBOARD EXPECTATION
            "winning_model": rf_metrics["winning_model"],
            "accuracy":      rf_metrics["accuracy"],
            "f1_score":      rf_metrics["f1_score"],
            "cv_accuracy":   rf_metrics["cv_accuracy"],
            "forward_days":  rf_metrics["forward_days"]
        }
    }    # Persist summary JSON
    summary_path = os.path.join(cfg.PROCESSED_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║                PIPELINE COMPLETE ✓                   ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info(f"Total time : {summary['pipeline_duration_seconds']}s")
    log.info(f"Summary    : {summary_path}")
    log.info("\nNext step  : streamlit run dashboard/app.py\n")

    stop_spark(spark)
    return summary


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Portfolio Analysis Pipeline"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip yfinance download and reload existing raw CSVs.",
    )
    args = parser.parse_args()
    run_full_pipeline(skip_download=args.skip_download)