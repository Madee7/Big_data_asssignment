"""
ml/clustering.py
----------------
Stage 4a — Portfolio Clustering (K-Means).

Clusters stock tickers into 4 portfolio risk profiles:
    Conservative | Balanced | Growth | Aggressive

Cluster features
----------------
    annualised_return   average daily return × 252
    volatility          annualised std of daily returns
    sharpe_ratio        risk-adjusted return approximation
    beta                covariance with SPY / SPY variance

Outputs (saved to models/)
--------------------------
    kmeans_model.joblib     trained KMeans instance
    cluster_scaler.joblib   StandardScaler fitted on cluster features
    cluster_results.csv     one row per ticker with cluster label
"""

import os
import logging
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Saved artefact paths
CLUSTER_MODEL_PATH  = os.path.join(cfg.MODELS_DIR, "kmeans_model.joblib")
CLUSTER_SCALER_PATH = os.path.join(cfg.MODELS_DIR, "cluster_scaler.joblib")
CLUSTER_CSV_PATH    = os.path.join(cfg.MODELS_DIR, "cluster_results.csv")

# Columns used by the clustering model
CLUSTER_FEATURES = [
    "annualised_return",
    "volatility",
    "sharpe_ratio",
    "beta",
]


# ─────────────────────────────────────────────────────────────────
# Step 1 — aggregate feature DataFrame → one row per ticker
# ─────────────────────────────────────────────────────────────────
def build_ticker_profile(features_sdf: DataFrame) -> pd.DataFrame:
    """
    Aggregate per-row features to a single profile row per ticker.

    Returns
    -------
    pd.DataFrame
        One row per ticker with CLUSTER_FEATURES + auxiliary columns.
    """
    agg = (
        features_sdf
        .groupBy("ticker")
        .agg(
            (F.avg("daily_return") * 252).alias("annualised_return"),
            F.avg("volatility").alias("volatility"),
            F.avg("sharpe_ratio").alias("sharpe_ratio"),
            F.avg("beta").alias("beta"),
            F.avg("rsi").alias("avg_rsi"),
            F.avg("bb_pct_b").alias("avg_bb_pct_b"),
            F.count("date").alias("trading_days"),
        )
    )
    pdf = agg.toPandas().dropna(subset=CLUSTER_FEATURES)
    log.info(f"  Ticker profiles built: {len(pdf)}")
    return pdf


# ─────────────────────────────────────────────────────────────────
# Step 2 — find optimal k via silhouette score
# ─────────────────────────────────────────────────────────────────
def _find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 8)) -> dict:
    scores = {}
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = round(silhouette_score(X_scaled, labels), 4)
        log.info(f"    k={k}  silhouette={scores[k]:.4f}")
    return scores


# ─────────────────────────────────────────────────────────────────
# Step 3 — train K-Means and assign semantic labels
# ─────────────────────────────────────────────────────────────────
def train_kmeans(ticker_profile: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Scale features, run K-Means, assign human-readable cluster labels
    sorted by a composite risk score.

    Returns
    -------
    ticker_profile : pd.DataFrame
        Original profile DataFrame with cluster_id, cluster_label,
        cluster_color columns added.
    stats : dict
        Silhouette scores, label mapping, cluster size distribution.
    """
    X        = ticker_profile[CLUSTER_FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Silhouette sweep
    log.info("  Silhouette sweep ...")
    sil_scores = _find_optimal_k(X_scaled)
    best_k     = max(sil_scores, key=sil_scores.get)
    log.info(f"  Best k by silhouette = {best_k} "
             f"(configured N_CLUSTERS = {cfg.N_CLUSTERS})")

    # Train with the configured k (allows override)
    kmeans = KMeans(
        n_clusters=cfg.N_CLUSTERS,
        random_state=cfg.RANDOM_STATE,
        n_init=20,
        max_iter=500,
    )
    cluster_ids = kmeans.fit_predict(X_scaled)

    ticker_profile = ticker_profile.copy()
    ticker_profile["cluster_id"] = cluster_ids

    # ── Assign semantic labels by risk score ──────────────────────
    # risk_score = volatility + max(beta, 0) − 0.3 × clipped_sharpe
    cluster_means = (
        ticker_profile
        .groupby("cluster_id")[CLUSTER_FEATURES]
        .mean()
        .reset_index()
    )
    cluster_means["risk_score"] = (
        cluster_means["volatility"]
        + cluster_means["beta"].clip(lower=0)
        - cluster_means["sharpe_ratio"].clip(-2, 5) * 0.3
    )
    cluster_means = cluster_means.sort_values("risk_score").reset_index(drop=True)

    rank_to_label = {
        row["cluster_id"]: cfg.CLUSTER_LABELS[i]
        for i, row in cluster_means.iterrows()
    }
    ticker_profile["cluster_label"] = ticker_profile["cluster_id"].map(rank_to_label)
    ticker_profile["cluster_color"] = ticker_profile["cluster_label"].map(cfg.CLUSTER_COLORS)

    # Final silhouette
    final_sil = silhouette_score(X_scaled, cluster_ids)
    log.info(f"  Final silhouette (k={cfg.N_CLUSTERS}): {final_sil:.4f}")

    # ── Persist artefacts ─────────────────────────────────────────
    joblib.dump(kmeans, CLUSTER_MODEL_PATH)
    joblib.dump(scaler, CLUSTER_SCALER_PATH)
    ticker_profile.to_csv(CLUSTER_CSV_PATH, index=False)
    log.info(f"  Artefacts saved → {cfg.MODELS_DIR}")

    stats = {
        "n_clusters":        cfg.N_CLUSTERS,
        "silhouette_score":  round(final_sil, 4),
        "silhouette_by_k":   sil_scores,
        "cluster_label_map": rank_to_label,
        "cluster_sizes":     ticker_profile["cluster_label"].value_counts().to_dict(),
    }
    return ticker_profile, stats


# ─────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────
def load_cluster_results() -> pd.DataFrame:
    if not os.path.exists(CLUSTER_CSV_PATH):
        raise FileNotFoundError(
            f"Run run_pipeline.py first — {CLUSTER_CSV_PATH} not found."
        )
    return pd.read_csv(CLUSTER_CSV_PATH)


def predict_cluster(ticker_stats: dict) -> str:
    """
    Predict the cluster label for a new ticker given its feature dict.

    Parameters
    ----------
    ticker_stats : dict
        Keys must match CLUSTER_FEATURES.

    Returns
    -------
    str
        Cluster label (e.g. "Balanced").
    """
    kmeans  = joblib.load(CLUSTER_MODEL_PATH)
    scaler  = joblib.load(CLUSTER_SCALER_PATH)
    X       = np.array([[ticker_stats[f] for f in CLUSTER_FEATURES]])
    cid     = kmeans.predict(scaler.transform(X))[0]
    results = load_cluster_results()
    label_map = dict(zip(results["cluster_id"], results["cluster_label"]))
    return label_map.get(cid, "Unknown")


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────
def run_clustering(features_sdf: DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full clustering stage.

    Parameters
    ----------
    features_sdf : DataFrame
        Enriched PySpark DataFrame from feature engineering.

    Returns
    -------
    ticker_profile : pd.DataFrame
        One row per ticker with cluster labels.
    stats : dict
        Silhouette scores and cluster size distribution.
    """
    log.info("=" * 60)
    log.info("STAGE 4a — PORTFOLIO CLUSTERING (K-MEANS)")
    log.info("=" * 60)

    ticker_profile        = build_ticker_profile(features_sdf)
    ticker_profile, stats = train_kmeans(ticker_profile)

    log.info("\n  Cluster distribution:")
    for label, count in stats["cluster_sizes"].items():
        log.info(f"    {label:15s}: {count}")

    log.info("Clustering complete.\n")
    return ticker_profile, stats


if __name__ == "__main__":
    from utils.spark_session import get_spark_session
    from etl.ingest import run_ingestion
    from etl.transform import run_transform
    from etl.feature_engineering import run_feature_engineering

    spark    = get_spark_session()
    raw      = run_ingestion(spark)
    clean    = run_transform(spark, raw)
    features = run_feature_engineering(spark, clean)
    profile, stats = run_clustering(features)
    print(profile[["ticker", "cluster_label",
                   "annualised_return", "volatility", "sharpe_ratio"]].to_string())
    spark.stop()
