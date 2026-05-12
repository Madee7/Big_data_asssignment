"""
ml/return_predictor.py
----------------------
Stage 4b — Forward Return Prediction (CLASSIFICATION TOURNAMENT).

Pivots from Regression (exact price prediction) to Classification (Directional).
Trains BOTH a Random Forest and a Gradient Boosting Classifier 
to predict if the 10-day forward return will be > 0 (Profitable).

Outputs (saved to models/)
--------------------------
    rf_return_model.joblib      The winning trained Classifier
    rf_scaler.joblib            StandardScaler fitted on predictor features
    feature_importances.csv     feature name + importance score, sorted desc
"""

import os
import logging
import joblib

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report
from pyspark.sql import DataFrame

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RF_MODEL_PATH    = os.path.join(cfg.MODELS_DIR, "rf_return_model.joblib")
RF_SCALER_PATH   = os.path.join(cfg.MODELS_DIR, "rf_scaler.joblib")
FEATURE_IMP_PATH = os.path.join(cfg.MODELS_DIR, "feature_importances.csv")

TARGET_COL = "forward_return"
CLASS_COL  = "is_profitable"


# ─────────────────────────────────────────────────────────────────
# Step 1 — convert Spark features → clean Pandas ML dataset
# ─────────────────────────────────────────────────────────────────
def prepare_ml_dataset(features_sdf: DataFrame) -> pd.DataFrame:
    spark_cols  = set(features_sdf.columns)
    missing     = [f for f in cfg.PREDICTOR_FEATURES if f not in spark_cols]
    if missing:
        raise ValueError(
            f"The following features are listed in cfg.PREDICTOR_FEATURES "
            f"but are NOT present in the Spark DataFrame.\n"
            f"Missing: {missing}\n"
            f"Make sure etl/feature_engineering.py computes these columns."
        )

    cols      = ["ticker", "date"] + cfg.PREDICTOR_FEATURES + [TARGET_COL]
    available = [c for c in cols if c in features_sdf.columns]

    pdf = features_sdf.select(available).toPandas()
    pdf = pdf.dropna(subset=cfg.PREDICTOR_FEATURES + [TARGET_COL])
    
    # NEW: Convert continuous return into binary Classification (1 = Profit, 0 = Loss)
    pdf[CLASS_COL] = (pdf[TARGET_COL] > 0.0).astype(int)
    
    # We can drop the exact float target now
    pdf = pdf.drop(columns=[TARGET_COL])

    log.info(f"  ML dataset: {len(pdf):,} samples  "
             f"{len(cfg.PREDICTOR_FEATURES)} features")
    return pdf


# ─────────────────────────────────────────────────────────────────
# Step 2 — Train and Select the Best Model
# ─────────────────────────────────────────────────────────────────
def _walk_forward_split(pdf: pd.DataFrame, test_size: float = cfg.TEST_SIZE):
    pdf_sorted = pdf.sort_values("date").reset_index(drop=True)
    split_idx  = int(len(pdf_sorted) * (1 - test_size))
    train_df   = pdf_sorted.iloc[:split_idx]
    test_df    = pdf_sorted.iloc[split_idx:]
    log.info(f"  Walk-forward split — train: {len(train_df):,} rows "
             f"({train_df['date'].min()} → {train_df['date'].max()})  "
             f"test: {len(test_df):,} rows "
             f"({test_df['date'].min()} → {test_df['date'].max()})")
    return train_df, test_df


def train_return_predictor(pdf: pd.DataFrame) -> dict:
    train_df, test_df = _walk_forward_split(pdf)

    X_train = train_df[cfg.PREDICTOR_FEATURES].values
    y_train = train_df[CLASS_COL].values
    X_test  = test_df[cfg.PREDICTOR_FEATURES].values
    y_test  = test_df[CLASS_COL].values

    # Check baseline (if the market always goes up, what is our "dumb" accuracy?)
    baseline_acc = sum(y_test == 1) / len(y_test)
    log.info(f"  Baseline Test Accuracy (Always guessing UP): {baseline_acc*100:.2f}%")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- Contender 1: Random Forest Classifier ---
    log.info(f"  Training Contender 1: Random Forest Classifier ...")
    rf = RandomForestClassifier(
        n_estimators=cfg.N_ESTIMATORS,
        max_depth=cfg.MAX_DEPTH,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=cfg.RANDOM_STATE,
        class_weight="balanced", # Helps if data is skewed
        n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)
    rf_pred = rf.predict(X_test_sc)
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_f1   = f1_score(y_test, rf_pred)


    # --- Contender 2: Gradient Boosting Classifier ---
    log.info(f"  Training Contender 2: Gradient Boosting Classifier ...")
    gbt = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.01, 
        max_depth=3,        
        subsample=0.8,      # Row sampling to reduce variance
        max_features="sqrt",
        random_state=cfg.RANDOM_STATE,
    )
    gbt.fit(X_train_sc, y_train)
    gbt_pred = gbt.predict(X_test_sc)
    gbt_acc  = accuracy_score(y_test, gbt_pred)
    gbt_f1   = f1_score(y_test, gbt_pred)

    # --- The Tournament: Who Won? ---
    log.info(f"  [SCORES] RF  Accuracy: {rf_acc*100:.2f}%  |  F1: {rf_f1:.4f}")
    log.info(f"  [SCORES] GBT Accuracy: {gbt_acc*100:.2f}%  |  F1: {gbt_f1:.4f}")

    if gbt_acc > rf_acc:
        log.info("  🏆 WINNER: Gradient Boosting!")
        best_model = gbt
        best_acc   = gbt_acc
        best_f1    = gbt_f1
        winner_str = "Gradient Boosting Classifier"
        best_pred  = gbt_pred
    else:
        log.info("  🏆 WINNER: Random Forest!")
        best_model = rf
        best_acc   = rf_acc
        best_f1    = rf_f1
        winner_str = "Random Forest Classifier"
        best_pred  = rf_pred

    # Print a quick report for the winner
    log.info(f"\n{classification_report(y_test, best_pred, target_names=['Loss (0)', 'Profit (1)'])}")

    # --- Evaluate the Winner via Cross Validation ---
    cv_raw  = cross_val_score(best_model, X_train_sc, y_train, cv=5, scoring="accuracy")
    cv_acc  = float(cv_raw.mean())
    log.info(f"  Winner CV Accuracy = {cv_acc*100:.2f}% ± {cv_raw.std()*100:.2f}%")

    importances = (
        pd.DataFrame({
            "feature":    cfg.PREDICTOR_FEATURES,
            "importance": best_model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(FEATURE_IMP_PATH, index=False)

    joblib.dump(best_model, RF_MODEL_PATH)
    joblib.dump(scaler, RF_SCALER_PATH)
    log.info(f"  Artefacts saved → {cfg.MODELS_DIR}")

    return {
        "winning_model":       winner_str,
        "accuracy":            round(best_acc, 4),
        "f1_score":            round(best_f1, 4),
        "cv_accuracy":         round(cv_acc, 4),
        "baseline_accuracy":   round(baseline_acc, 4),
        "feature_importances": importances,
        "n_train":             len(X_train),
        "n_test":              len(X_test),
        "forward_days":        cfg.FORWARD_DAYS,
    }


# ─────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────
def predict_forward_return(feature_row: dict) -> float:
    # Note: Streamlit might still look for a float. The classifier outputs 0 or 1.
    rf     = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(RF_SCALER_PATH)
    X      = np.array([[feature_row.get(f, 0.0) for f in cfg.PREDICTOR_FEATURES]])
    prediction = float(rf.predict(scaler.transform(X))[0])
    return prediction # Will return 1.0 (UP) or 0.0 (DOWN)

def load_feature_importances() -> pd.DataFrame:
    if not os.path.exists(FEATURE_IMP_PATH):
        raise FileNotFoundError("Run run_pipeline.py first.")
    return pd.read_csv(FEATURE_IMP_PATH)


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────
def run_return_predictor(features_sdf: DataFrame) -> dict:
    log.info("=" * 60)
    log.info("STAGE 4b — RETURN PREDICTION (CLASSIFICATION TOURNAMENT)")
    log.info("=" * 60)

    pdf     = prepare_ml_dataset(features_sdf)
    metrics = train_return_predictor(pdf)

    log.info(f"\n  BEST MODEL : {metrics['winning_model']}")
    log.info(f"  ACCURACY   : {metrics['accuracy']*100:.2f}% (Baseline: {metrics['baseline_accuracy']*100:.2f}%)")
    log.info(f"  F1-SCORE   : {metrics['f1_score']:.4f}")
    log.info(f"  CV ACCURACY: {metrics['cv_accuracy']*100:.2f}%")
    log.info("Return prediction complete.\n")
    return metrics


if __name__ == "__main__":
    from utils.spark_session import get_spark_session
    from etl.ingest import run_ingestion
    from etl.transform import run_transform
    from etl.feature_engineering import run_feature_engineering

    spark    = get_spark_session()
    raw      = run_ingestion(spark)
    clean    = run_transform(spark, raw)
    features = run_feature_engineering(spark, clean)
    metrics  = run_return_predictor(features)
    print(metrics["feature_importances"].to_string(index=False))
    spark.stop()