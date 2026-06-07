"""
train.py — trains a fraud classifier on the Kaggle credit card fraud dataset.

Dataset: mlg-ulb/creditcardfraud (kaggle)
  284,807 transactions, 492 fraud (~0.17% positive rate)
  Features: V1–V28 (PCA), Amount, Time

Usage:
  # with kaggle API
  pip install kaggle
  kaggle datasets download mlg-ulb/creditcardfraud -p ./data --unzip
  python scripts/train.py

  # or download manually from:
  # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  # place creditcard.csv in ./data/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
import datetime
from datetime import timedelta
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

from dotenv import load_dotenv
load_dotenv()

start = time.time()

root = Path(__file__).parent.parent

# ── core config ───────────────────────────────────────────────────────────────
DATA_PATH    = Path(root / "data" / "creditcard.csv")
MODEL_DIR    = Path(root / "model")
MODEL_PATH   = MODEL_DIR / "fraud_classifier.joblib"
SCALER_PATH  = MODEL_DIR / "scaler.joblib"
AMOUNT_SCALER_PATH = MODEL_DIR / "amount_scaler.joblib"
TIME_SCALER_PATH   = MODEL_DIR / "time_scaler.joblib"
META_PATH    = MODEL_DIR / "metadata.json"

MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
MODEL_ID     = os.getenv("MODEL_ID", "fraud-classifier-v1")
ENVIRONMENT  = os.getenv("ENVIRONMENT", "production")
FULL_TRAIN   = os.getenv("FULL_TRAIN", "false").lower() == "true"
TRIGGERED_BY = os.getenv("TRIGGERED_BY", "manual")

# ── prescription params from remediation agent ────────────────────────────────
# These are forwarded by retrain.yml as env vars from the workflow_dispatch inputs.
# All have sensible defaults so the script works for manual runs too.
DRIFT_DATASET = os.getenv("DRIFT_DATASET", "baseline")
DATA_STRATEGY        = os.getenv("DATA_STRATEGY", "recent_window")
WINDOW_DAYS          = int(os.getenv("WINDOW_DAYS", "30"))
DRIFT_PERIOD_WEIGHT  = float(os.getenv("DRIFT_PERIOD_WEIGHT", "1.5"))
EXCLUDE_BEFORE       = os.getenv("EXCLUDE_BEFORE", "")           # ISO-8601 date or ""
REFIT_PREPROCESSORS  = os.getenv("REFIT_PREPROCESSORS", "true").lower() == "true"
DRIFTED_FEATURES     = json.loads(os.getenv("DRIFTED_FEATURES", "[]"))
OPTIMIZE_FOR         = os.getenv("OPTIMIZE_FOR", "recall")        # recall | f2_score | roc_auc | precision
TARGET_RECALL        = float(os.getenv("TARGET_RECALL", "0.80"))
TARGET_ROC_AUC       = float(os.getenv("TARGET_ROC_AUC", "0.88"))
DEPLOYMENT_STRATEGY  = os.getenv("DEPLOYMENT_STRATEGY", "canary") # canary | blue_green | immediate

MODEL_DIR.mkdir(exist_ok=True)


dataset_path = Path(__file__).parent.parent.parent.parent / "mlops_agents/data/datasets"
if DRIFT_DATASET and Path( dataset_path / f"{DRIFT_DATASET}.csv").exists():
    DATA_PATH = Path( dataset_path / f"{DRIFT_DATASET}.csv")
    print(f"Training on drifted dataset: {DRIFT_DATASET}")
else:
    DATA_PATH = Path( dataset_path / "baseline.csv")

print(f"\n{'='*60}")
print(f"  Fraud Classifier Training")
print(f"{'='*60}")
print(f"  model_id      : {MODEL_ID}")
print(f"  environment   : {ENVIRONMENT}")
print(f"  full_train    : {FULL_TRAIN}")
print(f"  triggered_by  : {TRIGGERED_BY}")
if FULL_TRAIN:
    print(f"\n  Prescription:")
    print(f"    data_strategy       : {DATA_STRATEGY}")
    print(f"    window_days         : {WINDOW_DAYS}")
    print(f"    drift_period_weight : {DRIFT_PERIOD_WEIGHT}")
    print(f"    exclude_before      : {EXCLUDE_BEFORE or 'none'}")
    print(f"    refit_preprocessors : {REFIT_PREPROCESSORS}")
    print(f"    drifted_features    : {DRIFTED_FEATURES or 'all'}")
    print(f"    optimize_for        : {OPTIMIZE_FOR}")
    print(f"    target_recall       : {TARGET_RECALL}")
    print(f"    target_roc_auc      : {TARGET_ROC_AUC}")
    print(f"    deployment_strategy : {DEPLOYMENT_STRATEGY}")
print(f"{'='*60}\n")
# training
print("loading env time:", time.time() - start)

# ── load data ─────────────────────────────────────────────────────────────────
print(f"Loading dataset from {DATA_PATH}...")
start = time.time()
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found.\n"
        "release assets were not downloaded in the retrain workflow - not proceeding with the train script"
    )
df = pd.read_csv(DATA_PATH)
print(f"Dataset head: {df.head()}")
print(f"Dataset columns: {df.columns.tolist()}")
print(f"Dataset shape : {df.shape}")
print(f"Fraud rate    : {df['Class'].mean():.4%}")
# load data
print("Load time:", time.time() - start)

start = time.time()
# ── feature engineering ───────────────────────────────────────────────────────
# V1-V28 are already PCA-transformed. We scale Amount and Time.
# If REFIT_PREPROCESSORS=false (retrain without schema change), reuse the saved scaler.
scaler_fitted = False
if not REFIT_PREPROCESSORS and AMOUNT_SCALER_PATH.exists() and TIME_SCALER_PATH.exists():
    print("Reusing existing scalers (REFIT_PREPROCESSORS=false)")
    amount_scaler = joblib.load(AMOUNT_SCALER_PATH)
    time_scaler = joblib.load(TIME_SCALER_PATH)
    df["Amount_scaled"] = amount_scaler.transform(df[["Amount"]])
    df["Time_scaled"]   = time_scaler.transform(df[["Time"]])
else:
    amount_scaler = StandardScaler()
    time_scaler   = StandardScaler()

    df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = time_scaler.fit_transform(df[["Time"]])
    scaler_fitted = True
    print("Fitted new scaler (REFIT_PREPROCESSORS=true)")

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

# reference distribution for drift computation (saved in metadata)
X_all = df[feature_cols].values
reference_stats = {
    "mean":         X_all.mean(axis=0).tolist(),
    "std":          X_all.std(axis=0).tolist(),
    "feature_cols": feature_cols,
}

# ── synthetic datetime index ──────────────────────────────────────────────────
# The Kaggle dataset has a "Time" column (seconds from first transaction).
# We derive a synthetic datetime so window/drift-period strategies work.
df["datetime"] = pd.to_datetime(df["Time"], unit="s",
                                origin=pd.Timestamp("2024-01-01"))
dataset_end   = df["datetime"].max()
dataset_start = df["datetime"].min()
print(f"Synthetic date range: {dataset_start.date()} → {dataset_end.date()}")

# ── data selection strategy ───────────────────────────────────────────────────
# Applied only for full retrains. Poor initial model always uses the full pool
# (subsampled later by TRAIN_FRAC).

if FULL_TRAIN:
    if DATA_STRATEGY == "recent_window":
        cutoff       = dataset_end - timedelta(days=WINDOW_DAYS)
        df_train_pool = df[df["datetime"] >= cutoff].copy()
        print(f"\nData strategy: recent_window — last {WINDOW_DAYS} days")
        print(f"  cutoff       : {cutoff.date()}")
        print(f"  rows selected: {len(df_train_pool)}")

    elif DATA_STRATEGY == "drift_period_only":
        if EXCLUDE_BEFORE:
            cutoff       = pd.Timestamp(EXCLUDE_BEFORE)
            df_train_pool = df[df["datetime"] >= cutoff].copy()
            print(f"\nData strategy: drift_period_only — from {EXCLUDE_BEFORE}")
        else:
            cutoff       = dataset_end - timedelta(days=WINDOW_DAYS)
            df_train_pool = df[df["datetime"] >= cutoff].copy()
            print(f"\nData strategy: drift_period_only — last {WINDOW_DAYS} days (no exclude_before set)")
        print(f"  rows selected: {len(df_train_pool)}")

    elif DATA_STRATEGY == "weighted_recent":
        # use all data but upsample the recent window
        df_train_pool = df.copy()
        print(f"\nData strategy: weighted_recent — full history + upsampled recent {WINDOW_DAYS}d")

    else:  # full_history
        df_train_pool = df.copy()
        print(f"\nData strategy: full_history — all {len(df)} rows")

    # apply exclude_before as a hard floor on top of any strategy
    if EXCLUDE_BEFORE:
        floor         = pd.Timestamp(EXCLUDE_BEFORE)
        before_count  = len(df_train_pool)
        df_train_pool = df_train_pool[df_train_pool["datetime"] >= floor].copy()
        print(f"  exclude_before filter: removed {before_count - len(df_train_pool)} rows before {EXCLUDE_BEFORE}")

    # drift-period upsampling (weighted_recent or explicit weight > 1.0)
    if DRIFT_PERIOD_WEIGHT > 1.0:
        drift_cutoff   = dataset_end - timedelta(days=WINDOW_DAYS)
        df_drift       = df_train_pool[df_train_pool["datetime"] >= drift_cutoff]
        n_extra        = int(len(df_drift) * (DRIFT_PERIOD_WEIGHT - 1.0))
        if n_extra > 0:
            df_extra       = df_drift.sample(n=n_extra, replace=True, random_state=42)
            df_train_pool  = pd.concat([df_train_pool, df_extra], ignore_index=True)
            print(f"  drift upsampling: added {n_extra} extra rows from last {WINDOW_DAYS}d "
                  f"(weight={DRIFT_PERIOD_WEIGHT})")

    X_pool = df_train_pool[feature_cols].values
    y_pool = df_train_pool["Class"].values
    print(f"  final pool   : {len(X_pool)} rows  |  fraud rate: {y_pool.mean():.4%}")

else:
    # poor initial model — use entire dataset as pool, subsample below
    X_pool = df[feature_cols].values
    y_pool = df["Class"].values

# ── train / test split ────────────────────────────────────────────────────────
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_pool, y_pool,
    test_size=0.20,
    random_state=42,
    stratify=y_pool,
)

if not FULL_TRAIN:
    # subsample to 15% → deliberately underfit / poor model
    TRAIN_FRAC = 0.15
    n_samples  = int(len(X_train_full) * TRAIN_FRAC)
    idx        = np.random.RandomState(42).choice(len(X_train_full), n_samples, replace=False)
    X_train    = X_train_full[idx]
    y_train    = y_train_full[idx]
    print(f"\nPoor model mode: {n_samples} / {len(X_train_full)} samples ({TRAIN_FRAC:.0%})")
else:
    X_train = X_train_full
    y_train = y_train_full
    TRAIN_FRAC = 1.0
    print(f"\nFull retrain: {len(X_train)} training samples")

print(f"Train fraud rate: {y_train.mean():.4%}")
print(f"Test  fraud rate: {y_test.mean():.4%}")

# ── handle class imbalance ────────────────────────────────────────────────────
print("\nApplying SMOTE for class imbalance...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled: {X_resampled.shape}  |  fraud rate: {y_resampled.mean():.4%}")

# ── model selection ───────────────────────────────────────────────────────────
# LogisticRegression for both initial and FULL_TRAIN runs — trains in seconds
# on this dataset and reaches ~0.95+ ROC-AUC with SMOTE-balanced input, which
# is well above the severity thresholds the agent loop checks against.
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    n_jobs=-1,
    random_state=42,
)
# preprocessing
print("Preprocess time:", time.time() - start)


start = time.time()
print(f"\nTraining {model.__class__.__name__}...")
model.fit(X_resampled, y_resampled)
# training
print("Training time:", time.time() - start)

# ── threshold tuning ──────────────────────────────────────────────────────────
# For full retrains the agent prescribes what to optimise for.
# Default threshold 0.5 is wrong for fraud — we optimise per prescription.
y_proba = model.predict_proba(X_test)[:, 1]
precisions_arr, recalls_arr, thresholds_arr = precision_recall_curve(y_test, y_proba)
MIN_PRECISION = float(os.getenv("MIN_PRECISION", "0.10"))  # precision floor for recall optimisation

if FULL_TRAIN:
    if OPTIMIZE_FOR == "recall":
        # Lowest threshold that achieves TARGET_RECALL **while** keeping precision
        # above the floor. The unconditional "lowest threshold for recall>=T" rule
        # collapses to threshold≈0 on imbalanced data (the trivial predict-everything-
        # as-fraud solution gives recall=1.0 but ~0.18% precision). If no threshold
        # satisfies both, fall back to F2 — recall-weighted but precision-aware —
        # so we never deploy a model that flags 100% of traffic.
        valid_mask = (recalls_arr[:-1] >= TARGET_RECALL) & (precisions_arr[:-1] >= MIN_PRECISION)
        valid_thresholds = thresholds_arr[valid_mask]
        if len(valid_thresholds) > 0:
            optimal_threshold = float(valid_thresholds.min())
            print(
                f"\nThreshold optimised for recall >= {TARGET_RECALL} "
                f"with precision >= {MIN_PRECISION}"
            )
        else:
            f2_scores = (5 * precisions_arr * recalls_arr) / \
                        (4 * precisions_arr + recalls_arr + 1e-9)
            optimal_threshold = float(thresholds_arr[np.argmax(f2_scores[:-1])])
            print(
                f"\nNo threshold satisfies recall>={TARGET_RECALL} & precision>={MIN_PRECISION} "
                f"— falling back to F2 maximizer (threshold={optimal_threshold:.4f})"
            )

    elif OPTIMIZE_FOR == "f2_score":
        # F2 weights recall twice as much as precision
        f2_scores         = (5 * precisions_arr * recalls_arr) / \
                            (4 * precisions_arr + recalls_arr + 1e-9)
        optimal_threshold = float(thresholds_arr[np.argmax(f2_scores[:-1])])
        print(f"\nThreshold optimised for F2 score")

    elif OPTIMIZE_FOR == "precision":
        # highest threshold that keeps recall >= 0.5 (floor to avoid degenerate model)
        valid_mask        = recalls_arr[:-1] >= 0.50
        valid_thresholds  = thresholds_arr[valid_mask]
        optimal_threshold = float(valid_thresholds.max()) if len(valid_thresholds) > 0 else 0.5
        print(f"\nThreshold optimised for precision (recall floor=0.50)")

    else:  # roc_auc — default 0.5 is fine, ROC doesn't depend on threshold
        optimal_threshold = 0.5
        print(f"\nNo threshold tuning for optimize_for=roc_auc (using 0.5)")
else:
    # Vanilla / Initial Run Fix: Maximize F2 score on the imbalanced X_test split 
    # to automatically shift the decision line past the synthetic SMOTE cloud.
    f2_scores = (5 * precisions_arr * recalls_arr) / (4 * precisions_arr + recalls_arr + 1e-9)
    if len(thresholds_arr) > 0:
        optimal_threshold = float(thresholds_arr[np.argmax(f2_scores[:-1])])
    else:
        optimal_threshold = 0.95  # Strict safety fallback if test set is tiny
    print(f"\nInitial model threshold tuned via F2-Score to counteract SMOTE probability inflation.")

print(f"  optimal_threshold : {optimal_threshold:.4f}")

# ── evaluation at optimal threshold ──────────────────────────────────────────
y_pred = (y_proba >= optimal_threshold).astype(int)

# ── real latency via timing ───────────────────────────────────────────────────
_ = model.predict_proba(X_test[:10])   # warm up
start      = time.perf_counter()
_          = model.predict_proba(X_test)
latency_ms = round((time.perf_counter() - start) * 1000 / len(X_test), 4)
print(f"Latency (per sample): {latency_ms:.4f}ms")

metrics = {
    "accuracy":      round(accuracy_score(y_test, y_pred), 4),
    "precision":     round(precision_score(y_test, y_pred, zero_division=0), 4),
    "recall":        round(recall_score(y_test, y_pred, zero_division=0), 4),
    "f1":            round(f1_score(y_test, y_pred, zero_division=0), 4),
    "roc_auc":       round(roc_auc_score(y_test, y_proba), 4),
    "avg_precision": round(average_precision_score(y_test, y_proba), 4),
    "latency_ms":    latency_ms,
}

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*60}")
print(f"  {model.__class__.__name__}  (threshold={optimal_threshold:.4f})")
print(f"{'='*60}")
for k, v in metrics.items():
    print(f"  {k:<20} {v}")
print(f"\n  Confusion matrix:")
print(f"    TP={tp}  FP={fp}")
print(f"    FN={fn}  TN={tn}")
print(f"{'='*60}\n")

# ── save artefacts ────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)

if REFIT_PREPROCESSORS or scaler_fitted:
    joblib.dump(amount_scaler, AMOUNT_SCALER_PATH)
    joblib.dump(time_scaler, TIME_SCALER_PATH)
    print(f"Saved scalers   → {AMOUNT_SCALER_PATH}, {TIME_SCALER_PATH}")

metadata = {
    "model_id":           MODEL_ID,
    "environment":        ENVIRONMENT,
    "model_type":         model.__class__.__name__,
    "train_frac":         TRAIN_FRAC,
    "full_train":         FULL_TRAIN,
    "train_samples":      int(len(X_resampled)),
    "test_samples":       int(len(X_test)),
    "feature_cols":       feature_cols,
    "reference_stats":    reference_stats,
    "metrics":            metrics,
    "optimal_threshold":  optimal_threshold,
    "trained_at":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "triggered_by":       TRIGGERED_BY,
    # prescription snapshot — for full auditability
    "prescription": {
        "data_strategy":       DATA_STRATEGY,
        "window_days":         WINDOW_DAYS,
        "drift_period_weight": DRIFT_PERIOD_WEIGHT,
        "exclude_before":      EXCLUDE_BEFORE,
        "refit_preprocessors": REFIT_PREPROCESSORS,
        "drifted_features":    DRIFTED_FEATURES,
        "optimize_for":        OPTIMIZE_FOR,
        "target_recall":       TARGET_RECALL,
        "target_roc_auc":      TARGET_ROC_AUC,
        "deployment_strategy": DEPLOYMENT_STRATEGY,
    } if FULL_TRAIN else None,
}
META_PATH.write_text(json.dumps(metadata, indent=2))

print(f"Saved model    → {MODEL_PATH}")
print(f"Saved metadata → {META_PATH}")


start = time.time()

# ── log to MLflow ─────────────────────────────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
token        = os.getenv("MLFLOW_TRACKING_TOKEN", "")
os.environ["MLFLOW_TRACKING_TOKEN"] = token

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT)

mlflow.set_registry_uri("databricks-uc")


run_name = f"{MODEL_ID}-{'retrain' if FULL_TRAIN else 'initial'}"

with mlflow.start_run(run_name=run_name):

    # ── performance metrics ──
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # ── training params ──
    mlflow.log_param("model_type",         model.__class__.__name__)
    mlflow.log_param("train_frac",         TRAIN_FRAC)
    mlflow.log_param("train_samples",      len(X_resampled))
    mlflow.log_param("smote",             True)
    mlflow.log_param("optimal_threshold",  optimal_threshold)

    # ── prescription params (only meaningful for full retrains) ──
    if FULL_TRAIN:
        mlflow.log_param("data_strategy",       DATA_STRATEGY)
        mlflow.log_param("window_days",         WINDOW_DAYS)
        mlflow.log_param("drift_period_weight", DRIFT_PERIOD_WEIGHT)
        mlflow.log_param("exclude_before",      EXCLUDE_BEFORE or "none")
        mlflow.log_param("refit_preprocessors", REFIT_PREPROCESSORS)
        mlflow.log_param("optimize_for",        OPTIMIZE_FOR)
        mlflow.log_param("target_recall",       TARGET_RECALL)
        mlflow.log_param("target_roc_auc",      TARGET_ROC_AUC)
        mlflow.log_param("deployment_strategy", DEPLOYMENT_STRATEGY)
        mlflow.log_param("drifted_features",    json.dumps(DRIFTED_FEATURES))

    # ── tags — used by the watcher thread to detect a retrained model ──
    mlflow.set_tag("model_id",     MODEL_ID)
    mlflow.set_tag("environment",  ENVIRONMENT)
    mlflow.set_tag("chaos",        "false")
    mlflow.set_tag("triggered_by", TRIGGERED_BY)
    mlflow.set_tag("full_train",   str(FULL_TRAIN))

    input_sample = X_test[:5]
    
    # 1. ENHANCED: Pass registered_model_name to trigger MLflow auto-version numbering
    model_info = mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model",
        registered_model_name=MODEL_ID,  # e.g., "Fraud_Detection_Model"
        input_example=input_sample,
    )
    # Capture the auto-generated version number string (e.g., "3")
    new_version_num = model_info.registered_model_version
    print(f"MLflow auto-assigned version ID: {new_version_num}")
    
    mlflow.log_artifact(str(META_PATH))
    # Scalers must travel with the model — watch_for_retrain reloads them
    # alongside model + metadata so the new LR coefficients see correctly
    # scaled features at inference time.
    mlflow.log_artifact(str(AMOUNT_SCALER_PATH))
    mlflow.log_artifact(str(TIME_SCALER_PATH))

    # Save per-feature reference histograms for drift computation
    # 10-15 bins is the sweet spot for a 1b LLM's context window
    reference_histograms = {}
    for i, col in enumerate(feature_cols):
        counts, bin_edges = np.histogram(X_train_full[:, i], bins=12) # Reduced for LLM
        reference_histograms[col] = {
            "counts":     counts.tolist(),
            "bin_edges":  bin_edges.tolist(),
            "mean":       float(X_train_full[:, i].mean()),
            "std":        float(X_train_full[:, i].std()),
        }

    ref_hist_path = MODEL_DIR / "reference_histograms.json"
    ref_hist_path.write_text(json.dumps(reference_histograms))
    mlflow.log_artifact(str(ref_hist_path))

    run_id = mlflow.active_run().info.run_id
    print(f"MLflow run ID : {run_id}")

    # 3. AUTOMATED PROMOTION: Assign 'champion' alias directly from the runner context
    # This completely eliminates the need for your LangGraph server to run polling code
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Only auto-promote if the run wasn't flagged for an isolated shadow test
        if os.getenv("DEPLOYMENT_STRATEGY", "immediate") != "shadow":
            print(f"Promoting Model version {new_version_num} to 'champion' alias dynamically...")
            
            client.set_registered_model_alias(
                name=MODEL_ID,
                alias="champion",
                version=str(new_version_num)
            )
            print(f"SUCCESS: 'champion' alias updated to Version {new_version_num} successfully.")
            alias_metadata = client.get_model_version_by_alias(
                name=MODEL_ID,
                alias="champion"
            )
            model_version = alias_metadata.version
            print(f"Testing method for retrieval of model version -- expected:{new_version_num}, actual:{model_version}")
        else:
            print(f"Shadow deployment strategy requested. Version {new_version_num} registered but not promoted.")     
    except Exception as promotion_error:
        print(f"ERROR: Model registration succeeded, but alias promotion failed: {promotion_error}")
        # We don't want to crash the whole runner if just the alias API fails

print(f"Time taken to log stuff to mlflow: {time.time() - start:.2f} seconds")

start = time.time()

# ── post-training validation gate ────────────────────────────────────────────
# Mirror of the validation step in retrain.yml — catches failures locally too.
if FULL_TRAIN:
    print(f"\n{'='*60}")
    print(f"  Post-training validation")
    print(f"{'='*60}")

    passed = True

    roc_auc_val = metrics["roc_auc"]
    recall_val  = metrics["recall"]

    if roc_auc_val < TARGET_ROC_AUC:
        print(f"  FAIL  roc_auc {roc_auc_val:.4f} < target {TARGET_ROC_AUC}")
        passed = False
    else:
        print(f"  PASS  roc_auc {roc_auc_val:.4f} >= {TARGET_ROC_AUC}")

    if recall_val < TARGET_RECALL:
        print(f"  FAIL  recall  {recall_val:.4f} < target {TARGET_RECALL}")
        passed = False
    else:
        print(f"  PASS  recall  {recall_val:.4f} >= {TARGET_RECALL}")

    print(f"{'='*60}")

    if not passed:
        raise SystemExit(
            "\nValidation failed — model does not meet prescription targets. "
            "The MLflow run has been logged for diagnostics. "
            "Remediation agent will record this outcome."
        )

    print(f"\n  Deployment strategy : {DEPLOYMENT_STRATEGY}")
    if DEPLOYMENT_STRATEGY == "canary":
        print(f"  Ready for canary rollout.")
    elif DEPLOYMENT_STRATEGY == "blue_green":
        print(f"  Ready for blue/green swap.")
    else:
        print(f"  Ready for immediate promotion.")

print(f"Post-training validation time: {time.time() - start:.2f} seconds")

print("\nTraining complete.")