"""
server.py — FastAPI inference server + MCP server for the fraud classifier.

Option B: all inference-time drift injection removed. The server is now a
pure prediction + monitoring service. Drift is introduced at the data layer
(dataset_generator.py creates drifted CSVs; transaction_generator.py replays
them). The model predicts on genuinely shifted features — retrain has a real
effect.

Retained:
  - /predict, /predict/batch
  - hot-reload watcher (MLflow retrained model detection)
  - get_current_metrics, get_prediction_history, predict_fraud, get_model_info
  - reset_reference MCP tool

Removed:
  - state["drift_config"] and state["drift_factor"]
  - apply_data_drift()
  - apply_concept_drift()
  - _feature_index() helper
  - inject_drift MCP tool
  - get_drift_status MCP tool
  - simulate_drift MCP tool
  - all drift application logic in transaction_to_features()
"""

import os
import json
import time
import uuid
import threading
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mlflow.tracking import MlflowClient

import tempfile

from dotenv import load_dotenv
load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
MODEL_DIR    = Path(__file__).parent / "model"
MODEL_PATH   = MODEL_DIR / "fraud_classifier.joblib"
AMOUNT_SCALER_PATH = MODEL_DIR / "amount_scaler.joblib"
TIME_SCALER_PATH   = MODEL_DIR / "time_scaler.joblib"
META_PATH    = MODEL_DIR / "metadata.json"

MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_ID     = os.getenv("MODEL_ID", "fraud-classifier-v1")
ENVIRONMENT  = os.getenv("ENVIRONMENT", "production")
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/production-model-evals")
RELOAD_CHECK = int(os.getenv("RELOAD_CHECK_INTERVAL", "60"))
HISTORY_SIZE = int(os.getenv("PREDICTION_HISTORY_SIZE", "1000"))

# ── shared state ──────────────────────────────────────────────────────────────
state: dict[str, Any] = {
    "model":         None,
    "amount_scaler": None,
    "time_scaler":   None,
    "loaded_run_id": None,
    "last_mlflow_metric_log": datetime.min.replace(tzinfo=timezone.utc),
    "model_name": MODEL_ID,
    "model_version": os.getenv("MLFLOW_MODEL_VERSION", "1"),


    # ── NEW: Required by the telemetry worker to resolve active version tags ──
    "metadata": {
        "feature_cols":  [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    },

    "prediction_history": deque(maxlen=5000),
    "feature_history":    deque(maxlen=5000),

    "stats": {
        "total_predictions": 0,
        "fraud_detected":    0,
        "errors":            0,
        "latencies_ms":      deque(maxlen=200),
    },

}

state_lock = threading.Lock()

def mlflow_telemetry_worker():
    """
    Asynchronous daemon loop. Periodically checks MLflow for the active champion model version,
    transforms feature history into 12-bin structures, and writes to that version's tags.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    
    while True:

        print(f"[Telemetry Worker] Updating production feature distributions in MLflow...")
        
        with state_lock:
            feature_history = list(state["feature_history"])
            metadata        = state["metadata"]
            
        feature_cols = metadata.get("feature_cols", [])
        model_name   = state.get("model_name")
        print(f"[Telemetry Worker] Model name from state: {model_name}")

        active_run_id = None
        
        # 1. NEW: Poll MLflow to find out what the active version integer is right now
        try:
            alias_metadata = client.get_model_version_by_alias(name=model_name, alias="champion")
            active_version = alias_metadata.version  # Captures new versions automatically
            active_run_id = alias_metadata.run_id
            # Sync back to shared server state metadata block
            with state_lock:
                state["model_version"] = active_version
            print(f"[Telemetry Worker] updated state to active version: {state["model_version"]}")
        except Exception as exc:
            print(f"[Telemetry Worker Warning] Could not resolve 'champion' alias, using last known: {exc}")
            active_version = state.get("model_version", "1")
    
        if len(feature_history) < 50:
            # Check and update telemetry distributions every 60 seconds
            time.sleep(15)
            continue 
        
        # 2. Calculate distributions using optimized bin sizes
        arr = np.array(feature_history)
        production_histograms = {}

        for i, col in enumerate(feature_cols):
            counts, bin_edges = np.histogram(arr[:, i], bins=12)
            production_histograms[col] = {
                "counts":     counts.tolist(),
                "bin_edges":  bin_edges.tolist(),
                "mean":       float(arr[:, i].mean()),
                "std":        float(arr[:, i].std()),
            }

        # 3. Asynchronously push to the dynamically resolved model registry version tags.
        # NOTE: MLflow's `active_run()` is process-global, not thread-local. Using
        # `with mlflow.start_run(...)` here races against the metrics-snapshot run
        # created inside compute_current_metrics() when both fire near-simultaneously.
        # We attach artifacts to the existing training run via the client API
        # directly, which never touches the global active-run pointer.
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = Path(tmpdir) / "latest_production_histogram.json"
                artifact_path.write_text(json.dumps(production_histograms))

                client.log_artifact(run_id=active_run_id, local_path=str(artifact_path))
                client.set_model_version_tag(
                    name=model_name,
                    version=active_version,
                    key="latest_production_histogram_run_id",
                    value=active_run_id,
                )
        except Exception as exc:
            print(f"[Telemetry Worker Error] Failed to upload metrics to version {active_version}: {exc}")

        print(f"[Telemetry Worker] Updated production feature distributions for version {active_version} with {len(feature_history)} samples.")


        # Check and update telemetry distributions every 60 seconds
        time.sleep(60)

# Start the background task immediately on startup
threading.Thread(target=mlflow_telemetry_worker, daemon=True).start()

# ── model loading ─────────────────────────────────────────────────────────────

def load_model_from_disk():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}. Run scripts/train.py first."
        )
    model    = joblib.load(MODEL_PATH)
    amount_scaler = joblib.load(AMOUNT_SCALER_PATH)
    time_scaler   = joblib.load(TIME_SCALER_PATH)
    metadata = json.loads(META_PATH.read_text())
    return model, amount_scaler, time_scaler, metadata


def _retry_download(label: str, fn, attempts: int = 3, delay_seconds: int = 10):
    """
    Run a flaky MLflow artifact download with backoff. Returns the call result
    on success or re-raises the last exception after `attempts` retries.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                print(
                    f"[watcher] {label} attempt {attempt}/{attempts} failed: {exc} "
                    f"— retrying in {delay_seconds}s"
                )
                time.sleep(delay_seconds)
            else:
                print(f"[watcher] {label} failed after {attempts} attempts: {exc}")
    raise last_exc


def _check_and_swap_latest_retrain() -> bool:
    """
    Find the latest retrain run in MLflow and atomically swap model + scalers
    + metadata into the server's state.

    Returns:
        True  → a swap occurred (new run picked up)
        False → no new run (already on the latest) or no retrain runs exist

    Exceptions during artifact download/load propagate to the caller so it can
    decide whether to retry (background watcher) or proceed anyway (startup).
    """
    client     = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if not experiment:
        return False

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.triggered_by IN ('remediation_agent','manual') and tags.model_id = '{MODEL_ID}'"
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return False

    latest        = runs[0]
    latest_run_id = latest.info.run_id

    with state_lock:
        if latest_run_id == state.get("loaded_run_id"):
            return False

        print(f"\n[watcher] new retrained model — run {latest_run_id}")

        # 1. Model (hard requirement — failure raises and aborts the swap)
        new_model = _retry_download(
            f"model download (run {latest_run_id[:8]})",
            lambda: mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model"),
        )

        # 2. Scalers (HARD requirement — train.py refits these every run.
        # If we keep stale v(N-1) scalers, v(N) LR coefficients receive
        # mis-scaled features and the model silently underperforms.)
        tmpdir = tempfile.mkdtemp()
        _retry_download(
            f"amount_scaler download (run {latest_run_id[:8]})",
            lambda: client.download_artifacts(latest_run_id, "amount_scaler.joblib", tmpdir),
        )
        _retry_download(
            f"time_scaler download (run {latest_run_id[:8]})",
            lambda: client.download_artifacts(latest_run_id, "time_scaler.joblib", tmpdir),
        )
        new_amount_scaler = joblib.load(Path(tmpdir) / "amount_scaler.joblib")
        new_time_scaler   = joblib.load(Path(tmpdir) / "time_scaler.joblib")

        # 3. Metadata (soft requirement — fall back to existing if download fails)
        metadata = state["metadata"]
        try:
            _retry_download(
                f"metadata.json download (run {latest_run_id[:8]})",
                lambda: client.download_artifacts(latest_run_id, "metadata.json", tmpdir),
            )
            metadata = json.loads((Path(tmpdir) / "metadata.json").read_text())
        except Exception as e:
            print(f"[watcher] warning: failed to download metadata.json for run {latest_run_id}, using existing metadata")
            print(f"[watcher] error: {e}")

        # 4. Atomic swap — model, scalers, and metadata all move together.
        state["model"]          = new_model
        state["amount_scaler"]  = new_amount_scaler
        state["time_scaler"]    = new_time_scaler
        state["loaded_run_id"]  = latest_run_id
        state["metadata"]       = metadata

        # 5. Drain the prediction window. Records left from the previous model
        # would otherwise dominate compute_current_metrics for up to
        # `maxlen` predictions (≈8–80 min depending on replay rate), causing
        # the monitor to attribute prior-model behaviour to the new model and
        # potentially trigger a spurious retrain. A clean window means the
        # next snapshot returns the training-time fallback from metadata until
        # enough fresh labelled records accumulate.
        state["prediction_history"].clear()
        state["feature_history"].clear()
        state["stats"]["total_predictions"] = 0
        state["stats"]["fraud_detected"]    = 0
        state["stats"]["errors"]            = 0
        state["stats"]["latencies_ms"].clear()

        print(f"[watcher] model + scalers hot-reloaded; prediction window cleared.\n")
        return True


def watch_for_retrain():
    """Background thread — polls MLflow on RELOAD_CHECK intervals."""
    token = os.getenv("MLFLOW_TRACKING_TOKEN", "")
    if token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = token
    mlflow.set_tracking_uri(MLFLOW_URI)

    while True:
        time.sleep(RELOAD_CHECK)
        try:
            _check_and_swap_latest_retrain()
        except Exception as e:
            print(f"[watcher] error: {e}")

# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    token = os.getenv("MLFLOW_TRACKING_TOKEN", "")
    if token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = token

    model, amount_scaler, time_scaler, metadata = load_model_from_disk()
    with state_lock:
        state["model"] = model
        state["amount_scaler"] = amount_scaler
        state["time_scaler"] = time_scaler
        state["metadata"] = metadata
        state["loaded_run_id"] = state.get("loaded_run_id")
        state["last_mlflow_metric_log"] = state.get(
            "last_mlflow_metric_log",
            datetime.min.replace(tzinfo=timezone.utc),
        )

    print(f"Loaded model  : {metadata['model_type']}")
    print(f"Train samples : {metadata['train_samples']}")
    print(f"Trained at    : {metadata['trained_at']}")

    # Immediate MLflow sync — the on-disk model may be stale relative to what
    # an external trainer (Databricks, another worker) has registered since the
    # last restart. If MLflow has a newer retrain, swap to it before serving
    # any traffic. If MLflow is unreachable or empty, we keep the on-disk model
    # and the background watcher will retry on its own cadence.
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        if _check_and_swap_latest_retrain():
            print("[watcher] startup: synced to latest MLflow retrain")
        else:
            print("[watcher] startup: on-disk model is current (no newer MLflow run)")
    except Exception as e:
        print(f"[watcher] startup: MLflow sync failed, continuing with on-disk model — {e}")

    t = threading.Thread(target=watch_for_retrain, daemon=True)
    t.start()
    print(f"[watcher] polling MLflow every {RELOAD_CHECK}s for retrained model\n")

    yield
    print("Server shutting down.")


app = FastAPI(
    title="Fraud Classifier MCP Server",
    description="Credit card fraud inference + MCP tool interface for mlops_agents",
    version="2.0.0",
    lifespan=lifespan,
)

# ── pydantic models ───────────────────────────────────────────────────────────

class Transaction(BaseModel):
    # Allow the generator to send `_true_label` alongside the features — the
    # field name can't start with underscore in Python, hence the alias.
    model_config = ConfigDict(populate_by_name=True)

    v1: float;  v2: float;  v3: float;  v4: float;  v5: float
    v6: float;  v7: float;  v8: float;  v9: float;  v10: float
    v11: float; v12: float; v13: float; v14: float; v15: float
    v16: float; v17: float; v18: float; v19: float; v20: float
    v21: float; v22: float; v23: float; v24: float; v25: float
    v26: float; v27: float; v28: float
    amount: float = Field(..., ge=0)
    time:   float = Field(..., ge=0)
    # Ground-truth label, supplied by the dataset replayer for evaluation.
    # None in real production traffic — handled gracefully by metric code.
    true_label: Optional[int] = Field(default=None, alias="_true_label")

class MCPCallRequest(BaseModel):
    tool:   str
    params: dict = {}

# ── feature extraction ────────────────────────────────────────────────────────

def transaction_to_features(tx: Transaction) -> np.ndarray:
    """
    Pure feature extraction and scaling. No drift application — drift is
    introduced at the data layer via drifted dataset CSVs, not at inference time.
    """
    with state_lock:
        amount_scaler = state["amount_scaler"]
        time_scaler   = state["time_scaler"]

    raw = np.array([[
        tx.v1,  tx.v2,  tx.v3,  tx.v4,  tx.v5,
        tx.v6,  tx.v7,  tx.v8,  tx.v9,  tx.v10,
        tx.v11, tx.v12, tx.v13, tx.v14, tx.v15,
        tx.v16, tx.v17, tx.v18, tx.v19, tx.v20,
        tx.v21, tx.v22, tx.v23, tx.v24, tx.v25,
        tx.v26, tx.v27, tx.v28,
        tx.amount, tx.time,
    ]])

    raw[0, 28] = amount_scaler.transform([[tx.amount]])[0][0]
    raw[0, 29] = time_scaler.transform([[tx.time]])[0][0]

    return raw


def run_inference(features: np.ndarray) -> dict:
    with state_lock:
        model = state["model"]
        threshold = state.get("metadata", {}).get("optimal_threshold", 0.5)

    start   = time.perf_counter()
    proba   = model.predict_proba(features)[0]
    pred    = 1 if proba[1] >= threshold else 0
    latency = (time.perf_counter() - start) * 1000

    return {
        "prediction": pred,
        "fraud_prob": round(float(proba[1]), 4),
        "legit_prob": round(float(proba[0]), 4),
        "latency_ms": round(latency, 3),
    }


def compute_current_metrics() -> dict:
    with state_lock:
        history  = list(state["prediction_history"])
        stats    = dict(state["stats"])
        metadata = state["metadata"]

    base = metadata["metrics"]
    total_predictions = stats["total_predictions"]

    error_rate = round(
        stats["errors"] / max(total_predictions, 1), 4
    )

    # ── fallback when no history yet ─────────────────────────────────────────
    if not history:
        return {
            "fraud_rate":  0.0,
            "latency_ms":  0.0,
            "error_rate":  error_rate,
            "sample_size": 0,
            "precision":   base.get("precision", 0.0),
            "recall":      base.get("recall", 0.0),
            "f1":          base.get("f1", 0.0),
            "roc_auc":     base.get("roc_auc", 0.0),
            "accuracy":    base.get("accuracy", 0.0),
        }

    # ── window-wide stats (label-independent) ────────────────────────────────
    fraud_rate = round(
        sum(1 for h in history if h["prediction"] == 1) / len(history), 4
    )
    latencies   = list(stats["latencies_ms"])
    p95_latency = (
        round(float(np.percentile(latencies, 95)), 2)
        if latencies else 0.0
    )

    # ── classification metrics from labelled records ─────────────────────────
    # Records without a `true_label` (real production traffic) are skipped —
    # those metrics fall back to training-time values from the registry.
    labelled = [h for h in history if h.get("true_label") is not None]

    if not labelled:
        return {
            "fraud_rate":  fraud_rate,
            "latency_ms":  p95_latency,
            "error_rate":  error_rate,
            "sample_size": len(history),
            "labelled_size": 0,
            "precision":   base.get("precision", 0.0),
            "recall":      base.get("recall", 0.0),
            "f1":          base.get("f1", 0.0),
            "roc_auc":     base.get("roc_auc", 0.0),
            "accuracy":    base.get("accuracy", 0.0),
        }

    y_true = np.array([int(h["true_label"]) for h in labelled])
    y_pred = np.array([int(h["prediction"]) for h in labelled])
    y_prob = np.array([float(h["fraud_prob"]) for h in labelled])

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy  = round(float((y_pred == y_true).mean()), 4)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) > 0 else 0.0
    )

    # roc_auc requires both classes present in y_true — otherwise undefined.
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = (
            round(float(roc_auc_score(y_true, y_prob)), 4)
            if len(np.unique(y_true)) == 2 else base.get("roc_auc", 0.0)
        )
    except Exception:
        roc_auc = base.get("roc_auc", 0.0)

    return {
        "fraud_rate":  fraud_rate,
        "latency_ms":  p95_latency,
        "error_rate":  error_rate,
        "sample_size": len(history),
        "labelled_size": len(labelled),
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "roc_auc":     roc_auc,
        "accuracy":    accuracy,
    }

# ── inference endpoints ───────────────────────────────────────────────────────

@app.post("/predict")
async def predict(tx: Transaction):
    try:
        features = transaction_to_features(tx)
        result   = run_inference(features)
        tx_id    = str(uuid.uuid4())

        record = {
            "transaction_id": tx_id,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "true_label":     tx.true_label,  # None for real production traffic
            **result,
        }

        with state_lock:
            state["prediction_history"].append(record)
            state["feature_history"].append(features[0].tolist())
            state["stats"]["total_predictions"] += 1
            state["stats"]["latencies_ms"].append(result["latency_ms"])
            if result["prediction"] == 1:
                state["stats"]["fraud_detected"] += 1

        return record

    except Exception as e:
        with state_lock:
            state["stats"]["errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(transactions: list[Transaction]):
    start_time = time.perf_counter()
    
    # 1. Vectorize the incoming batch payload
    features_matrix = np.array([tx.features for tx in transactions])
    batch_size = features_matrix.shape[0]
    
    # 2. Critical State Updates & Inference (Combined under one lock pass)
    with state_lock:
        model = state["model"]
        threshold = state.get("optimal_threshold", 0.5)
        
        # FIX: Feed the batch data directly into your telemetry pipeline!
        # Convert the NumPy matrix to a list of lists so deque handles it seamlessly
        state["feature_history"].extend(features_matrix.tolist())

    # 3. Fast matrix prediction
    probas = model.predict_proba(features_matrix)
    fraud_probs = probas[:, 1]
    predictions = (fraud_probs >= threshold).astype(int)
    
    # 4. Format response payload
    latency_ms = (time.perf_counter() - start_time) * 1000
    response = []
    for i in range(batch_size):
        response.append({
            "prediction": int(predictions[i]),
            "fraud_prob": round(float(fraud_probs[i]), 4),
            "latency_batch_ms": round(latency_ms / batch_size, 3)
        })
        
    return response


@app.get("/metrics")
async def metrics():
    return compute_current_metrics()


@app.get("/model/info")
async def model_info():
    with state_lock:
        meta       = state["metadata"]
        run_id     = state["loaded_run_id"]
        stats      = dict(state["stats"])

    return {
        "model_id":      MODEL_ID,
        "environment":   ENVIRONMENT,
        "model_type":    meta["model_type"],
        "trained_at":    meta["trained_at"],
        "train_samples": meta["train_samples"],
        "full_train":    meta["full_train"],
        "triggered_by":  meta["triggered_by"],
        "loaded_run_id": run_id,
        "runtime_stats": {
            "total_predictions": stats["total_predictions"],
            "fraud_detected":    stats["fraud_detected"],
            "errors":            stats["errors"],
        },
    }


@app.get("/health")
async def health():
    with state_lock:
        model_loaded = state["model"] is not None
    return {
        "status":          "ok" if model_loaded else "degraded",
        "model_loaded":    model_loaded,
        "model_id":        MODEL_ID,
        "environment":     ENVIRONMENT,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }

# ── MCP manifest ──────────────────────────────────────────────────────────────

MCP_MANIFEST = {
    "schema_version": "v1",
    "name":           "fraud-classifier",
    "description":    "Credit card fraud classifier with monitoring tools",
    "tools": [
        {
            "name":        "predict_fraud",
            "description": "Run fraud inference on a single credit card transaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction": {
                        "type":        "object",
                        "description": "Transaction features (V1-V28, amount, time)",
                    }
                },
                "required": ["transaction"],
            },
        },
        {
            "name":        "get_current_metrics",
            "description": (
                "Get live runtime and operational metrics including latency, fraud rate, error rate, and evaluation metrics."
                "Also logs a snapshot to MLflow tagged run_type=metrics_snapshot."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name":        "get_prediction_history",
            "description": "Retrieve the last N predictions with fraud probabilities and latencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "default": 50}
                },
            },
        },
        {
            "name":        "get_model_info",
            "description": "Get model metadata, reference window status, and runtime stats.",
            "parameters":  {"type": "object", "properties": {}},
        },
    ],
}


@app.get("/mcp")
async def mcp_manifest():
    return MCP_MANIFEST

# ── MCP tool execution ────────────────────────────────────────────────────────

@app.post("/mcp/call")
async def mcp_call(req: MCPCallRequest):
    tool   = req.tool
    params = req.params

    if tool == "predict_fraud":
        tx_data = params.get("transaction", {})
        try:
            tx = Transaction(**{k.lower(): v for k, v in tx_data.items()})
        except Exception as e:
            raise HTTPException(400, f"Invalid transaction: {e}")
        return await predict(tx)

    elif tool == "get_current_metrics":
        token = os.getenv("MLFLOW_TRACKING_TOKEN", "")
        if token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT)

        metrics_data = compute_current_metrics()

        now = datetime.now(timezone.utc)
        time_elapsed = None

        with state_lock:
            last_log_time = state["last_mlflow_metric_log"]
            active_version = state.get("model_version", "1")
            time_elapsed = now - last_log_time

        if time_elapsed < timedelta(minutes=1):
            print(f"[Mlflow Snapshot] Skipping log to MLflow (last log was {time_elapsed.seconds}s ago)")
            with state_lock:
                state["last_mlflow_metric_log"] = now
            return metrics_data

        try:
            # 1. Resolve what version this server is currently serving out of state
            with state_lock:
                active_version = state.get("model_version", "1")

            with mlflow.start_run(run_name=f"{MODEL_ID}-metrics-snapshot"):
                for k, v in metrics_data.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.set_tag("model_id",    MODEL_ID)
                mlflow.set_tag("environment", ENVIRONMENT)
                mlflow.set_tag("run_type",    "metrics_snapshot")
                mlflow.set_tag("triggered_by", "mcp_tool")
                # NEW ALIGNMENT TAG: Bind this tracking run to the registry version scale
                mlflow.set_tag("model_version", str(active_version))
        except Exception as e:
            print(f"[mlflow] snapshot log failed (non-fatal): {e}")

        print(f"[Mlflow Snapshot] Logged current metrics snapshot for version {active_version}: {metrics_data}")

        return metrics_data

    elif tool == "get_prediction_history":
        n = int(params.get("n", 50))
        with state_lock:
            history = list(state["prediction_history"])[-n:]
        return {"count": len(history), "predictions": history}

    elif tool == "get_model_info":
        return await model_info()

    else:
        raise HTTPException(
            400,
            f"Unknown tool: '{tool}'. Available: {[t['name'] for t in MCP_MANIFEST['tools']]}",
        )

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )