# agents/monitor_agent.py

from __future__ import annotations
import json
import logging
import os
from typing import Any, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, ValidationError

from state import AgentState
from mlops_agents.rag.store import RAGStore
from mlops_agents.tools.metrics_source import fetch_model_metrics, MetricsSourceError

# NEW IMPORT: Add MLflow tracking library access
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

SeverityLevel = Literal["none", "minor", "major", "critical"]

class SeverityClassification(BaseModel):
    severity: SeverityLevel = Field(
        description="Dynamic determination based on comparison against dynamic thresholds."
    )
    confidence: float = Field(description="Confidence rating between 0.0 and 1.0.", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanatory bridge highlighting anomalies or threshold breaches.")

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v: Any) -> str:
        if isinstance(v, str):
            normalised = v.strip().lower()
            if normalised in ("none", "minor", "major", "critical"):
                return normalised
        raise ValueError(f"Invalid severity value '{v}'.")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try: return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError): return 0.5

# ---------------------------------------------------------------------------
# Severity thresholds (overridable via env)
# ---------------------------------------------------------------------------

def _threshold(env_key: str, default: float) -> float:
    return float(os.getenv(env_key, str(default)))


THRESHOLDS = {
    "accuracy_critical":   lambda: _threshold("THRESHOLD_ACCURACY_CRITICAL",  0.65),
    "accuracy_major":      lambda: _threshold("THRESHOLD_ACCURACY_MAJOR",     0.72),
    "accuracy_minor":      lambda: _threshold("THRESHOLD_ACCURACY_MINOR",     0.80),
    "drift_critical":      lambda: _threshold("THRESHOLD_DRIFT_CRITICAL",     0.60),
    "drift_major":         lambda: _threshold("THRESHOLD_DRIFT_MAJOR",        0.35),
    "drift_minor":         lambda: _threshold("THRESHOLD_DRIFT_MINOR",        0.20),
    "latency_critical_ms": lambda: _threshold("THRESHOLD_LATENCY_CRITICAL_MS", 2000),
    "latency_major_ms":    lambda: _threshold("THRESHOLD_LATENCY_MAJOR_MS",   1000),
    "error_rate_critical": lambda: _threshold("THRESHOLD_ERROR_RATE_CRITICAL", 0.10),
    "error_rate_major":    lambda: _threshold("THRESHOLD_ERROR_RATE_MAJOR",    0.05),
}


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def _default_thresholds() -> dict:
    return {k: fn() for k, fn in THRESHOLDS.items()}


def _get_thresholds(model_id: str, rag: RAGStore) -> dict:
    try:
        dynamic = rag.get_dynamic_thresholds(model_id=model_id)
        if dynamic:
            logger.info("Using dynamic thresholds from RAG (model=%s)", model_id)
            return dynamic
    except Exception as e:
        logger.warning("Failed loading dynamic thresholds: %s", e)
    
    # Fallback default dict mapping structures dynamically
    return _default_thresholds()

def _build_severity_llm() -> Any:
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model_name, base_url=ollama_url, temperature=0
    ).with_structured_output(SeverityClassification).with_retry(
        retry_exception_types=(ValidationError, Exception),
        max_attempt_number=3,
        wait_exponential_jitter=True
    )

def monitor_agent(state: AgentState, rag: RAGStore) -> AgentState:
    model_id = state.get("model_id", os.getenv("DEFAULT_MODEL_ID", "main.default.fraud_classifier_v1"))
    environment = state.get("environment", os.getenv("DEFAULT_ENVIRONMENT", "production"))
    model_version = state.get("model_version", os.getenv("DEFAULT_MODEL_VERSION", "1"))
    
    if not model_id:
        raise ValueError("state['model_id'] is a missing dependency context.")

    # Explicitly configure backend connection URI before client initialization
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    # 1. NEW: Resolve the active version integer dynamically via the "champion" alias
    try:
        alias_metadata = client.get_model_version_by_alias(name=model_id, alias="champion")
        model_version = alias_metadata.version  # Dynamically yields "1", "2", "3", etc.
        logger.info("Resolved active champion version for monitoring: %s", model_version)
    except Exception as exc:
        logger.error("Failed to resolve 'champion' alias in MLflow. Falling back to version '1': %s", exc)
        model_version = "1"

    try:
        metrics = fetch_model_metrics(model_id=model_id, environment=environment, window_minutes=15)
    except MetricsSourceError as exc:
        metrics = {"model_id": model_id, "environment": environment, "fetch_error": str(exc)}

    # 2. Fetch reference baseline distribution from model run artifact (Uses the dynamic model_version)
    ref_histograms = None
    try:
        model_details = client.get_model_version(name=model_id, version=model_version)
        local_path = client.download_artifacts(model_details.run_id, "reference_histograms.json")
        ref_histograms = json.loads(Path(local_path).read_text())
    except Exception as exc:
        logger.warning("Could not load reference histograms from MLflow: %s", exc)

    # 3. Fetch production distribution from MLflow registry tags (Uses the dynamic model_version)
    prod_histograms = None
    try:
        model_details = client.get_model_version(name=model_id, version=model_version)
        prod_hist_str = model_details.tags.get("latest_production_histogram", "{}")
        prod_histograms = json.loads(prod_hist_str)
    except Exception as exc:
        logger.warning("Could not fetch live production histograms from MLflow tags: %s", exc)

    thresholds = _get_thresholds(model_id, rag)
    trend = rag.query_recent_metrics(model_id=model_id, n_results=10, environment=environment)

    # Agent reasoning prompt replacing structural hardcoded if-else statements
    prompt = f"""You are an autonomous MLOps evaluator. Analyze the operational metrics against the target context bounds.
    
    Current telemetry signals: {json.dumps(metrics)}
    Dynamic reference constraints: {json.dumps(thresholds)}
    Historical runs: {json.dumps(trend[:3])}

    Determine if metrics fall out-of-bounds and categorize severity Level. Ensure critical drops yield immediate 'critical' classifications."""

    llm = _build_severity_llm()
    try:
        result = llm.invoke([
            SystemMessage(content="You parse runtime anomalies accurately into valid JSON schemas."),
            HumanMessage(content=prompt)
        ])
    except Exception as exc:
        logger.error("LLM evaluation failure, entering autonomous safe fallback: %s", exc)
        result = SeverityClassification(severity="minor", confidence=0.1, reasoning="Fallback triggered due to runtime LLM timeout.")

    rag.save_metrics_snapshot(metrics=metrics, severity=result.severity)

    # 3. CHANGED: Inject statistical summaries directly into state. 
    # This replaces raw CSV path strings, enabling your simplified reasoning diagnosis agent.
    metrics["reference_histograms"]  = ref_histograms
    metrics["production_histograms"] = prod_histograms
    metrics["monitored_features_list"]  = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    active_dataset_path = Path(__file__).parent.parent / "data" / "active_dataset.json"
    active_dataset_name = "baseline" 
    with open(active_dataset_path, "r") as f:
        active_dataset_name = json.load(f).get("dataset", "baseline")
    metrics["active_dataset"] = active_dataset_name

    # Add a unified metadata footprint that matches what the model server uses
    metrics["metadata"] = {
        "model_name": model_id,
        "model_version": model_version,
    }

    return {
        **state,
        "metrics": metrics,
        "model_id": model_id,          # Explicitly saved to top-level state
        "environment": environment,    # Explicitly saved to top-level state
        "model_version": model_version,
        "severity": result.severity,
        "thresholds": thresholds,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Monitor Agent] Classified status as {result.severity}. Reason: {result.reasoning}")
        ]
    }
