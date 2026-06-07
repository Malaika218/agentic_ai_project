# agents/monitor_agent.py

from __future__ import annotations
import json
import logging
import os
from typing import Any, Literal
from mlops_agents.llm_manager import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, ValidationError

from mlops_agents.state import AgentState
from mlops_agents.rag.store import RAGStore
from mlops_agents.tools.metrics_source import fetch_model_metrics, MetricsSourceError
from mlops_agents.tools.severity_classifier import classify_severity
from mlops_agents.tools.token_tracker import TokenUsageHandler

# NEW IMPORT: Add MLflow tracking library access
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

SeverityLevel = Literal["none", "minor", "major", "critical"]


class SeverityNarrative(BaseModel):
    """
    LLM contract — narrative only. The `severity` value is computed
    deterministically by the `classify_severity` tool; the LLM is given
    the answer and only writes the human-readable reasoning.
    """
    reasoning: str = Field(description="Short human-readable explanation of the severity classification.")
    confidence: float = Field(default=1.0, description="Confidence rating between 0.0 and 1.0.", ge=0.0, le=1.0)

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
    "latency_critical_ms": lambda: _threshold("THRESHOLD_LATENCY_CRITICAL_MS", 2000),
    "latency_major_ms":    lambda: _threshold("THRESHOLD_LATENCY_MAJOR_MS",   1000),
    "error_rate_critical": lambda: _threshold("THRESHOLD_ERROR_RATE_CRITICAL", 0.10),
    "error_rate_major":    lambda: _threshold("THRESHOLD_ERROR_RATE_MAJOR",    0.05),
    # Recall — fraction of fraud caught. Critical when missing ≥ 40 % of fraud.
    "recall_critical":     lambda: _threshold("THRESHOLD_RECALL_CRITICAL",    0.60),
    "recall_major":        lambda: _threshold("THRESHOLD_RECALL_MAJOR",       0.75),
    # ROC-AUC — discrimination ability. 0.5 = random, 1.0 = perfect.
    "roc_auc_critical":    lambda: _threshold("THRESHOLD_ROC_AUC_CRITICAL",   0.75),
    "roc_auc_major":       lambda: _threshold("THRESHOLD_ROC_AUC_MAJOR",      0.85),
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
        logger.info("Failed loading dynamic thresholds: %s", e)
    
    # Fallback default dict mapping structures dynamically
    return _default_thresholds()

def _build_narrative_llm() -> Any:
    from tenacity import retry_if_exception_type
    return (
        get_llm(temperature=0)
        .with_structured_output(SeverityNarrative)
        .with_retry(
            retry_if_exception_type=
                (ValidationError, Exception),
            stop_after_attempt=3,
            wait_exponential_jitter=True,
        )
    )

def monitor_agent(state: AgentState, rag: RAGStore) -> AgentState:
    model_id = state.get("model_id") or os.getenv(
        "DEFAULT_MODEL_ID", "main.default.fraud_classifier_v1"
    )
    environment = state.get("environment", os.getenv("DEFAULT_ENVIRONMENT", "production"))
    model_version = state.get("model_version", os.getenv("DEFAULT_MODEL_VERSION", "1"))

    if not model_id:
        raise ValueError("state['model_id'] is a missing dependency context.")

    # Explicitly configure backend connection URI before client initialization.
    # Only propagate the token if it's actually set — writing an empty string
    # into the env makes MLflow auth treat it as a present-but-empty token,
    # which fails differently from "unset" and breaks token-less auth modes.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    token = os.getenv("MLFLOW_TRACKING_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = token
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    # 1. Resolve the active champion version identifier first (The Source of Truth)
    try:
        alias_metadata = client.get_model_version_by_alias(name=model_id, alias="champion")
        model_version = alias_metadata.version       # e.g., "3"
        training_run_id = alias_metadata.run_id      # The actual run ID that trained the model!
    except Exception as exc:
        logger.info("Failed resolving 'champion' alias in MLflow: %s", exc)
        return {**state, "severity": "none"}

    logger.info(
        "[Monitor] starting — model_id=%s environment=%s version=%s",
        model_id, environment, model_version,
    )

    # 2. Fetch performance numbers from the latest metrics snapshot matching this model ID
    try:
        # Instead of generic search, explicitly fetch metrics from the tracking experiment
        # or rely on the metrics payload returned natively by fetch_model_metrics()
        metrics = fetch_model_metrics(model_id=model_id, environment=environment, window_minutes=15)
    except Exception as exc:
        metrics = {"fetch_error": str(exc)}

    # 3. ALIGNED: Fetch reference baseline distribution from the TRAINING run ID (Not the metric run)
    ref_histograms = None
    try:
        # Guarantees we download from the true training run containing the file asset
        local_path = client.download_artifacts(training_run_id, "reference_histograms.json")
        ref_histograms = json.loads(Path(local_path).read_text())
    except Exception as exc:
        logger.info("Could not load reference histograms from training run %s: %s", training_run_id, exc)

    # 4. ALIGNED: Fetch live production distribution from Registry Tags directly
    prod_histograms = None
    try:
        local_path = client.download_artifacts(training_run_id, "latest_production_histogram.json")
        prod_histograms = json.loads(Path(local_path).read_text())
    except Exception as exc:
        logger.info("Could not fetch live production histograms from model registry: %s", exc)

    thresholds = _get_thresholds(model_id, rag)
    trend = rag.query_recent_metrics(model_id=model_id, n_results=10, environment=environment)
    
    # ── Deterministic classification via @tool ──────────────────────────────
    # Tools created with @tool are invoked via .invoke({...}) with a kwargs dict.
    classification = classify_severity.invoke({
        "metrics": metrics,
        "thresholds": thresholds,
        "trend": trend or [],
    })
    severity_value: str = classification["severity"]
    breaches: list[dict] = classification["breaches"]
    trend_note: str = classification["trend_note"]

    logger.info(
        "[Monitor] tool classification — severity=%s breaches=%d trend_note=%s",
        severity_value, len(breaches), trend_note,
    )

    # ── LLM narrative only (the answer is already known) ────────────────────
    breach_lines = (
        "\n".join(
            f"- {b['metric']}={b['value']} breached {b['level']} threshold "
            f"{b['threshold']} (direction: {b['direction']})"
            for b in breaches
        )
        if breaches else "No threshold breaches."
    )

    narrative_prompt = (
        f"Severity has been classified as '{severity_value}'.\n\n"
        f"Breach details:\n{breach_lines}\n\n"
        f"Trend note: {trend_note}\n\n"
        "Write a 1-2 sentence human-readable reasoning for an incident report. "
        "State which metrics drove the classification and reference the values. "
        "Do NOT contradict the severity above — it is final."
    )
    logger.info("[Monitor] narrative prompt:\n%s", narrative_prompt)

    llm = _build_narrative_llm()
    tracker = TokenUsageHandler()
    try:
        narrative = llm.invoke(
            [
                SystemMessage(content="You write concise factual MLOps incident summaries."),
                HumanMessage(content=narrative_prompt),
            ],
            config={"callbacks": [tracker]},
        )
        reasoning_text = narrative.reasoning
        confidence_value = narrative.confidence
    except Exception as exc:
        logger.info("[Monitor] narrative LLM failed — using template fallback: %s", exc)
        reasoning_text = (
            f"Severity={severity_value}. " +
            (f"Breaches: {breach_lines}. " if breaches else "No threshold breaches. ") +
            trend_note
        )
        confidence_value = 1.0

    logger.info(
        "[Monitor] classification — severity=%s confidence=%.2f reasoning=%s",
        severity_value, confidence_value, reasoning_text,
    )

    rag.save_metrics_snapshot(metrics=metrics, severity=severity_value)

    # 3. CHANGED: Inject statistical summaries directly into state. 
    # This replaces raw CSV path strings, enabling your simplified reasoning diagnosis agent.
    metrics["reference_histograms"]  = ref_histograms
    metrics["production_histograms"] = prod_histograms
    metrics["monitored_features_list"]  = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    active_dataset_path = Path(__file__).parent.parent / "data" / "active_dataset.json"
    active_dataset_name = "baseline"
    try:
        with open(active_dataset_path, "r") as f:
            active_dataset_name = json.load(f).get("dataset", "baseline")
    except FileNotFoundError:
        logger.info("active_dataset.json not found at %s — defaulting to '%s'.", active_dataset_path, active_dataset_name)
    except (json.JSONDecodeError, OSError) as exc:
        logger.info("Could not read active_dataset.json (%s) — defaulting to '%s'.", exc, active_dataset_name)
    metrics["active_dataset"] = active_dataset_name
    logger.info(
        "[Monitor] context — active_dataset=%s ref_hist=%s prod_hist=%s",
        active_dataset_name,
        "present" if ref_histograms else "missing",
        "present" if prod_histograms else "missing",
    )

    # Add a unified metadata footprint that matches what the model server uses
    metrics["metadata"] = {
        "model_name": model_id,
        "model_version": model_version,
    }

    logger.info(
        "[Monitor] complete — severity=%s model_id=%s environment=%s",
        severity_value, model_id, environment,
    )

    token_summary = tracker.summary()
    logger.info(
        "[Monitor] token usage — in=%d out=%d total=%d calls=%d model=%s cost=$%.6f",
        token_summary["input_tokens"], token_summary["output_tokens"],
        token_summary["total_tokens"], token_summary["calls"],
        token_summary["model"], token_summary["cost_usd"],
    )

    return {
        **state,
        "metrics": metrics,
        "model_id": model_id,          # Explicitly saved to top-level state
        "environment": environment,    # Explicitly saved to top-level state
        "model_version": model_version,
        "severity": severity_value,
        "thresholds": thresholds,
        "token_usage": {"monitor": token_summary},
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Monitor Agent] Classified status as {severity_value}. Reason: {reasoning_text}")
        ]
    }
