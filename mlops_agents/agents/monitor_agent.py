"""
agents/monitor_agent.py

Monitor Agent — polls real-time model metrics, classifies severity,
and saves a snapshot to the RAG metrics history collection.

LLM calls use Pydantic structured outputs + `.with_structured_output()`
+ `.with_retry()` — identical pattern to the threshold advisor — so small
1B models cannot produce unparseable JSON.
"""

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

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------

SeverityLevel = Literal["none", "minor", "major", "critical"]


class SeverityClassification(BaseModel):
    """
    Output schema for the grey-zone severity classifier LLM call.

    Constrains the model to a single valid severity token plus a short
    reasoning string so we can log *why* it chose that level.
    """

    severity: SeverityLevel = Field(
        description=(
            "Severity level of the current monitoring alert. "
            "Must be exactly one of: none, minor, major, critical."
        )
    )
    confidence: float = Field(
        description="Confidence in the classification, between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description=(
            "One or two sentences explaining which metric(s) drove this "
            "classification and why it is ambiguous enough to require LLM review."
        )
    )

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v: Any) -> str:
        """Accept 'MINOR', ' Critical ', etc. and normalise to lowercase."""
        if isinstance(v, str):
            normalised = v.strip().lower()
            if normalised in ("none", "minor", "major", "critical"):
                return normalised
        raise ValueError(
            f"Invalid severity value '{v}'. Must be one of: none, minor, major, critical."
        )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        """Accept string floats from small models and clamp to [0, 1]."""
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


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
        logger.warning("Failed to load dynamic thresholds: %s", e)
    logger.info("Using default thresholds (model=%s)", model_id)
    return _default_thresholds()


# ---------------------------------------------------------------------------
# Rule-based severity classifier
# ---------------------------------------------------------------------------

def _rule_based_severity(metrics: dict, thresholds: dict) -> str | None:
    accuracy   = metrics.get("accuracy")
    drift      = metrics.get("drift_score")
    latency    = metrics.get("latency_p99_ms")
    error_rate = metrics.get("error_rate")

    if (
        (accuracy   is not None and accuracy   < thresholds["accuracy_critical"])
        or (drift   is not None and drift       > thresholds["drift_critical"])
        or (latency is not None and latency     > thresholds["latency_critical_ms"])
        or (error_rate is not None and error_rate > thresholds["error_rate_critical"])
    ):
        return "critical"

    all_ok = all([
        accuracy   is None or accuracy   >= thresholds["accuracy_minor"],
        drift      is None or drift      <= thresholds["drift_minor"],
        latency    is None or latency    <= thresholds["latency_major_ms"],
        error_rate is None or error_rate <= thresholds["error_rate_major"],
    ])
    if all_ok:
        return "none"

    if (
        (accuracy is not None and accuracy < thresholds["accuracy_major"])
        or (drift  is not None and drift    > thresholds["drift_major"])
    ):
        return "major"

    return None  # grey-zone


# ---------------------------------------------------------------------------
# LLM-based grey-zone classifier (structured output — mirrors threshold agent)
# ---------------------------------------------------------------------------

def _build_severity_llm() -> Any:
    """
    ChatOllama bound to SeverityClassification schema with exponential-backoff
    retry — identical construction pattern to _llm_threshold_advisor().
    """
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0,
    ).with_structured_output(SeverityClassification)

    llm = llm.with_retry(
        retry_exception_types=(ValidationError, Exception),
        max_attempt_number=3,
        wait_exponential_jitter=True,
    )
    return llm


def _llm_severity(
    metrics: dict,
    trend: list[dict],
    thresholds: dict,
) -> SeverityClassification:
    """
    Call the structured LLM to classify a grey-zone severity.
    Returns a SeverityClassification Pydantic object.
    Falls back to severity='minor', confidence=0.0 on total failure.
    """
    llm = _build_severity_llm()

    trend_summary = "No history available."
    if trend:
        rows = [
            f"  {m.get('sampled_at', 'N/A')[:19]}  "
            f"acc={m.get('accuracy', 'N/A')}  "
            f"drift={m.get('drift_score', 'N/A')}  "
            f"lat={m.get('latency_p99_ms', 'N/A')}ms  "
            f"err={m.get('error_rate', 'N/A')}"
            for m in trend[:5]
        ]
        trend_summary = "\n".join(rows)

    prompt = f"""You are an ML reliability expert classifying a grey-zone monitoring alert.
The rule-based classifier could not make a definitive determination — the metrics
fall between threshold boundaries. Use trend context to decide.

Current metrics:
{json.dumps(metrics, indent=2)}

Active thresholds:
{json.dumps(thresholds, indent=2)}

Recent trend (newest first):
{trend_summary}

Severity definitions:
- none    : all metrics healthy, no action needed
- minor   : slight degradation, monitor closely
- major   : significant degradation, human review required before remediation
- critical: severe degradation, immediate automated remediation required

Return a JSON object with exactly these fields:
  severity   — one of: none, minor, major, critical
  confidence — float between 0.0 and 1.0
  reasoning  — one or two sentences explaining the decision"""

    try:
        result: SeverityClassification = llm.invoke([
            SystemMessage(content="You classify ML model alert severity. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        logger.info(
            "LLM severity: %s (confidence=%.2f) — %s",
            result.severity, result.confidence, result.reasoning,
        )
        return result

    except Exception as exc:
        logger.error("All LLM severity retries exhausted: %s — defaulting to minor", exc)
        return SeverityClassification(
            severity="minor",
            confidence=0.0,
            reasoning=f"LLM failed after all retries: {exc}",
        )


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def monitor_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Monitor Agent.

    1. Fetches real-time metrics from the configured data source.
    2. Loads thresholds (dynamic from RAG or env defaults).
    3. Queries RAG for recent trend history.
    4. Classifies severity:
         - Rule-based for clear cases.
         - Structured LLM call (SeverityClassification) for grey zones.
    5. Saves a metrics snapshot to RAG.
    6. Returns the updated state.
    """
    model_id    = state.get("model_id", os.getenv("DEFAULT_MODEL_ID", ""))
    environment = state.get("environment", os.getenv("DEFAULT_ENVIRONMENT", "production"))
    window_mins = int(os.getenv("METRICS_WINDOW_MINUTES", "15"))

    if not model_id:
        raise ValueError(
            "state['model_id'] is required but not set. "
            "Pass model_id when invoking the graph."
        )

    logger.info("Monitor Agent: fetching metrics for %s (%s)", model_id, environment)

    # ── 1. Fetch real-time metrics ──────────────────────────────────────────
    try:
        metrics = fetch_model_metrics(
            model_id=model_id,
            environment=environment,
            window_minutes=window_mins,
        )
    except MetricsSourceError as exc:
        logger.error("Metrics fetch failed: %s", exc)
        metrics = {
            "model_id":      model_id,
            "model_version": "unknown",
            "environment":   environment,
            "sampled_at":    "",
            "fetch_error":   str(exc),
        }

    logger.info(
        "Metrics: accuracy=%.3f drift=%.3f latency_p99=%.1fms error_rate=%.4f",
        metrics.get("accuracy")       or 0.0,
        metrics.get("drift_score")    or 0.0,
        metrics.get("latency_p99_ms") or 0.0,
        metrics.get("error_rate")     or 0.0,
    )

    # ── 2. Load thresholds ──────────────────────────────────────────────────
    thresholds = _get_thresholds(model_id, rag)
    logger.info("Thresholds used: %s", thresholds)

    # ── 3. Query RAG trend window ───────────────────────────────────────────
    trend = rag.query_recent_metrics(
        model_id=model_id,
        n_results=int(os.getenv("TREND_WINDOW_SIZE", "20")),
        environment=environment,
    )

    # ── 4. Severity classification ──────────────────────────────────────────
    severity              = _rule_based_severity(metrics, thresholds)
    classification_method = "rule-based"
    llm_confidence: float = 1.0
    llm_reasoning:  str   = ""

    if severity is None:
        logger.info("Grey-zone — delegating to structured LLM severity classifier")
        result                = _llm_severity(metrics, trend, thresholds)
        severity              = result.severity
        classification_method = "llm"
        llm_confidence        = result.confidence
        llm_reasoning         = result.reasoning

    logger.info("Severity: %s (method=%s)", severity, classification_method)

    # ── 5. Save snapshot to RAG ─────────────────────────────────────────────
    rag.save_metrics_snapshot(metrics=metrics, severity=severity)

    # ── 6. Update state ─────────────────────────────────────────────────────
    return {
        **state,
        "metrics":    metrics,
        "severity":   severity,
        "thresholds": thresholds,
        "messages": state.get("messages", []) + [
            HumanMessage(
                content=(
                    f"[Monitor] severity={severity} "
                    f"accuracy={metrics.get('accuracy')} "
                    f"drift={metrics.get('drift_score')} "
                    f"latency_p99={metrics.get('latency_p99_ms')}ms "
                    f"error_rate={metrics.get('error_rate')} "
                    f"(method={classification_method}"
                    + (f" confidence={llm_confidence:.2f}" if classification_method == "llm" else "")
                    + (f" reason='{llm_reasoning[:80]}'" if llm_reasoning else "")
                    + ")"
                )
            )
        ],
    }
