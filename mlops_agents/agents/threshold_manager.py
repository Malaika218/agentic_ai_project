import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Self
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

from mlops_agents.llm_manager import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from mlops_agents.state import AgentState
from mlops_agents.rag.store import RAGStore

logger = logging.getLogger(__name__)

# Bounding boxes for clamping final outputs
THRESHOLD_LIMITS = {
    "accuracy_major":      (0.6, 0.85),
    "accuracy_critical":   (0.5, 0.75),
    "latency_major_ms":    (500.0, 3000.0),
    "latency_critical_ms": (1000.0, 5000.0),
    "error_rate_major":    (0.01, 0.2),
    "error_rate_critical": (0.05, 0.3),
    # Recall and ROC-AUC bands — critical must stay strictly below major.
    "recall_major":        (0.65, 0.85),
    "recall_critical":     (0.50, 0.70),
    "roc_auc_major":       (0.75, 0.92),
    "roc_auc_critical":    (0.65, 0.82),
}

# ---------------------------------------------------------------------------
# Split Validation Schema Architecture
# ---------------------------------------------------------------------------
class MetricDeltas(BaseModel):
    # --- Percentage Bounded Metrics (0.0 to 1.0 scales) ---
    accuracy_major: Optional[float] = Field(default=0.0, description="Delta adjustment for major accuracy.")
    accuracy_critical: Optional[float] = Field(default=0.0, description="Delta adjustment for critical accuracy.")
    error_rate_major: Optional[float] = Field(default=0.0, description="Delta adjustment for major error rate.")
    error_rate_critical: Optional[float] = Field(default=0.0, description="Delta adjustment for critical error rate.")
    recall_major: Optional[float] = Field(default=0.0, description="Delta adjustment for major recall.")
    recall_critical: Optional[float] = Field(default=0.0, description="Delta adjustment for critical recall.")
    roc_auc_major: Optional[float] = Field(default=0.0, description="Delta adjustment for major ROC-AUC.")
    roc_auc_critical: Optional[float] = Field(default=0.0, description="Delta adjustment for critical ROC-AUC.")

    # --- Millisecond Bounded Metrics (Absolute Time Scales) ---
    latency_major_ms: Optional[float] = Field(default=0.0, description="Delta adjustment for major latency in MILLISECONDS. Max step +/- 100.0ms.")
    latency_critical_ms: Optional[float] = Field(default=0.0, description="Delta adjustment for critical latency in MILLISECONDS. Max step +/- 200.0ms.")

    @model_validator(mode="after")
    def validate_individual_metric_dimensions(self) -> Self:
        """
        Splits validation gates based on real-world metric types.
        Prevents millisecond values from hitting percentage ceilings.
        """
        pct_fields = [
            "accuracy_major", "accuracy_critical",
            "error_rate_major", "error_rate_critical",
            "recall_major", "recall_critical",
            "roc_auc_major", "roc_auc_critical",
        ]
        
        # 1. Enforce strict 0.02 step limit on standard ratio metrics
        for field_name in pct_fields:
            val = getattr(self, field_name)
            if val is not None and (val < -0.02 or val > 0.02):
                raise ValueError(f"Percentage adjustment '{field_name}' ({val}) must be between -0.02 and +0.02")

        # 2. Enforce structural real-world millisecond adjustments for latency fields
        if self.latency_major_ms is not None and (self.latency_major_ms < -100.0 or self.latency_major_ms > 100.0):
            raise ValueError(f"Latency major adjustment ({self.latency_major_ms}ms) must be between -100.0ms and +100.0ms")
            
        if self.latency_critical_ms is not None and (self.latency_critical_ms < -200.0 or self.latency_critical_ms > 200.0):
            raise ValueError(f"Latency critical adjustment ({self.latency_critical_ms}ms) must be between -200.0ms and +200.0ms")

        return self

class ThresholdAdjustment(BaseModel):
    should_update: bool = Field(description="Whether thresholds should be updated")
    confidence: float = Field(description="Confidence score for the recommendation from 0.0 to 1.0")
    reasoning: str = Field(description="Detailed text explaining the decision context")
    adjustments: MetricDeltas = Field(description="Key-value pairs of metric names and their delta adjustments")


# ---------------------------------------------------------------------------
# Utility Metrics Processing Layer
# ---------------------------------------------------------------------------
def _clamp_thresholds(thresholds: dict) -> dict:
    for key, (low, high) in THRESHOLD_LIMITS.items():
        if key in thresholds:
            thresholds[key] = max(low, min(high, float(thresholds[key])))
    return thresholds

def _compute_metric_trend(trend: list[dict], metric_key: str) -> str:
    """Direction of a metric over the recent telemetry window (newest first)."""
    if len(trend) < 3: return "flat"
    vals = [m.get(metric_key) for m in trend[:5] if m.get(metric_key) is not None]
    if len(vals) < 3: return "flat"
    # Index 0 is newest. If index 0 < index -1 by >5%, the metric is declining.
    if vals[0] < vals[-1] * 0.95: return "declining"
    if vals[0] > vals[-1] * 1.05: return "improving"
    return "flat"


# Backwards-compatible alias.
def _compute_accuracy_trend(trend: list[dict]) -> str:
    return _compute_metric_trend(trend, "accuracy")

def _llm_threshold_advisor(
    state: AgentState,
    thresholds: dict,
    historical_stats: dict,
    trend: list[dict],
    tracker=None,
) -> dict:
    """Invokes structured compiler over low-token formatted datasets."""
    from tenacity import retry_if_exception_type
    llm = get_llm(temperature=0).with_structured_output(ThresholdAdjustment).with_retry(
        retry_if_exception_type=(ValidationError, Exception),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    metrics = state.get("metrics") or {}
    
    # NEW SECURITY/TOKEN FIX: Strip large raw statistical matrices before prompt insertion
    clean_metrics = {
        k: v for k, v in metrics.items() 
        if k not in ["reference_histograms", "production_histograms"]
    }

    prompt = f"""You are an adaptive ML system reliability coordinator.
Your task: Propose minor metric alert threshold adjustments to balance sensitivity and false-positive fatigue.

Rules:
- For ratio/percentage metrics (accuracy, error_rate, recall, roc_auc), max delta step is +/-0.02.
- For latency metrics (latency_major_ms, latency_critical_ms), propose real millisecond adjustments (e.g., +25.0, -50.0). Max steps are limited to 100ms/200ms.
- Higher-is-better metrics (accuracy, recall, roc_auc): NEGATIVE delta tightens (alerts earlier), POSITIVE relaxes.
- Lower-is-better metrics (error_rate, latency_*): POSITIVE delta tightens (alerts earlier), NEGATIVE relaxes.

Current baseline thresholds:
{json.dumps(thresholds, indent=2)}

Active incident parameters:
{json.dumps({
    "severity": state.get("severity"),
    "metrics": clean_metrics, # Low token count payload
    "diagnosis": state.get("diagnosis"),
    "remediation_status": state.get("remediation_status"),
}, indent=2)}

Historical metadata snapshots:
{json.dumps(historical_stats, indent=2)}

Recent sliding telemetry run trends:
{json.dumps(trend[:5], indent=2)}
"""
    logger.info("[Threshold Advisor] generated prompt:\n%s", prompt)
    invoke_kwargs = {"config": {"callbacks": [tracker]}} if tracker is not None else {}
    try:
        response = llm.invoke(
            [
                SystemMessage(content="You are an adaptive threshold schema tuner."),
                HumanMessage(content=prompt),
            ],
            **invoke_kwargs,
        )
        return response.model_dump()
    except Exception as exc:
        logger.info("[Threshold Advisor] all retry/backoff loops exhausted: %s", exc)
        return {"should_update": False, "confidence": 0.0, "reasoning": str(exc), "adjustments": {}}

# ---------------------------------------------------------------------------
# Main Execution Node Hook
# ---------------------------------------------------------------------------
def run_threshold_update(state: AgentState, rag: RAGStore, tracker=None) -> None:
    metrics = state.get("metrics") or {}
    metadata = metrics.get("metadata") or {}
    model_id = state.get("model_id") or metadata.get("model_name", "unknown")
    env = state.get("environment") or metadata.get("environment", "production")

    logger.info("[Threshold] starting — model_id=%s environment=%s", model_id, env)

    # Cooldown Gatekeeper Check
    existing = rag.get_dynamic_thresholds(model_id=model_id)
    now = datetime.now(timezone.utc)
    if existing and (now - datetime.fromisoformat(existing["updated_at"])).total_seconds() < 1800:
        logger.info(
            "[Threshold] skipped — cooldown active (last update at %s, < 1800s ago)",
            existing["updated_at"],
        )
        return

    # Extract reference metrics bounds
    thresholds = existing["thresholds"].copy() if existing else (state.get("thresholds") or {})
    if not thresholds: return
    
    hist_stats = rag.get_incident_stats(model_id=model_id, environment=env)
    trend = rag.query_recent_metrics(model_id=model_id, environment=env)
    
    # Learning Rate dampener
    lr = min(0.1, 1 / max(hist_stats.get("total", 1), 1))

    proposal = _llm_threshold_advisor(state, thresholds, hist_stats, trend, tracker=tracker)
    logger.info(
        "[Threshold] LLM proposal — should_update=%s confidence=%.2f reasoning=%s",
        proposal.get("should_update"),
        float(proposal.get("confidence", 0.0)),
        proposal.get("reasoning", "")[:200],
    )
    if not proposal.get("should_update"):
        logger.info("[Threshold] no update applied — proposal declined for model %s", model_id)
        return

    conf = max(0.0, min(1.0, float(proposal.get("confidence", 0.0))))
    applied = {}
    adjustments_dict = proposal.get("adjustments", {})
    
    for key, delta in adjustments_dict.items():
        if delta != 0.0 and key in thresholds:
            # Latency checks use raw adjustments, while percentage values are moderated by confidence and learning rates
            if "ms" in key:
                # Latency uses a faster scaling mechanism to avoid taking weeks to adapt
                eff_delta = delta * conf * min(1.0, lr * 10) 
            else:
                eff_delta = delta * conf * lr
                
            thresholds[key] += eff_delta
            applied[key] = eff_delta
    
    # Programmatic trend-protection step — for higher-is-better metrics, a
    # sustained decline tightens the major threshold so we alert sooner on the
    # next slide. Step size is dampened by the learning-rate `lr`.
    for metric in ("accuracy", "recall", "roc_auc"):
        major_key = f"{metric}_major"
        if major_key in thresholds and _compute_metric_trend(trend, metric) == "declining":
            logger.info(
                "[Threshold] declining %s trend detected — auto-tightening %s for %s",
                metric, major_key, model_id,
            )
            thresholds[major_key] -= (0.005 * lr)

    # Commit values back to database telemetry configurations
    clamped = _clamp_thresholds(thresholds)
    rag.save_dynamic_thresholds(model_id=model_id, thresholds={
        "model_id": model_id, "environment": env, "updated_at": now.isoformat(),
        "thresholds": clamped, "applied_adjustments": applied
    })
    logger.info(
        "[Threshold] updated — model_id=%s applied=%s new_thresholds=%s",
        model_id, applied, clamped,
    )