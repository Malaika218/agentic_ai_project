import json
import logging
import os
from datetime import datetime, timezone
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from mlops_agents.rag.store import RAGStore
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional
from langchain_core.runnables import RunnableRetry

logger = logging.getLogger(__name__)

THRESHOLD_LIMITS = {
    "accuracy_major": (0.6, 0.85),
    "accuracy_critical": (0.5, 0.75),
    "drift_major": (0.1, 0.7),
    "drift_critical": (0.3, 0.9),
    "latency_major_ms": (500, 3000),
    "latency_critical_ms": (1000, 5000),
    "error_rate_major": (0.01, 0.2),
    "error_rate_critical": (0.05, 0.3),
}

class MetricDeltas(BaseModel):
    # Every valid threshold key is explicitly mapped as an optional float defaulting to 0.0
    accuracy_major: Optional[float] = Field(default=0.0, description="Delta adjustment for major accuracy")
    accuracy_critical: Optional[float] = Field(default=0.0, description="Delta adjustment for critical accuracy")
    drift_major: Optional[float] = Field(default=0.0, description="Delta adjustment for major drift")
    drift_critical: Optional[float] = Field(default=0.0, description="Delta adjustment for critical drift")
    latency_major_ms: Optional[float] = Field(default=0.0, description="Delta adjustment for major latency ms")
    latency_critical_ms: Optional[float] = Field(default=0.0, description="Delta adjustment for critical latency ms")
    error_rate_major: Optional[float] = Field(default=0.0, description="Delta adjustment for major error rate")
    error_rate_critical: Optional[float] = Field(default=0.0, description="Delta adjustment for critical error rate")

    @field_validator('*')
    @classmethod
    def validate_deltas(cls, v: Optional[float]) -> Optional[float]:
        """Enforces the maximum delta ceiling at the parser layer."""
        if v is not None and (v < -0.02 or v > 0.02):
            raise ValueError("Delta adjustment must be strictly between -0.02 and 0.02")
        return v

class ThresholdAdjustment(BaseModel):
    should_update: bool = Field(description="Whether thresholds should be updated")
    confidence: float = Field(description="Confidence score for the recommendation from 0.0 to 1.0")
    reasoning: str = Field(description="Detailed text explaining the decision context")
    adjustments: MetricDeltas = Field(description="Key-value pairs of metric names and their delta adjustments")

MAX_THRESHOLD_DELTA = 0.02

def _clamp_thresholds(thresholds: dict) -> dict:
    for key, (low, high) in THRESHOLD_LIMITS.items():
        if key in thresholds:
            thresholds[key] = max(low, min(high, thresholds[key]))
    return thresholds

def _compute_drift_trend(trend: list[dict]) -> str:
    if len(trend) < 3: return "flat"
    vals = [m.get("drift_score", 0.0) for m in trend[:5] if m.get("drift_score") is not None]
    if len(vals) < 3: return "flat"
    if vals[0] > vals[-1] * 1.1: return "increasing"
    if vals[0] < vals[-1] * 0.9: return "decreasing"
    return "flat"

def _llm_threshold_advisor(
    state: AgentState,
    thresholds: dict,
    historical_stats: dict,
    trend: list[dict],
) -> dict:
    """Restored full LLM prompt with trend context using strict Ollama JSON schema."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # 2. Initialize the model and bind the structured output schema
    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0,
    ).with_structured_output(ThresholdAdjustment)

    # 2. Wrap it with built-in LangChain exponential backoff retry logic
    llm = llm.with_retry(
        retry_exception_types=(ValidationError, Exception),  # Catches schema + network drops
        max_attempt_number=3,                                # Total try attempts
        wait_exponential_jitter=True                         # Applies exponential delay with jitter
    )

    metrics = state.get("metrics") or {}

    # Prompt text remains clean since formatting rules are handled at the schema level
    prompt = f"""
You are an adaptive ML reliability agent.

Your task:
Recommend SMALL monitoring threshold adjustments.

Goals:
- reduce false positives
- detect degradation earlier
- avoid unnecessary retraining

Rules:
- only propose small deltas
- max delta per threshold is +/-0.02

Current thresholds:
{json.dumps(thresholds, indent=2)}

Current incident:
{json.dumps({
    "severity": state.get("severity"),
    "metrics": metrics,
    "diagnosis": state.get("diagnosis"),
    "remediation_status": state.get("remediation_status"),
    "human_approved": state.get("human_approved"),
}, indent=2)}

Historical stats:
{json.dumps(historical_stats, indent=2)}

Recent trend:
{json.dumps(trend[:5], indent=2)}
"""
    try:
        # 3. Invoke the structured model; it directly returns a Pydantic object
        response = llm.invoke([
            SystemMessage(content="You are an adaptive threshold tuning agent."),
            HumanMessage(content=prompt),
        ])
        
        # Convert the parsed model back into a standard dictionary
        return response.model_dump()
        
    except Exception as exc:
        # Triggers only if all 3 backoff attempts fail completely
        logger.error("All threshold advisor backoff retries exhausted: %s", exc)
        return {"should_update": False, "confidence": 0.0, "reasoning": str(exc), "adjustments": {}}
    
def run_threshold_update(state: AgentState, rag: RAGStore) -> None:
    """Main entry point for the threshold learning subsystem."""
    metrics = state.get("metrics") or {}
    model_id, env = metrics.get("model_id", "unknown"), metrics.get("environment", "production")

    # Cooldown Check
    existing = rag.get_dynamic_thresholds(model_id=model_id)
    now = datetime.now(timezone.utc)
    if existing and (now - datetime.fromisoformat(existing["updated_at"])).total_seconds() < 1800:
        return

    # Data Gathering
    thresholds = existing["thresholds"].copy() if existing else (state.get("thresholds") or {})
    if not thresholds: return
    
    hist_stats = rag.get_incident_stats(model_id=model_id, environment=env)
    trend = rag.query_recent_metrics(model_id=model_id, environment=env)
    
    # Logic: Learning Rate & Advisor
    lr = min(0.1, 1 / max(hist_stats.get("total", 1), 1))

    # The proposal is guaranteed to only contain valid keys and legal bounds
    proposal = _llm_threshold_advisor(state, thresholds, hist_stats, trend)

    if not proposal.get("should_update"): return

    conf = max(0.0, min(1.0, float(proposal.get("confidence", 0.0))))
    applied = {}
    # Access fields directly and safely using model_dump()
    adjustments_dict = proposal.get("adjustments", {})
    for key, delta in adjustments_dict.items():
        if delta != 0.0 and key in thresholds:  # Only process intentional changes
            eff_delta = delta * conf * lr
            thresholds[key] += eff_delta
            applied[key] = eff_delta
    
    # Trend awareness override
    if _compute_drift_trend(trend) == "increasing":
        thresholds["drift_major"] -= 0.01 * lr

    # Persist
    rag.save_dynamic_thresholds(model_id=model_id, thresholds={
        "model_id": model_id, "environment": env, "updated_at": now.isoformat(),
        "thresholds": _clamp_thresholds(thresholds), "applied_adjustments": applied
    })