# agents/diagnosis_agent.py
from __future__ import annotations
import json
import logging
import os
from typing import Any, Literal, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from state import AgentState
from mlops_agents.rag.store import RAGStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YOUR ORIGINAL TYPE LITERALS (Preserved Exactly)
# ---------------------------------------------------------------------------
RecommendedAction = Literal["retrain", "rollback", "scale", "investigate"]
RootCauseCategory = Literal[
    "concept_drift", "data_drift", "model_staleness",
    "infrastructure", "data_quality", "unknown"
]
DataStrategy = Literal["recent_window", "full_history",
                       "weighted_recent", "drift_period_only"]
OptimizeFor = Literal["f2_score", "roc_auc", "recall", "precision"]
DeploymentStrategy = Literal["canary", "blue_green", "immediate"]
DriftDatasets = Literal["baseline", "data_drift_amount", "data_drift_features", "concept_drift",
                        "mixed_drift", "gradual_drift_step_1", "gradual_drift_step_2", "gradual_drift_step_3"]

# ---------------------------------------------------------------------------
# YOUR ORIGINAL PYDANTIC SCHEMAS (Preserved Exactly)
# ---------------------------------------------------------------------------


class RetrainPrescription(BaseModel):
    data_strategy: DataStrategy = Field(
        default="recent_window",
        description="How to select training data for the retrain run.",
    )
    window_days: int = Field(
        default=30,
        description="How many calendar days of data to include.",
    )
    drift_period_weight: float = Field(
        default=1.5,
        description="Upsampling multiplier for drift-period records (1.0 = no upsampling).",
    )
    exclude_before: str = Field(
        default="",
        description="ISO-8601 date; exclude data before this date. Empty string = no exclusion.",
    )
    refit_preprocessors: bool = Field(
        default=True,
        description="Whether to refit scalers/encoders on the new training window.",
    )
    drifted_features: list[str] = Field(
        default_factory=list,
        description="Feature names that showed statistically significant drift.",
    )
    optimize_for: OptimizeFor = Field(
        default="recall",
        description="Primary metric to optimise during hyperparameter search.",
    )
    target_recall: float = Field(
        default=0.80,
        description="Minimum acceptable recall on the validation set.",
    )
    target_roc_auc: float = Field(
        default=0.88,
        description="Minimum acceptable ROC-AUC on the validation set.",
    )
    deployment_strategy: DeploymentStrategy = Field(
        default="canary",
        description="How to promote the retrained model to production.",
    )
    canary_traffic_pct: int = Field(
        default=10,
        description="Percentage of traffic routed to the canary model (1–50).",
    )
    shadow_period_hours: int = Field(
        default=2,
        description="Hours to run the new model in shadow mode before promotion.",
    )
    drift_dataset: DriftDatasets = Field(
        default="baseline",
        description="Which dataset to use for retraining"
    )


class DiagnosisOutput(BaseModel):
    root_cause: str = Field(
        description="Concise one-sentence root cause of the degradation.",
    )
    root_cause_category: RootCauseCategory = Field(
        default="unknown",
        description=(
            "Category of root cause: concept_drift | data_drift | model_staleness "
            "| infrastructure | data_quality | unknown"
        ),
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="List of evidence points supporting the root cause.",
    )
    recommended_action: RecommendedAction = Field(
        description="Remediation action: retrain | rollback | scale | investigate",
    )
    confidence: float = Field(
        description="Confidence score for this diagnosis, between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="2-3 sentence reasoning chain connecting evidence to recommendation.",
    )
    retrain_prescription: Optional[RetrainPrescription] = Field(
        default=None,
        description="Retraining parameters — only populated when recommended_action is 'retrain'.",
    )

# ---------------------------------------------------------------------------
# LLM Initializer & RAG Formatting Layout Helpers
# ---------------------------------------------------------------------------


def _build_diagnosis_llm():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model_name, base_url=ollama_url, temperature=0
    ).with_structured_output(DiagnosisOutput)


def _format_similar_incidents(incidents: list) -> str:
    if not incidents:
        return "No matching past historical incidents found."
    return "\n".join([f"- Incident ID: {idx}\n  Summary: {doc}" for idx, doc in enumerate(incidents)])


def _format_runbooks(runbooks: list) -> str:
    if not runbooks:
        return "No active playbook execution context mapped."
    return "\n".join([f"- Playbook doc matches:\n  {doc}" for doc in runbooks])


def _format_trend(trend: list) -> str:
    if not trend:
        return "Telemetry trend snapshot sequence empty."
    return json.dumps(trend, indent=2)

# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def diagnosis_agent(state: AgentState, rag: RAGStore) -> AgentState:
    metrics: dict = state.get("metrics") or {}
    severity: str = state.get("severity", "minor")

    metadata = metrics.get("metadata") or {}
    model_id: str = state.get("model_id") or metadata.get(
        "model_name", "unknown")
    environment: str = state.get("environment", "production")

    logger.info(
        "Diagnosis Agent: Executing hybrid analysis for model: %s", model_id)

    # Short-circuit logic if monitor flags no anomalies
    if severity == "none":
        return {
            **state,
            "diagnosis": "System boundaries verified. Performance levels nominal.",
            "remediation_action": "none",
            "messages": state.get("messages", []) + [HumanMessage(content="[Diagnosis] Bypassed. Baseline secure.")]
        }

    # Extract our low-token, zero-PII data summaries from state cache
    ref_histograms = metrics.get("reference_histograms")
    prod_histograms = metrics.get("production_histograms")

    # Build context lookup keys for RAG spaces
    query_text = (
        f"Model {model_id} in {environment}. Severity: {severity}. "
        f"Accuracy: {metrics.get('accuracy')}. Drift score: {metrics.get('drift_score')}."
    )

    # Execute RAG Retrieval
    similar_incidents = rag.query_similar_incidents(query_text=query_text, n_results=3, where={
                                                    "environment": environment} if environment != "unknown" else None)
    relevant_runbooks = rag.query_runbooks(query_text=query_text, n_results=2)
    trend = rag.query_recent_metrics(
        model_id=model_id, n_results=5, environment=environment)

    system_prompt = "You are an autonomous MLOps Diagnostic engine. Synthesize incident logs, vector runbooks, and abstract feature distributions to resolve root cause anomalies into structured formats."

    prompt = f"""## Active Operational Incident Context
Model Name: {model_id}
Environment Context: {environment}
Severity Tier: {severity}

### Dynamic Performance Telemetry Signals
{json.dumps({k: v for k, v in metrics.items() if k not in ['reference_histograms', 'production_histograms']}, indent=2)}

### Statistical Feature Distribution Shapes (Zero-PII Summaries)
Reference Baseline Histograms (Training Set Footprint):
{json.dumps(ref_histograms) if ref_histograms else "UNAVAILABLE"}

Live Production Histograms (Recent Sliding Buffer Footprint):
{json.dumps(prod_histograms) if prod_histograms else "UNAVAILABLE"}

### Institutional Knowledge Base (RAG Matches)
Similar Past Operational Outages:
{_format_similar_incidents(similar_incidents)}

Mapped Execution Runbooks & Post-Mortems:
{_format_runbooks(relevant_runbooks)}

Recent Telemetry History Trend Line:
{_format_trend(trend)}

Instructions: Formulate a cohesive root cause evaluation by contrasting data histograms against baseline shapes and historical runbooks. Output strictly via the required JSON target model configuration layout."""

    llm = _build_diagnosis_llm()

    try:
        result: DiagnosisOutput = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
    except Exception as exc:
        logger.error(
            "Structured LLM compilation failure. Executing safe hardcoded fallback: %s", exc)
        return {
            **state,
            "diagnosis": "Fallback: Suspected performance degradation triggered safety loop.",
            "remediation_action": "investigate"
        }

    # Materialize prescription parameters for safe hand-off to remediation agent tools
    prescription = result.retrain_prescription.model_dump(
    ) if result.retrain_prescription else None
    drifted = prescription.get("drifted_features", []) if prescription else []

    # Map the output tokens to fit your updated remediation routing signatures
    action_map = {
        "retrain": "trigger_retraining",
        "scale": "scale_infrastructure",
        "rollback": "rollback",
        "investigate": "investigate"
    }
    workflow_action_token = action_map.get(
        result.recommended_action, "investigate")

    return {
        **state,
        "diagnosis":            result.root_cause,
        "diagnosis_json":       result.model_dump(),
        # Maps back into workflow.py branches
        "remediation_action":   workflow_action_token,
        # Maintained for compatibility strings
        "recommended_action":   result.recommended_action,
        "retrain_prescription": prescription,
        "drifted_features":     drifted,
        "similar_incidents":    similar_incidents,
        "relevant_runbooks":    relevant_runbooks,
        "messages": state.get("messages", []) + [HumanMessage(content=f"[Diagnosis] Cause='{result.root_cause}' Category={result.root_cause_category} Action={result.recommended_action}")
        ]
        }
