# agents/diagnosis_agent.py
from __future__ import annotations
import json
import logging
import os
from typing import Any, Literal, Optional
from mlops_agents.llm_manager import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mlops_agents.state import AgentState
from mlops_agents.rag.store import RAGStore
from mlops_agents.tools.histogram_drift import compute_histogram_drift
from mlops_agents.tools.token_tracker import TokenUsageHandler

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
    return get_llm(temperature=0).with_structured_output(DiagnosisOutput)


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
        "[Diagnosis] starting — model_id=%s environment=%s severity=%s",
        model_id, environment, severity,
    )

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

    # ── Deterministic drift quantification via @tool ────────────────────────
    drift = compute_histogram_drift.invoke({
        "reference": ref_histograms or {},
        "production": prod_histograms or {},
    })
    logger.info(
        "[Diagnosis] drift — %s drifted_features=%s",
        drift["summary"], drift["drifted_features"][:10],
    )

    # Compact, LLM-readable rendering of the drift result. Raw histograms are
    # no longer pushed into the prompt — the math already happened in code.
    if drift["top_drifted"]:
        drift_table = "\n".join(
            f"- {row['feature']}: PSI={row['psi']:.3f} KS={row['ks']:.3f} "
            f"mean_shift_z={row['mean_shift_z']:.2f} → {row['drift_level']}"
            for row in drift["top_drifted"]
        )
    else:
        drift_table = "No features exceed the moderate-drift threshold (PSI ≥ 0.10)."

    # Build context lookup keys for RAG spaces
    query_text = (
        f"Model {model_id} in {environment}. Severity: {severity}. "
        f"Accuracy: {metrics.get('accuracy')}."
    )

    # Execute RAG Retrieval
    similar_incidents = rag.query_similar_incidents(query_text=query_text, n_results=3, where={
                                                    "environment": environment} if environment != "unknown" else None)
    relevant_runbooks = rag.query_runbooks(query_text=query_text, n_results=2)
    trend = rag.query_recent_metrics(
        model_id=model_id, n_results=5, environment=environment)

    logger.info(
        "[Diagnosis] RAG retrieval — similar_incidents=%d runbooks=%d trend_points=%d",
        len(similar_incidents or []), len(relevant_runbooks or []), len(trend or []),
    )

    system_prompt = "You are an autonomous MLOps Diagnostic engine. Synthesize incident logs, vector runbooks, and pre-computed drift statistics to resolve root cause anomalies into structured formats."

    prompt = f"""## Active Operational Incident Context
Model Name: {model_id}
Environment Context: {environment}
Severity Tier: {severity}

### Dynamic Performance Telemetry Signals
{json.dumps({k: v for k, v in metrics.items() if k not in ['reference_histograms', 'production_histograms']}, indent=2)}

### Feature Distribution Drift (pre-computed via PSI / KS / mean-shift)
Summary: {drift["summary"]}

Top drifted features (PSI ≥ 0.10 — production vs. training):
{drift_table}

Drift bands: PSI < 0.10 stable · 0.10–0.25 moderate · ≥ 0.25 significant.
Use these statistics as ground truth — do NOT re-derive them. The list above
is sorted by PSI descending.

### Institutional Knowledge Base (RAG Matches)
Similar Past Operational Outages:
{_format_similar_incidents(similar_incidents)}

Mapped Execution Runbooks & Post-Mortems:
{_format_runbooks(relevant_runbooks)}

Recent Telemetry History Trend Line:
{_format_trend(trend)}

NOTE: The possible actions you can recommend are strictly limited to the following:
- "retrain": The model shows signs of concept or data drift. Recommend a retraining with a prescription for data selection and retraining parameters.
- "scale": The model itself is sound but is experiencing infrastructure-related performance bottlenecks. Recommend scaling up resources or optimizing deployment.
- "rollback": A recent model update introduced instability. Recommend rolling back to the last known good model version.
- "investigate": The root cause is unclear or doesn't fit known patterns. Recommend a manual investigation by the engineering team.

- Confidence: should be between 0.0 and 1.0
- Root Cause Categories: concept_drift | data_drift | model_staleness | infrastructure | data_quality | unknown
- Retrain Prescription: Only include if recommending "retrain". Should specify data strategy, window size, drifted features, and optimization targets.

Instructions: Formulate a cohesive root cause evaluation by contrasting data histograms against baseline shapes and historical runbooks. Output strictly via the required JSON target model configuration layout."""

    llm = _build_diagnosis_llm()
    tracker = TokenUsageHandler()

    try:
        result: DiagnosisOutput = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ],
            config={"callbacks": [tracker]},
        )
    except Exception as exc:
        logger.info(
            "[Diagnosis] structured LLM call failed — falling back to investigate. exc=%s", exc)
        return {
            **state,
            "diagnosis": "Fallback: Suspected performance degradation triggered safety loop.",
            "token_usage": {"diagnosis": tracker.summary()},
            "remediation_action": "investigate"
        }

    logger.info(
        "[Diagnosis] result — root_cause=%r category=%s action=%s confidence=%.2f",
        result.root_cause, result.root_cause_category,
        result.recommended_action, result.confidence,
    )
    logger.info("[Diagnosis] full structured output: %s", result.model_dump_json())

    # Materialize prescription parameters for safe hand-off to remediation agent tools.
    # The tool's drifted_features list is the authoritative one — overlay it onto
    # the LLM's prescription so retraining targets the features the maths flagged.
    prescription = result.retrain_prescription.model_dump(
    ) if result.retrain_prescription else None
    drifted = drift["drifted_features"] or (
        prescription.get("drifted_features", []) if prescription else []
    )
    if prescription is not None:
        prescription["drifted_features"] = drifted
        logger.info(
            "[Diagnosis] retrain prescription — strategy=%s window_days=%s "
            "drift_period_weight=%s exclude_before=%r refit_preprocessors=%s "
            "optimize_for=%s target_recall=%s target_roc_auc=%s "
            "deployment_strategy=%s canary_pct=%s shadow_hours=%s "
            "drifted_features=%s",
            prescription.get("data_strategy"),
            prescription.get("window_days"),
            prescription.get("drift_period_weight"),
            prescription.get("exclude_before"),
            prescription.get("refit_preprocessors"),
            prescription.get("optimize_for"),
            prescription.get("target_recall"),
            prescription.get("target_roc_auc"),
            prescription.get("deployment_strategy"),
            prescription.get("canary_traffic_pct"),
            prescription.get("shadow_period_hours"),
            drifted,
        )

    # Map the output tokens to fit your updated remediation routing signatures
    action_map = {
        "retrain": "trigger_retraining",
        "scale": "scale_infrastructure",
        "rollback": "rollback",
        "investigate": "investigate"
    }
    workflow_action_token = action_map.get(
        result.recommended_action, "investigate")

    logger.info(
        "[Diagnosis] complete — workflow_action=%s drifted_features=%s prescription=%s",
        workflow_action_token,
        drifted,
        "present" if prescription else "none",
    )

    token_summary = tracker.summary()
    logger.info(
        "[Diagnosis] token usage — in=%d out=%d total=%d calls=%d model=%s cost=$%.6f",
        token_summary["input_tokens"], token_summary["output_tokens"],
        token_summary["total_tokens"], token_summary["calls"],
        token_summary["model"], token_summary["cost_usd"],
    )

    per_feature = drift.get("per_feature", {})
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
        "per_feature_psi":      {f: m["psi"] for f, m in per_feature.items()},
        "per_feature_ks":       {f: m["ks"]  for f, m in per_feature.items()},
        "similar_incidents":    similar_incidents,
        "relevant_runbooks":    relevant_runbooks,
        "token_usage":          {"diagnosis": token_summary},
        "messages": state.get("messages", []) + [HumanMessage(content=f"[Diagnosis] Cause='{result.root_cause}' Category={result.root_cause_category} Action={result.recommended_action}")
        ]
        }
