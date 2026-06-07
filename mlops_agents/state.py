"""
state.py

Shared LangGraph AgentState.

Every node reads from and writes to this TypedDict.
All fields are optional at construction time so nodes can populate them
incrementally as the pipeline progresses.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages


def _merge_token_usage(left: dict, right: dict) -> dict:
    """Reducer: merge per-agent token-usage dicts by agent name.

    Each agent emits its own slice (e.g. {"monitor": {...}}); the reducer
    composes them so the final state.token_usage holds every agent's
    contribution to the run.
    """
    if not left:
        return dict(right or {})
    if not right:
        return dict(left or {})
    merged = dict(left)
    merged.update(right)  # later wins; agents only write their own keys
    return merged


class AgentState(TypedDict, total=False):
    # ── Runtime source metadata ──────────────────────────────────────────────
    # Injected by the entry-point before the graph is invoked.
    # Identifies which model deployment triggered this pipeline run.
    model_id: str                    # e.g. "fraud-classifier-v2"
    environment: str                 # production | staging | canary

    # ── Raw metrics (populated by Monitor Agent) ─────────────────────────────
    # Full snapshot dict as returned by the metrics data source integration.
    metrics: Optional[dict[str, Any]]

    # ── Severity classification (Monitor Agent) ──────────────────────────────
    severity: str                    # none | minor | major | critical

    # ── Diagnosis (Diagnosis Agent) ──────────────────────────────────────────
    diagnosis: str                   # Free-text root cause summary
    diagnosis_json: Optional[dict]   # Parsed structured JSON from LLM
    similar_incidents: Optional[list[dict]]   # Top-k RAG results
    relevant_runbooks: Optional[list[dict]]   # Top-k runbook RAG results

    # ── Remediation (Remediation Agent) ─────────────────────────────────────
    recommended_action: str          # retrain | rollback | scale | investigate
    remediation_action: str          # action actually executed
    remediation_status: str          # success | failed | skipped
    remediation_detail: Optional[str]  # tool response or error message

    # ── Human-in-the-loop ────────────────────────────────────────────────────
    human_approved: bool             # True once a human approves a major incident

    # ── Reporting (Reporting Agent) ──────────────────────────────────────────
    report: str                      # Final markdown incident report
    notifications_sent: Optional[list[str]]  # ["slack", "email", …]
    incident_id: Optional[str]       # ChromaDB incident ID after save
    postmortem_runbook_id: Optional[str]  # Set when LLM gate promoted incident to a runbook

    # ── LangGraph message history ────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Retrain prescription (Diagnosis Agent → Remediation Agent) ──────────
    retrain_prescription: Optional[dict]   # full structured prescription
    drift_onset_at: Optional[str]          # ISO-8601 — when drift started
    
    reference_histograms:  Optional[dict]   # per-feature reference from training
    production_histograms: Optional[dict]   # per-feature current production
    per_feature_psi:       Optional[dict]   # PSI per feature — populated by diagnosis agent
    per_feature_ks:        Optional[dict]   # KS statistic per feature
    drifted_features:      Optional[list[str]]  # already exists — now computed not guessed

    # ── LLM token + cost tracking ────────────────────────────────────────────
    # Per-agent dict: {"monitor": {input_tokens, output_tokens, calls, model, cost_usd, ...}, ...}
    # Reducer merges contributions from every agent that ran during the pipeline.
    token_usage: Annotated[dict[str, dict], _merge_token_usage]
