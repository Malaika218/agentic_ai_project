"""
mlops_agents/tools/severity_classifier.py

Deterministic severity classifier exposed as a LangChain @tool.

Why this exists: small LLMs (e.g. llama3.2:1b) are unreliable at directional
threshold logic ("LOWER is BAD" vs "HIGHER is BAD"). The classification is a
textbook rule-based comparison — we do it in Python and let the LLM only
narrate the result afterwards.

Returned shape:
    {
        "severity": "none" | "minor" | "major" | "critical",
        "breaches": [
            {"metric": "accuracy", "value": 0.71, "threshold": 0.72,
             "level": "major", "direction": "<"},
            ...
        ],
        "trend_note": "<short text>",
    }
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Threshold spec — direction tells us which way a breach goes.
#   "<" → metric should be HIGH; breach when value < threshold (accuracy, etc.)
#   ">" → metric should be LOW;  breach when value > threshold (latency, errors)
# ---------------------------------------------------------------------------
_RULES: list[tuple[str, str, str, str]] = [
    # (metric_key, critical_threshold_key, major_threshold_key, direction)
    ("accuracy",   "accuracy_critical",   "accuracy_major",   "<"),
    ("error_rate", "error_rate_critical", "error_rate_major", ">"),
    ("latency_ms", "latency_critical_ms", "latency_major_ms", ">"),
    # Recall and ROC-AUC are the load-bearing signals under concept drift on
    # imbalanced data — accuracy stays high while the model silently degrades.
    ("recall",     "recall_critical",     "recall_major",     "<"),
    ("roc_auc",    "roc_auc_critical",    "roc_auc_major",    "<"),
]

_SEVERITY_ORDER = {"none": 0, "minor": 1, "major": 2, "critical": 3}


def _breach(value: float, threshold: float, direction: str) -> bool:
    """Return True iff `value` violates the threshold in the given direction."""
    return value < threshold if direction == "<" else value > threshold


def _classify(metrics: dict, thresholds: dict, trend: list[dict] | None) -> dict:
    """Pure-Python classification — also used directly by the monitor agent."""
    breaches: list[dict] = []
    severity = "none"

    for metric_key, crit_key, major_key, direction in _RULES:
        value = metrics.get(metric_key)
        # Fall back to p99 latency if the bare latency_ms key is absent.
        if value is None and metric_key == "latency_ms":
            value = metrics.get("latency_p99_ms")
        if value is None:
            continue

        crit_thr = thresholds.get(crit_key)
        major_thr = thresholds.get(major_key)

        if crit_thr is not None and _breach(value, crit_thr, direction):
            breaches.append({
                "metric": metric_key, "value": value, "threshold": crit_thr,
                "level": "critical", "direction": direction,
            })
            severity = "critical"
        elif major_thr is not None and _breach(value, major_thr, direction):
            breaches.append({
                "metric": metric_key, "value": value, "threshold": major_thr,
                "level": "major", "direction": direction,
            })
            if _SEVERITY_ORDER[severity] < _SEVERITY_ORDER["major"]:
                severity = "major"

    # Trend-based "minor" only kicks in when nothing else breached.
    trend_note = "no historical trend available"
    if severity == "none" and trend:
        prior_accuracies = [t.get("accuracy") for t in trend if t.get("accuracy") is not None]
        current_acc = metrics.get("accuracy")
        if current_acc is not None and len(prior_accuracies) >= 2:
            baseline = sum(prior_accuracies) / len(prior_accuracies)
            # >2% relative drop vs trend baseline = minor degradation
            if baseline > 0 and (baseline - current_acc) / baseline > 0.02:
                severity = "minor"
                trend_note = (
                    f"accuracy {current_acc:.4f} is >2% below trend baseline "
                    f"{baseline:.4f} (n={len(prior_accuracies)})"
                )
            else:
                trend_note = (
                    f"accuracy {current_acc:.4f} within 2% of trend baseline "
                    f"{baseline:.4f}"
                )

    return {"severity": severity, "breaches": breaches, "trend_note": trend_note}


@tool
def classify_severity(
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    trend: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Classify operational severity by comparing live metrics against thresholds.

    Rules:
      - "critical" if any critical threshold is breached
      - "major"    if any major threshold is breached (no critical breach)
      - "minor"    if no static breach but accuracy is >2% below trend baseline
      - "none"     otherwise

    Args:
        metrics: Live snapshot. Expected keys (any subset): accuracy, error_rate,
                 latency_ms (or latency_p99_ms as fallback).
        thresholds: Threshold dict — must contain accuracy_critical / accuracy_major,
                    error_rate_critical / error_rate_major, latency_critical_ms /
                    latency_major_ms (keys missing from thresholds are skipped).
        trend: Optional list of prior metrics snapshots (newest first).

    Returns:
        {"severity": ..., "breaches": [...], "trend_note": ...}
    """
    return _classify(metrics or {}, thresholds or {}, trend or [])
