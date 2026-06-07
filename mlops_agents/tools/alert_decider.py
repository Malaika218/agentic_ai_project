"""
mlops_agents/tools/alert_decider.py

LangChain-backed alert decision tool.

This module exposes a simple function `decide_slack_alert` which queries the
LLM to decide whether a given incident report should trigger an immediate
Slack alert. The function returns a dict with keys `alert` (bool) and
`reason` (str). Token usage is recorded via the provided `tracker` callback.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from mlops_agents.llm_manager import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from mlops_agents.tools.token_tracker import TokenUsageHandler

logger = logging.getLogger(__name__)


def decide_slack_alert(state: Dict[str, Any], report: str, tracker: TokenUsageHandler) -> Dict[str, Any]:
    """Query the LLM to decide whether to send a Slack alert.

    Returns:
        {"alert": bool, "reason": str}
    """

    llm = get_llm(temperature=0)

    system = SystemMessage(content=(
        "You are an MLOps assistant that decides whether incidents warrant an immediate Slack alert."
    ))

    # Keep the prompt concise but include the report context (truncated).
    truncated = report[:6000]
    human = HumanMessage(content=(
        "Decide whether the following incident report should trigger an immediate Slack alert."
        " Reply with a JSON object exactly like {\"alert\": true|false, \"reason\": \"short rationale\"}.\n\n"
        "Report:\n\n" + truncated
    ))

    try:
        resp = llm.invoke([system, human], config={"callbacks": [tracker]})
        text = resp.content.strip()

        # Try to parse JSON output first.
        try:
            parsed = json.loads(text)
            return {"alert": bool(parsed.get("alert")), "reason": str(parsed.get("reason", ""))}
        except Exception:
            # Fallback: interpret freeform yes/no answers.
            low = text.lower()
            if any(k in low for k in ("yes", "alert", "true", "send")):
                return {"alert": True, "reason": text}
            return {"alert": False, "reason": text}

    except Exception as exc:  # pragma: no cover - runtime environment dependent
        logger.info("Alert decision LLM failed: %s", exc)
        # Fail-open: if the LLM fails, default to alerting so incidents aren't missed.
        return {"alert": True, "reason": "llm_error_default_alert"}
