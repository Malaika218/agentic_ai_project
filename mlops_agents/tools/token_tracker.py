"""
mlops_agents/tools/token_tracker.py

Per-agent LLM token + cost tracking via a LangChain callback.

Why a callback instead of reading the response: agents commonly call
`.with_structured_output(Schema)` which returns the parsed Pydantic model
directly — the underlying `AIMessage.usage_metadata` is no longer in scope.
A callback fires on every `on_llm_end` regardless of wrapping (structured
output, retry, etc.) and gives us a single place to accumulate usage.

Cost rates default to per-1M-tokens. Override per-model via env:
    LLM_COST_GEMINI_2_0_FLASH_INPUT=0.10
    LLM_COST_GEMINI_2_0_FLASH_OUTPUT=0.40
Ollama (local) defaults to zero cost.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-1M token pricing (USD)
# ---------------------------------------------------------------------------

_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # Google Gemini (rates current as of 2026 — adjust per Gemini pricing page)
    "gemini-2.0-flash":      (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-1.5-flash":      (0.075, 0.30),
    "gemini-1.5-pro":        (1.25, 5.00),
    "gemini-2.5-flash":      (0.30, 2.50),
    "gemini-2.5-pro":        (1.25, 10.00),
    # Ollama / local models — zero monetary cost
}


def _normalise(model: str) -> str:
    """Strip path prefixes and version qualifiers from model names."""
    name = (model or "").split("/")[-1].split(":")[0].lower()
    return name


def _resolve_pricing(model: str) -> tuple[float, float]:
    """Return (input_rate, output_rate) per 1M tokens. Env overrides default."""
    base = _normalise(model)
    env_key = base.upper().replace("-", "_").replace(".", "_")
    env_in = os.getenv(f"LLM_COST_{env_key}_INPUT")
    env_out = os.getenv(f"LLM_COST_{env_key}_OUTPUT")
    if env_in is not None and env_out is not None:
        try:
            return float(env_in), float(env_out)
        except ValueError:
            logger.info("Invalid env pricing for %s — using defaults", base)
    return _DEFAULT_PRICING.get(base, (0.0, 0.0))


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _resolve_pricing(model)
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------

class TokenUsageHandler(BaseCallbackHandler):
    """
    Accumulate token counts across every LLM call this handler is attached to.

    Usage in an agent:
        tracker = TokenUsageHandler()
        result = llm.invoke(msgs, config={"callbacks": [tracker]})
        result2 = llm.invoke(msgs2, config={"callbacks": [tracker]})
        usage = tracker.summary()   # {input_tokens, output_tokens, total_tokens,
                                    #  calls, model, cost_usd}
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.calls: int = 0
        self.model_name: str | None = None

    def on_llm_end(self, response, **kwargs: Any) -> None:  # noqa: D401
        """Pull usage_metadata from each generation's AIMessage."""
        self.calls += 1

        # New-style: AIMessage.usage_metadata = {input_tokens, output_tokens, total_tokens}
        for generations in response.generations or []:
            for gen in generations:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    meta = getattr(msg, "usage_metadata", None)
                    if meta:
                        self.input_tokens  += int(meta.get("input_tokens", 0) or 0)
                        self.output_tokens += int(meta.get("output_tokens", 0) or 0)
                    rm = getattr(msg, "response_metadata", None) or {}
                    if self.model_name is None:
                        self.model_name = rm.get("model_name") or rm.get("model")

        # Older-style fallback: LLMResult.llm_output.token_usage
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if usage:
            self.input_tokens  += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        if self.model_name is None:
            self.model_name = llm_output.get("model_name") or llm_output.get("model")

    def summary(self) -> dict:
        """Return a serialisable usage record for persistence in state."""
        model = (
            self.model_name
            or os.getenv("GOOGLE_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or "unknown"
        )
        return {
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.input_tokens + self.output_tokens,
            "calls":         self.calls,
            "model":         model,
            "cost_usd":      round(compute_cost_usd(model, self.input_tokens, self.output_tokens), 6),
        }
