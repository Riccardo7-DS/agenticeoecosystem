"""
pricing.py — USD cost estimates for paid model API usage, for tracking spend
during benchmark generation runs.

Prices are $ per 1M tokens, (input, output), taken from each vendor's own
pricing page at the time they were added below. Self-hosted/free models
(EVE-Instruct, Mistral-Small, etc.) are intentionally absent — estimate_cost_usd
returns None for anything not listed, so callers can skip printing a cost
rather than report a fabricated number.
"""

from typing import Optional

_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-luna": (1.00, 6.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Return an estimated USD cost, or None if `model` has no known rate card.

    Accepts vendor-prefixed IDs too (e.g. OpenRouter's "openai/gpt-5.4") by
    falling back to the last "/"-separated segment.
    """
    rates = _PRICING_PER_MILLION.get(model) or _PRICING_PER_MILLION.get(model.rsplit("/", 1)[-1])
    if rates is None:
        return None
    in_rate, out_rate = rates
    return input_tokens / 1e6 * in_rate + output_tokens / 1e6 * out_rate
