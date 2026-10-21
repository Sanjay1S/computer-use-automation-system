"""Token usage and cost of a discovery run, per serving model.

Discovery is the only part of the system that calls a model, so this is the whole
model bill of a capability. The cost also feeds the run's budget. Prices are list
prices per million tokens; a model missing from the table is priced as the most
expensive one listed, so a budget never under-counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float
    """Five-minute cache writes: 1.25x input."""


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
    "claude-opus-5": Price(5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": Price(5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20, 2.50),
    "claude-fable-5-1": Price(10.00, 50.00, 0.25, 12.50),
}
_FALLBACK_PRICE = max(PRICES.values(), key=lambda p: p.output)


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    priced_as: str | None = None
    """Set when the model was not in the price table."""


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by_model: dict[str, ModelUsage] = Field(default_factory=dict)

    def add(self, model: str, usage: Any) -> None:
        """`usage` is a response's `usage` object (any object with the token fields)."""
        entry = self.by_model.setdefault(model, ModelUsage())
        price = PRICES.get(model)
        if price is None:
            entry.priced_as = "the highest listed price"
        tokens = {
            "input_tokens": _count(usage, "input_tokens"),
            "output_tokens": _count(usage, "output_tokens"),
            "cache_read_input_tokens": _count(usage, "cache_read_input_tokens"),
            "cache_creation_input_tokens": _count(usage, "cache_creation_input_tokens"),
        }
        entry.requests += 1
        for field, value in tokens.items():
            setattr(entry, field, getattr(entry, field) + value)
        entry.cost_usd += cost(price or _FALLBACK_PRICE, **tokens)

    @property
    def cost_usd(self) -> float:
        return sum(m.cost_usd for m in self.by_model.values())

    @property
    def models(self) -> list[str]:
        return sorted(self.by_model)


def cost(
    price: Price,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    return (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_input_tokens * price.cache_read
        + cache_creation_input_tokens * price.cache_write
    ) / 1_000_000


def _count(usage: Any, field: str) -> int:
    value = getattr(usage, field, None)
    return int(value) if isinstance(value, int) else 0
