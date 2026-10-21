"""Token usage and cost: the discovery budget is only as good as this arithmetic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from understudy.evidence.usage import PRICES, Usage, cost


def test_cost_counts_every_token_class_at_its_own_price() -> None:
    price = PRICES["claude-opus-5-5"]
    assert cost(
        price,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    ) == pytest.approx(4.00 + 20.00 + 0.20 + 5.00)


def test_usage_adds_up_per_serving_model() -> None:
    usage = Usage()
    usage.add("claude-opus-5-5", SimpleNamespace(input_tokens=10_000, output_tokens=1_000))
    usage.add(
        "claude-opus-5-5",
        SimpleNamespace(input_tokens=500, output_tokens=200, cache_read_input_tokens=10_000),
    )
    usage.add("claude-opus-5", SimpleNamespace(input_tokens=1_000, output_tokens=0))

    opus = usage.by_model["claude-opus-5-5"]
    assert (opus.requests, opus.input_tokens, opus.cache_read_input_tokens) == (2, 10_500, 10_000)
    assert usage.models == ["claude-opus-5", "claude-opus-5-5"]
    expected = (10_500 * 4 + 1_200 * 20 + 10_000 * 0.20) / 1e6 + 1_000 * 5 / 1e6
    assert usage.cost_usd == pytest.approx(expected)


def test_an_unknown_model_is_priced_high_never_free() -> None:
    usage = Usage()
    usage.add("claude-future-9", SimpleNamespace(input_tokens=1_000_000, output_tokens=0))

    entry = usage.by_model["claude-future-9"]
    assert entry.priced_as is not None
    assert entry.cost_usd == pytest.approx(max(p.input for p in PRICES.values()))


def test_missing_and_null_token_fields_count_as_zero() -> None:
    usage = Usage()
    usage.add("claude-opus-5-5", SimpleNamespace(input_tokens=100, output_tokens=None))

    assert usage.by_model["claude-opus-5-5"].output_tokens == 0
