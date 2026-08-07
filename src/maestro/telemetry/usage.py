"""Token & cost accounting for a swarm run.

Two ideas carry this module.

**Unknown is not zero.**  A backend that reports no usage at all (a local
llama.cpp build, a gateway that strips the field) must not be reported as having
consumed zero tokens — that is a measurement, and we did not make it.  Every
count here is therefore ``int | None``: ``None`` means *nobody told us*, ``0``
means *the provider said zero*.  The same rule applies to money: a model with no
entry in the price table has an **unknown** cost, never a free one.

**Roll up, never invent.**  :class:`UsageTotals` sums what was measured and keeps
a running count of the calls it could *not* account for, so a total always
arrives with the caveats attached (see :meth:`UsageTotals.render`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: List prices in US dollars per **million** tokens, as ``(input, output)``.
#:
#: Matching is a case-insensitive substring test against the model id, so dated
#: snapshots ("claude-opus-5-20260401") and vendor-prefixed ids
#: ("anthropic.claude-sonnet-5", as used on Bedrock) resolve to the same entry.
#: Longer keys are tried first, so a more specific family always wins.
#:
#: Only models whose price MAESTRO can state with confidence are listed.  An
#: unlisted model is *unpriced*, not free — register it yourself with
#: :func:`register_price` (or a ``prices:`` block in a swarm spec) if you want
#: costs for it.  Prices go stale; treat this table as a default, not gospel.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def register_price(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    """Teach MAESTRO what *model* costs, in dollars per million tokens.

    Use this for local models (where the honest answer is often ``0.0``), for
    gateways with their own rate card, or to correct a stale built-in price.
    """
    _PRICES[model.strip().lower()] = (float(input_per_mtok), float(output_per_mtok))


def price_for(model: str) -> tuple[float, float] | None:
    """Return ``(input, output)`` dollars per million tokens, or ``None``."""
    name = (model or "").strip().lower()
    if not name:
        return None
    for key in sorted(_PRICES, key=len, reverse=True):
        if key in name:
            return _PRICES[key]
    return None


def known_prices() -> dict[str, tuple[float, float]]:
    """A copy of the price table (for ``maestro prices`` and tests)."""
    return dict(_PRICES)


@dataclass(frozen=True)
class Usage:
    """What one backend call consumed.

    ``None`` token counts mean the provider reported nothing — the call still
    happened, we simply cannot say how large it was.
    """

    provider: str = ""
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def measured(self) -> bool:
        """True when the provider reported at least one token count."""
        return self.input_tokens is not None or self.output_tokens is not None

    @property
    def total_tokens(self) -> int | None:
        if not self.measured:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    def cost(self, prices: dict[str, tuple[float, float]] | None = None) -> float | None:
        """Dollar cost of this call, or ``None`` if it cannot be priced.

        Unknown either way — no price for the model, or no measured tokens —
        yields ``None`` rather than a confident-looking ``0.0``.
        """
        if not self.measured:
            return None
        rate = None
        if prices:
            name = (self.model or "").strip().lower()
            for key in sorted(prices, key=len, reverse=True):
                if key.strip().lower() in name:
                    rate = prices[key]
                    break
        if rate is None:
            rate = price_for(self.model)
        if rate is None:
            return None
        return (self.input_tokens or 0) / 1e6 * rate[0] + (self.output_tokens or 0) / 1e6 * rate[1]

    def as_dict(self) -> dict[str, object]:
        """A JSON-safe view.  Unknown counts stay ``None``, never ``0``."""
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class UsageTotals:
    """Aggregate usage, with an explicit account of what could not be measured."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    #: Calls whose provider reported no token counts at all.
    unmeasured_calls: int = 0
    #: Measured calls whose model has no known price.
    unpriced_calls: int = 0
    per_agent: dict[str, UsageTotals] = field(default_factory=dict)
    per_provider: dict[str, UsageTotals] = field(default_factory=dict)

    # -- properties -----------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_complete(self) -> bool:
        """True when every call contributed a measurement."""
        return self.calls > 0 and self.unmeasured_calls == 0

    @property
    def cost_complete(self) -> bool:
        """True when every call was both measured *and* priced."""
        return self.calls > 0 and self.unmeasured_calls == 0 and self.unpriced_calls == 0

    @property
    def cost(self) -> float | None:
        """Total cost, or ``None`` when no call could be priced at all.

        A partial figure (some calls priced, some not) is still returned — check
        :attr:`cost_complete` to know whether it is the whole bill.
        """
        priced = self.calls - self.unmeasured_calls - self.unpriced_calls
        return self.cost_usd if priced > 0 else None

    # -- construction ---------------------------------------------------------
    def add(self, usage: Usage, prices: dict[str, tuple[float, float]] | None = None) -> None:
        """Fold one call into this total (flat — no per-agent/provider split)."""
        self.calls += 1
        if not usage.measured:
            self.unmeasured_calls += 1
            return
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        cost = usage.cost(prices)
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += cost

    @classmethod
    def from_agent_usage(
        cls,
        by_agent: dict[str, list[Usage]],
        prices: dict[str, tuple[float, float]] | None = None,
    ) -> UsageTotals:
        """Roll ``{agent_name: [Usage, …]}`` up into a full breakdown."""
        totals = cls()
        for agent, uses in by_agent.items():
            for usage in uses:
                totals.add(usage, prices)
                totals.per_agent.setdefault(agent, cls()).add(usage, prices)
                key = usage.provider or "unknown"
                totals.per_provider.setdefault(key, cls()).add(usage, prices)
        return totals

    # -- presentation ---------------------------------------------------------
    def as_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost,
            "calls": self.calls,
            "unmeasured_calls": self.unmeasured_calls,
            "unpriced_calls": self.unpriced_calls,
            "tokens_complete": self.tokens_complete,
            "cost_complete": self.cost_complete,
            "per_agent": {k: v.as_dict() for k, v in self.per_agent.items()},
            "per_provider": {k: v.as_dict() for k, v in self.per_provider.items()},
        }

    def _one_line(self) -> str:
        tokens = f"{self.input_tokens} in / {self.output_tokens} out"
        if self.unmeasured_calls:
            tokens += f" (+{self.unmeasured_calls} call(s) unreported)"
        cost = self.cost
        if cost is None:
            money = "cost unknown"
        elif self.cost_complete:
            money = f"${cost:.6f}"
        else:
            money = f"${cost:.6f} (partial: {self.unpriced_calls} unpriced call(s))"
        return f"{tokens}  {money}"

    def render(self) -> str:
        """A human-readable usage report, caveats included."""
        lines = [f"calls:    {self.calls}", f"tokens:   {self._one_line()}"]
        if self.per_agent:
            lines.append("per agent:")
            for name in sorted(self.per_agent):
                lines.append(f"  {name:<14} {self.per_agent[name]._one_line()}")
        if self.per_provider:
            lines.append("per provider:")
            for name in sorted(self.per_provider):
                lines.append(f"  {name:<14} {self.per_provider[name]._one_line()}")
        if not self.tokens_complete:
            lines.append(
                "note: some calls reported no usage — totals are a floor, not a measurement."
            )
        if self.calls and not self.cost_complete:
            lines.append(
                "note: some models have no known price — register one with "
                "maestro.telemetry.usage.register_price() or a spec 'prices:' block."
            )
        return "\n".join(lines)


__all__ = [
    "Usage",
    "UsageTotals",
    "known_prices",
    "price_for",
    "register_price",
]
