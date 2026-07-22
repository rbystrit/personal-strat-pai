"""Transaction cost model — IBKR US equities, Tiered pricing (plan §10, §9.2).

All components are configurable; defaults approximate **IBKR Tiered** so
backtested costs are representative of live trading. Costs are conservative
— loosen them only with CEO approval for live trading (plan §12).

The model is **shared** between backtest and live: the live execution router
(P0-4) calls the same ``CostModel.estimate_fill`` to compute expected
per-share costs for pre-trade checks, and the same ``CommissionModel`` /
``FeeModel`` / ``SlippageModel`` classes to reconcile actual IBKR commissions
against expected. One code path (plan §10 acceptance criterion).

Cost components:

  * **Commission** (``CommissionModel``) — $0.0035/share, min $0.35/order,
    max 1% of notional (IBKR Tiered US equities).
  * **Regulatory / exchange fees** (``FeeModel``) — applied on **sells only**
    (matching US pass-throughs): SEC fee 0.0278 bps of sell principal, FINRA
    TAF $0.000166/share capped at $8.30/order.
  * **Slippage** (``SlippageModel``) — half-spread + square-root market
    impact. Half-spread is a configurable bps of price (default 5 bps).
    Sqrt-impact is ``sqrt_coeff * sqrt(shares / adv)`` in bps; requires
    average daily volume. Falls back to half-spread-only when ADV is missing.

``Fill.total_cost = commission + fees + slippage_cost``. The fill price
already incorporates slippage (buy fills above reference, sell fills below);
``slippage_cost`` is the dollar impact for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from typing import Protocol

__all__ = [
    "CommissionModel",
    "CostModel",
    "FeeModel",
    "Fill",
    "Side",
    "SlippageKind",
    "SlippageModel",
]


class Side(StrEnum):
    """Order side. ``BUY`` or ``SELL`` — long-only in v1 (plan §12)."""

    BUY = "BUY"
    SELL = "SELL"


class SlippageKind(StrEnum):
    """Slippage model variant (plan §10)."""

    HALF_SPREAD = "half_spread"  # fixed bps of price
    SQRT = "sqrt"  # half-spread + sqrt(shares/adv) market impact
    NONE = "none"  # fill at reference price (deterministic reconciliation)


@dataclass(frozen=True, slots=True)
class Fill:
    """A single executed fill with full cost attribution.

    ``fill_price`` is the price the order fills at (reference ± slippage).
    ``commission``, ``fees``, ``slippage_cost`` are the dollar cost components.
    ``total_cost`` is the sum. For a BUY, total cost increases the lot's cost
    basis; for a SELL, total cost reduces the net proceeds.

    ``ref_price`` is the pre-slippage reference price (bar close or next-bar
    open, depending on ``trade_on``); ``arrival_price`` is the price at the
    moment the signal was generated (for execution-quality attribution).
    """

    symbol: str
    side: Side
    qty: float
    ref_price: float
    fill_price: float
    arrival_price: float
    commission: float
    fees: float
    slippage_cost: float
    ts: str  # ISO-8601 timestamp of the fill bar

    @property
    def total_cost(self) -> float:
        """Total frictional cost = commission + fees + slippage_cost."""
        return self.commission + self.fees + self.slippage_cost

    @property
    def notional(self) -> float:
        """Notional value of the fill at the fill price."""
        return abs(self.qty) * self.fill_price

    @property
    def slippage_bps(self) -> float:
        """Realized slippage in bps vs the reference price."""
        if self.ref_price <= 0:
            return 0.0
        return abs(self.fill_price - self.ref_price) / self.ref_price * 10_000.0


class CommissionModel:
    """IBKR Tiered commission for US equities (plan §10).

    ``kind="per_share"`` (default): ``rate_per_share`` ($0.0035/share) with a
    ``min_per_order`` ($0.35) floor and a ``max_pct_of_notional`` (1%) cap.
    ``kind="bps"``: commission as basis points of notional.
    ``kind="zero"``: no commission (e.g. IBKR Lite on some plans).
    """

    def __init__(
        self,
        *,
        kind: str = "per_share",
        rate_per_share: float = 0.0035,
        min_per_order: float = 0.35,
        max_pct_of_notional: float = 0.01,
        bps_rate: float = 0.5,
    ) -> None:
        if kind not in ("per_share", "bps", "zero"):
            raise ValueError(f"unknown commission kind {kind!r}")
        self.kind = kind
        self.rate_per_share = rate_per_share
        self.min_per_order = min_per_order
        self.max_pct_of_notional = max_pct_of_notional
        self.bps_rate = bps_rate

    def compute(self, qty: float, price: float) -> float:
        """Dollar commission for ``qty`` shares at ``price``."""
        if self.kind == "zero":
            return 0.0
        notional = abs(qty) * price
        if self.kind == "bps":
            return notional * self.bps_rate / 10_000.0
        # per_share
        raw = abs(qty) * self.rate_per_share
        capped = min(raw, notional * self.max_pct_of_notional)
        return max(capped, self.min_per_order)


class FeeModel:
    """Regulatory / exchange fees — applied on **sells only** (plan §10).

    Matches US pass-throughs on IBKR Tiered:
      * SEC fee: ``sec_fee_sell_bps`` of sell principal (0.0278 bps = $27.80/$1M).
      * FINRA TAF: ``finra_taf_per_share_sell`` per share ($0.000166),
        capped at ``finra_taf_max`` ($8.30/order).
      * Exchange fee: ``exchange_fee_per_share`` per share (default 0 — net of
        liquidity rebates; raise for a more conservative estimate).
    """

    def __init__(
        self,
        *,
        sec_fee_sell_bps: float = 0.0278,
        finra_taf_per_share_sell: float = 0.000166,
        finra_taf_max: float = 8.30,
        exchange_fee_per_share: float = 0.0,
    ) -> None:
        self.sec_fee_sell_bps = sec_fee_sell_bps
        self.finra_taf_per_share_sell = finra_taf_per_share_sell
        self.finra_taf_max = finra_taf_max
        self.exchange_fee_per_share = exchange_fee_per_share

    def compute(self, qty: float, price: float, side: Side) -> float:
        """Dollar fees for ``qty`` shares at ``price`` on ``side``."""
        if side != Side.SELL:
            return 0.0
        notional = abs(qty) * price
        sec = notional * self.sec_fee_sell_bps / 10_000.0
        taf = min(abs(qty) * self.finra_taf_per_share_sell, self.finra_taf_max)
        exchange = abs(qty) * self.exchange_fee_per_share
        return sec + taf + exchange


class SlippageModel:
    """Slippage model — half-spread + optional sqrt market impact (plan §10).

    ``kind="half_spread"`` (default): a buy fills at ``ref * (1 + bps/1e4)``,
    a sell at ``ref * (1 - bps/1e4)``. Default 5 bps.

    ``kind="sqrt"``: half-spread + square-root market impact:
    ``impact_bps = sqrt_coeff * sqrt(shares / adv)``. Requires ``adv``
    (average daily volume) per fill. Falls back to half-spread-only when
    ADV is missing or zero.

    ``kind="none"``: fill at the reference price (used for the reconciliation
    gate so P&L is deterministic and hand-computable).

    ``jitter_bps`` is optional symmetric random jitter added to the impact
    (uses the engine's seeded RNG so results stay reproducible).
    """

    def __init__(
        self,
        *,
        kind: str = "half_spread",
        half_spread_bps: float = 5.0,
        sqrt_coeff: float = 1.0,
        jitter_bps: float = 0.0,
    ) -> None:
        if kind not in ("half_spread", "sqrt", "none"):
            raise ValueError(f"unknown slippage kind {kind!r}")
        self.kind = kind
        self.half_spread_bps = half_spread_bps
        self.sqrt_coeff = sqrt_coeff
        self.jitter_bps = jitter_bps

    def compute(
        self,
        qty: float,
        ref_price: float,
        side: Side,
        *,
        adv: float | None = None,
        jitter: float | None = None,
    ) -> tuple[float, float]:
        """Return ``(fill_price, slippage_cost)`` for the given order.

        ``jitter`` is a pre-drawn uniform [-1, 1] value from the engine's
        seeded RNG; when ``jitter_bps > 0`` and ``jitter`` is provided, it
        is scaled by ``jitter_bps`` and added to the impact. The caller
        owns the RNG so reproducibility is centralized.
        """
        if self.kind == "none":
            return ref_price, 0.0

        impact_bps = self.half_spread_bps
        if self.kind == "sqrt" and adv is not None and adv > 0:
            impact_bps += self.sqrt_coeff * sqrt(abs(qty) / adv)
        if self.jitter_bps > 0 and jitter is not None:
            impact_bps += jitter * self.jitter_bps

        frac = impact_bps / 10_000.0
        if side == Side.BUY:
            fill_price = ref_price * (1.0 + frac)
        else:
            fill_price = ref_price * (1.0 - frac)
        slippage_cost = abs(qty) * abs(fill_price - ref_price)
        return fill_price, slippage_cost


class CostModel(Protocol):
    """Unified cost model protocol — backtest and live share one implementation.

    The engine calls ``estimate_fill`` to simulate a fill; the live router
    (P0-4) calls the same method for pre-trade cost estimation. Both paths
    produce a ``Fill`` with full attribution.
    """

    def estimate_fill(
        self,
        symbol: str,
        side: Side,
        qty: float,
        ref_price: float,
        arrival_price: float,
        ts: str,
        *,
        adv: float | None = None,
        jitter: float | None = None,
    ) -> Fill: ...


class IbkrTieredCostModel:
    """Concrete ``CostModel`` — IBKR Tiered (plan §10, §9.2).

    Composes ``CommissionModel`` + ``FeeModel`` + ``SlippageModel``.
    Long-only in v1 (plan §12) — shorts are rejected by the risk guard,
    not by the cost model, but ``Side.SELL`` is supported for liquidation.
    """

    def __init__(
        self,
        commission: CommissionModel | None = None,
        fees: FeeModel | None = None,
        slippage: SlippageModel | None = None,
    ) -> None:
        self.commission = commission or CommissionModel()
        self.fees = fees or FeeModel()
        self.slippage = slippage or SlippageModel()

    def estimate_fill(
        self,
        symbol: str,
        side: Side,
        qty: float,
        ref_price: float,
        arrival_price: str | float,
        ts: str,
        *,
        adv: float | None = None,
        jitter: float | None = None,
    ) -> Fill:
        """Compute the full fill with cost attribution (plan §10).

        ``arrival_price`` may be a string (ISO timestamp — not supported in
        backtest, only live) or a float; in backtest it's the bar close at
        signal time for execution-quality attribution.
        """
        arrival = float(arrival_price)
        fill_price, slippage_cost = self.slippage.compute(
            qty, ref_price, side, adv=adv, jitter=jitter
        )
        commission = self.commission.compute(qty, fill_price)
        regulatory = self.fees.compute(qty, fill_price, side)
        return Fill(
            symbol=symbol,
            side=side,
            qty=qty,
            ref_price=ref_price,
            fill_price=fill_price,
            arrival_price=arrival,
            commission=commission,
            fees=regulatory,
            slippage_cost=slippage_cost,
            ts=ts,
        )
