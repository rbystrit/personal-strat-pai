"""Portfolio state — positions, cash, fills, mark-to-market (plan §10).

The portfolio tracks cash and per-ticker positions, applies fills from the
cost model, and marks to market at each bar close. NAV = cash + Σ(position
market value). Weights = position_value / NAV.

HIFO integration (plan §8, §10): when a ``LedgerRepo`` is provided, sells
flow through HIFO tax-lot selection — the same pure functions (``hifo_select``,
``realize_lot``, ``close_lot``) that live trading uses. In backtest, the
ledger is backed by ``InMemoryNoSqlStore``; in live, by ``OciNoSqlStore``.
One code path — only the store backend changes.

For the reconciliation gate, a ``simple_portfolio`` path (no ledger) tracks
average-cost lots directly — fully auditable for hand-computed scenarios.
The HIFO path and the simple path produce the same P&L when there's only one
lot per position (which is the case in the reconciliation scenario).

Long-only, margin=cash (plan §12): buying requires sufficient cash; shorts
are rejected by the risk guard, not the portfolio. Fractional shares are
supported in backtest; the live IBKR path must enforce integer rounding
(P0-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from personal_strat_pai.backtest.costs import Fill, Side

__all__ = [
    "Lot",
    "Portfolio",
    "Position",
    "TradeRecord",
]


@dataclass(slots=True)
class Lot:
    """A single tax lot within a position (simple average-cost tracking).

    For HIFO-backed portfolios, the canonical lot record lives in the
    ``LedgerRepo``; this is the in-memory mirror for mark-to-market.
    """

    qty: float
    cost_basis_per_share: float
    acquired_at: datetime


@dataclass(slots=True)
class Position:
    """Per-ticker position state.

    ``qty`` is the net share count (positive = long). ``lots`` is the list
    of open lots (for average-cost tracking). ``realized_pnl`` is the
    cumulative realized P&L from sells (excluding costs — costs are tracked
    separately in ``TradeRecord``).
    """

    symbol: str
    qty: float = 0.0
    lots: list[Lot] = field(default_factory=list)
    realized_pnl: float = 0.0
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Mark-to-market value at ``last_price``."""
        return self.qty * self.last_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of open lots."""
        return sum(lot.qty * lot.cost_basis_per_share for lot in self.lots)

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L at ``last_price``."""
        return self.market_value - self.cost_basis

    @property
    def avg_cost(self) -> float:
        """Volume-weighted average cost per share of open lots."""
        total_qty = sum(lot.qty for lot in self.lots)
        if total_qty <= 0:
            return 0.0
        return self.cost_basis / total_qty


@dataclass(slots=True)
class TradeRecord:
    """A completed fill with full attribution for trade-level reporting.

    Captures the fill details plus the portfolio state impact (cash delta,
    realized P&L, position change). Written to parquet as the trade-level
    report (plan §10 acceptance criterion).
    """

    ts: str
    symbol: str
    side: str
    qty: float
    fill_price: float
    ref_price: float
    arrival_price: float
    commission: float
    fees: float
    slippage_cost: float
    total_cost: float
    cash_delta: float
    realized_pnl: float
    nav_after: float
    bucket_id: int | None = None
    triplet_slot: str | None = None


class Portfolio:
    """Portfolio state — cash + positions, fills, mark-to-market (plan §10).

    ``apply_fill`` updates cash and positions from a ``Fill`` produced by the
    cost model. For buys: cash decreases by ``qty * fill_price + commission +
    fees``; a new lot is opened. For sells: cash increases by ``qty * fill_price
    - commission - fees``; lots are closed FIFO (or HIFO when a ledger is
    provided), and realized P&L is recorded.

    ``mark_to_market`` updates each position's ``last_price`` from the bar
    frame. ``nav`` returns cash + Σ(market_value). ``weights`` returns the
    target-weight vector.
    """

    def __init__(
        self,
        initial_capital: float,
        *,
        account: str = "backtest",
    ) -> None:
        self.cash = float(initial_capital)
        self.initial_capital = float(initial_capital)
        self.account = account
        self.positions: dict[str, Position] = {}
        self.trades: list[TradeRecord] = []
        self._peak_nav: float = float(initial_capital)

    @property
    def nav(self) -> float:
        """Net asset value = cash + Σ(position market value)."""
        pos_value = sum(p.market_value for p in self.positions.values() if p.qty != 0)
        return self.cash + pos_value

    @property
    def peak_nav(self) -> float:
        """Peak NAV — for drawdown calculation."""
        return self._peak_nav

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak NAV (positive = in drawdown)."""
        if self._peak_nav <= 0:
            return 0.0
        return max(0.0, (self._peak_nav - self.nav) / self._peak_nav)

    @property
    def gross_exposure(self) -> float:
        """Gross exposure = Σ|position market value| / NAV."""
        n = self.nav
        if n <= 0:
            return 0.0
        return sum(abs(p.market_value) for p in self.positions.values() if p.qty != 0) / n

    @property
    def net_exposure(self) -> float:
        """Net exposure = Σ(position market value) / NAV."""
        n = self.nav
        if n <= 0:
            return 0.0
        return sum(p.market_value for p in self.positions.values() if p.qty != 0) / n

    def weights(self) -> dict[str, float]:
        """Current portfolio weights = position_value / NAV."""
        n = self.nav
        if n <= 0:
            return {}
        return {
            sym: p.market_value / n
            for sym, p in self.positions.items()
            if p.qty != 0
        }

    def position(self, symbol: str) -> Position:
        """Get or create a position for ``symbol``."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update each position's last price and recompute peak NAV."""
        for sym, price in prices.items():
            pos = self.position(sym)
            pos.last_price = price
        current = self.nav
        self._peak_nav = max(self._peak_nav, current)

    def apply_fill(
        self,
        fill: Fill,
        *,
        bucket_id: int | None = None,
        triplet_slot: str | None = None,
    ) -> TradeRecord:
        """Apply a fill to the portfolio — update cash, position, lots.

        For a BUY: cash -= qty * fill_price + commission + fees; open a new lot.
        For a SELL: cash += qty * fill_price - commission - fees; close lots
        FIFO, record realized P&L.

        Returns a ``TradeRecord`` for the trade-level report.
        """
        pos = self.position(fill.symbol)
        cash_before = self.cash

        if fill.side == Side.BUY:
            self.cash -= fill.qty * fill.fill_price + fill.commission + fill.fees
            pos.lots.append(
                Lot(
                    qty=fill.qty,
                    cost_basis_per_share=fill.fill_price,
                    acquired_at=datetime.fromisoformat(fill.ts),
                )
            )
            pos.qty += fill.qty
            realized = 0.0
        else:  # SELL
            self.cash += fill.qty * fill.fill_price - fill.commission - fill.fees
            realized = self._close_lots(pos, fill.qty, fill.fill_price, fill.ts)
            pos.realized_pnl += realized
            pos.qty -= fill.qty

        nav_after = self.nav
        record = TradeRecord(
            ts=fill.ts,
            symbol=fill.symbol,
            side=fill.side.value,
            qty=fill.qty,
            fill_price=fill.fill_price,
            ref_price=fill.ref_price,
            arrival_price=fill.arrival_price,
            commission=fill.commission,
            fees=fill.fees,
            slippage_cost=fill.slippage_cost,
            total_cost=fill.total_cost,
            cash_delta=self.cash - cash_before,
            realized_pnl=realized,
            nav_after=nav_after,
            bucket_id=bucket_id,
            triplet_slot=triplet_slot,
        )
        self.trades.append(record)
        return record

    @staticmethod
    def _close_lots(
        pos: Position, sell_qty: float, fill_price: float, ts: str
    ) -> float:
        """Close lots FIFO, return realized P&L (proceeds - cost).

        Simple FIFO for the backtest's average-cost path. When a HIFO
        ``LedgerRepo`` is provided (live/paper), the engine routes sells
        through ``LedgerRepo.sell`` instead, which uses HIFO selection with
        NoSQL conditional-write atomicity. This FIFO path is the auditable
        fallback for the reconciliation gate.
        """
        remaining = sell_qty
        realized = 0.0
        while remaining > 1e-9 and pos.lots:
            lot = pos.lots[0]
            close_qty = min(lot.qty, remaining)
            proceeds = close_qty * fill_price
            cost = close_qty * lot.cost_basis_per_share
            realized += proceeds - cost
            remaining -= close_qty
            lot.qty -= close_qty
            if lot.qty <= 1e-9:
                pos.lots.pop(0)
        return realized

    def reset_peak(self) -> None:
        """Reset peak NAV to current — used after capital injections."""
        self._peak_nav = self.nav

    def inject_cash(self, amount: float) -> None:
        """Add cash to the portfolio (monthly injection, brief §3)."""
        self.cash += amount
        self.reset_peak()
