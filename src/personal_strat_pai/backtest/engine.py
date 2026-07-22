"""Backtester engine — one code path with live (plan §10).

The engine is the event loop that drives the strategy on historical data.
It replaces only the data source (parquet vs IBKR) and the execution venue
(simulated fills vs IBKR orders). The strategy, risk guard, cost model, and
portfolio state are **shared** between backtest and live — no backtest-only
strategy logic (plan §10 acceptance criterion).

Event loop (monthly rebalance, brief §3):
  1. For each month-end trading day (from ``TradingCalendar.month_ends``):
     a. Inject monthly cash (brief §3: $10,000 fresh cash + stopped-out cash).
     b. ``BarDriver.bars_through(as_of)`` → eager frame for the signal.
     c. ``Strategy.target_weights(bars, as_of, portfolio)`` → target weights.
     d. ``RiskGuard.check(weights, drawdown, ...)`` → adjusted weights.
     e. Generate orders: target_dollar = weight × NAV; delta = target - current.
     f. ``BarDriver.fill_bar_for(as_of)`` → next bar's open for fills.
     g. ``CostModel.estimate_fill`` per order → ``Fill`` objects.
     h. ``Portfolio.apply_fill`` → update cash, positions, lots.
  2. Between rebalances, mark to market daily and record the equity curve.
  3. After all rebalances, generate the run manifest and reports.

``trade_on="next_open"`` (default): signals at bar *t*'s close, fills at
bar *t+1*'s open. No look-ahead bias. ``trade_on="same_close"``: decide and
fill at the same bar's close (for strategies that trade at the close).

Reproducibility: the ``run_manifest`` captures the dataset, config hash, git
sha, and parquet data hash so any run can be verified and reproduced (plan
§10 acceptance criterion).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import polars as pl

from personal_strat_pai.backtest.costs import (
    CostModel,
    IbkrTieredCostModel,
    Side,
)
from personal_strat_pai.backtest.data import BarDriver, TradingCalendar
from personal_strat_pai.backtest.portfolio import Portfolio, TradeRecord
from personal_strat_pai.backtest.risk import BetaProvider, RiskGuard, RiskResult
from personal_strat_pai.backtest.signals import Strategy

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for a backtest run (plan §10).

    All defaults are conservative and match IBKR Tiered pricing. The config
    is frozen so the run manifest can hash it for reproducibility.
    """

    initial_capital: float = 100_000.0
    monthly_cash_injection: float = 0.0  # brief §3: $10,000/mo (0 for reconciliation)
    trade_on: str = "next_open"  # "next_open" | "same_close"
    rebalance_freq: str = "monthly"  # "monthly" | "daily" | "weekly"
    seed: int = 42  # for slippage jitter reproducibility
    account: str = "backtest"
    # Cost model params (IBKR Tiered defaults).
    commission_per_share: float = 0.0035
    commission_min: float = 0.35
    commission_max_pct: float = 0.01
    sec_fee_sell_bps: float = 0.0278
    finra_taf_per_share: float = 0.000166
    finra_taf_max: float = 8.30
    slippage_kind: str = "none"  # "none" for reconciliation gate
    slippage_half_spread_bps: float = 5.0
    slippage_sqrt_coeff: float = 1.0
    slippage_jitter_bps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the run manifest (reproducibility)."""
        return {
            "initial_capital": self.initial_capital,
            "monthly_cash_injection": self.monthly_cash_injection,
            "trade_on": self.trade_on,
            "rebalance_freq": self.rebalance_freq,
            "seed": self.seed,
            "account": self.account,
            "commission_per_share": self.commission_per_share,
            "commission_min": self.commission_min,
            "commission_max_pct": self.commission_max_pct,
            "sec_fee_sell_bps": self.sec_fee_sell_bps,
            "finra_taf_per_share": self.finra_taf_per_share,
            "finra_taf_max": self.finra_taf_max,
            "slippage_kind": self.slippage_kind,
            "slippage_half_spread_bps": self.slippage_half_spread_bps,
            "slippage_sqrt_coeff": self.slippage_sqrt_coeff,
            "slippage_jitter_bps": self.slippage_jitter_bps,
        }

    def config_hash(self) -> str:
        """SHA-256 hash of the config — for the run manifest."""
        raw = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(slots=True)
class BacktestResult:
    """The output of a backtest run (plan §10).

    ``equity_curve``: polars DataFrame with columns [date, nav, cash, gross_exposure].
    ``trades``: list of ``TradeRecord`` — the trade-level report.
    ``stats``: dict of aggregate metrics (from ``metrics.py``).
    ``manifest``: the run manifest dict (from ``manifest.py``).
    ``risk_events``: list of all risk events logged during the run.
    """

    equity_curve: pl.DataFrame
    trades: list[TradeRecord]
    stats: dict[str, float]
    manifest: dict[str, Any]
    risk_events: list[dict[str, str]]
    config: BacktestConfig
    final_nav: float

    def summary(self) -> str:
        """One-line summary for quick inspection."""
        s = self.stats
        return (
            f"NAV: {self.final_nav:,.2f} | "
            f"Return: {s.get('total_return', 0):.2%} | "
            f"Sharpe: {s.get('sharpe', 0):.3f} | "
            f"MaxDD: {s.get('max_drawdown', 0):.2%} | "
            f"Trades: {len(self.trades)}"
        )

    def trades_df(self) -> pl.DataFrame:
        """Trades as a polars DataFrame — for parquet export."""
        if not self.trades:
            return pl.DataFrame(schema={
                "ts": pl.String,
                "symbol": pl.String,
                "side": pl.String,
                "qty": pl.Float64,
                "fill_price": pl.Float64,
                "ref_price": pl.Float64,
                "arrival_price": pl.Float64,
                "commission": pl.Float64,
                "fees": pl.Float64,
                "slippage_cost": pl.Float64,
                "total_cost": pl.Float64,
                "cash_delta": pl.Float64,
                "realized_pnl": pl.Float64,
                "nav_after": pl.Float64,
                "bucket_id": pl.Int64,
                "triplet_slot": pl.String,
            })
        return pl.DataFrame([{
            "ts": t.ts,
            "symbol": t.symbol,
            "side": t.side,
            "qty": t.qty,
            "fill_price": t.fill_price,
            "ref_price": t.ref_price,
            "arrival_price": t.arrival_price,
            "commission": t.commission,
            "fees": t.fees,
            "slippage_cost": t.slippage_cost,
            "total_cost": t.total_cost,
            "cash_delta": t.cash_delta,
            "realized_pnl": t.realized_pnl,
            "nav_after": t.nav_after,
            "bucket_id": t.bucket_id,
            "triplet_slot": t.triplet_slot,
        } for t in self.trades])


class Backtester:
    """The backtester engine — one code path with live (plan §10).

    Composes a ``BarDriver`` (data), ``Strategy`` (signal), ``CostModel``
    (costs), and ``RiskGuard`` (risk). The engine's event loop is the
    backtest-only piece; the strategy, risk guard, and cost model are
    shared with live. The live scheduler (P0-4) calls the same
    ``strategy.target_weights`` and ``risk_guard.check`` on live data.

    Usage:
        driver = ParquetBarDriver("data/local/bars/daily")
        strategy = MomentumStrategy(config)
        guard = RiskGuard(RiskLimits())
        bt = Backtester(BacktestConfig(initial_capital=100_000))
        result = bt.run(driver, strategy, guard)
        print(result.summary())
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def run(
        self,
        driver: BarDriver,
        strategy: Strategy,
        risk_guard: RiskGuard,
        *,
        cost_model: CostModel | None = None,
        beta_provider: BetaProvider | None = None,
        bucket_map: dict[str, int] | None = None,
    ) -> BacktestResult:
        """Run the backtest over the full date range in the bar driver.

        Returns a ``BacktestResult`` with the equity curve, trades, stats,
        and run manifest. The run is deterministic given the same config,
        data, and seed.
        """
        costs = cost_model or self._default_cost_model()

        portfolio = Portfolio(
            self.config.initial_capital,
            account=self.config.account,
        )

        # Build the trading calendar and rebalance dates.
        all_symbols = driver.all_symbols()
        trading_days = driver.trading_days()
        if not trading_days:
            raise ValueError("no trading days in the bar driver — empty data?")

        calendar = TradingCalendar(trading_days)
        rebalance_dates = calendar.month_ends()

        # Equity curve records: (date, nav, cash, gross_exposure).
        equity_records: list[dict[str, Any]] = []
        all_risk_events: list[dict[str, str]] = []

        # Mark to market for all trading days (daily equity curve).
        # We process rebalances on month-ends and mark-to-market every day.
        rebalance_set = set(rebalance_dates)

        for td in trading_days:
            td_dt = datetime(td.year, td.month, td.day, tzinfo=UTC)
            is_rebalance = td in rebalance_set

            if is_rebalance:
                # 1. Inject monthly cash (brief §3).
                if self.config.monthly_cash_injection > 0:
                    portfolio.inject_cash(self.config.monthly_cash_injection)

                # 2. Get bars through the decision date (eager, D14(b)).
                bars = driver.bars_through(all_symbols, td_dt)

                # 3. Mark to market at the close before computing the signal.
                close_prices = self._close_prices(bars, td)
                portfolio.mark_to_market(close_prices)

                # 4. Strategy: compute target weights.
                target = strategy.target_weights(bars, td_dt, portfolio)

                # 5. Risk guard: adjust weights.
                risk_result = risk_guard.check(
                    target,
                    drawdown=portfolio.drawdown,
                    beta_provider=beta_provider,
                    bucket_map=bucket_map,
                )
                for ev in risk_result.events:
                    all_risk_events.append({
                        "date": td.isoformat(),
                        "kind": ev.kind,
                        "message": ev.message,
                        "severity": ev.severity,
                    })

                # 6. Generate and execute orders.
                if not risk_result.halted or not risk_result.weights:
                    self._rebalance(
                        portfolio,
                        risk_result,
                        driver,
                        all_symbols,
                        td_dt,
                        td,
                        costs,
                        calendar,
                        bucket_map,
                    )
                # Kill switch flatten: sell everything.
                elif risk_result.weights == {} and risk_guard.kill_switch_active:
                    self._flatten(
                        portfolio, driver, all_symbols, td_dt, td, costs, calendar
                    )

            else:
                # Non-rebalance day: just mark to market.
                bars = driver.bars_through(all_symbols, td_dt)
                close_prices = self._close_prices(bars, td)
                portfolio.mark_to_market(close_prices)

            # Record equity curve point.
            equity_records.append({
                "date": td,
                "nav": portfolio.nav,
                "cash": portfolio.cash,
                "gross_exposure": portfolio.gross_exposure,
                "net_exposure": portfolio.net_exposure,
                "drawdown": portfolio.drawdown,
            })

        # Build the equity curve DataFrame.
        equity_curve = pl.DataFrame(equity_records).with_columns(
            pl.col("date").cast(pl.Date)
        )

        # Compute stats.
        from personal_strat_pai.backtest.metrics import compute_stats

        stats = compute_stats(equity_curve, portfolio)

        # Build the run manifest.
        from personal_strat_pai.backtest.manifest import build_manifest

        manifest = build_manifest(
            config=self.config,
            symbols=all_symbols,
            trading_days=trading_days,
            source=getattr(driver, "_source", None),
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=portfolio.trades,
            stats=stats,
            manifest=manifest,
            risk_events=all_risk_events,
            config=self.config,
            final_nav=portfolio.nav,
        )

    def _rebalance(
        self,
        portfolio: Portfolio,
        risk_result: RiskResult,
        driver: BarDriver,
        all_symbols: list[str],
        td_dt: datetime,
        td: date,
        costs: CostModel,
        calendar: TradingCalendar,
        bucket_map: dict[str, int] | None,
    ) -> None:
        """Generate and execute orders to move from current to target weights.

        Sells first (free up cash), then buys (deploy freed cash). This
        ordering avoids over-allocation when the portfolio is fully invested.
        """
        target_weights = risk_result.weights
        nav = portfolio.nav

        # Compute target dollar amounts.
        target_dollars = {sym: w * nav for sym, w in target_weights.items()}

        # Compute current dollar amounts (at last close).
        current_dollars = {
            sym: pos.market_value
            for sym, pos in portfolio.positions.items()
            if pos.qty != 0
        }

        # All symbols that have a target or a current position.
        all_relevant = set(target_dollars.keys()) | set(current_dollars.keys())

        # Compute order dollars: positive = buy, negative = sell.
        orders: list[tuple[str, float]] = []
        for sym in all_relevant:
            target = target_dollars.get(sym, 0.0)
            current = current_dollars.get(sym, 0.0)
            delta = target - current
            if abs(delta) > 0.01:  # skip negligible orders
                orders.append((sym, delta))

        if not orders:
            return

        # Get fill bars (next bar after the decision date).
        fill_bars = driver.fill_bar_for(all_symbols, td_dt)
        fill_prices = self._fill_prices(fill_bars, td)

        # Execute sells first (free up cash).
        sells = [(sym, d) for sym, d in orders if d < 0]
        buys = [(sym, d) for sym, d in orders if d > 0]

        for sym, dollar in sorted(sells, key=lambda x: x[1]):  # most negative first
            price = fill_prices.get(sym)
            if price is None or price <= 0:
                continue
            pos = portfolio.position(sym)
            if pos.qty <= 0:
                continue
            # If target weight is 0 (full exit), sell the entire position.
            # Otherwise, sell the delta in shares (current - target) at fill price.
            target_weight = target_weights.get(sym, 0.0)
            if target_weight <= 1e-9:
                sell_qty = pos.qty
            else:
                # Dollar is the target - current delta (negative for sells).
                # Sell qty = abs(dollar) / fill_price, capped at current qty.
                sell_qty = min(abs(dollar) / price, pos.qty)
            if sell_qty <= 0:
                continue
            fill = costs.estimate_fill(
                symbol=sym,
                side=Side.SELL,
                qty=sell_qty,
                ref_price=price,
                arrival_price=pos.last_price,
                ts=fill_bars.filter(pl.col("symbol") == sym)["ts"][0].isoformat()
                if not fill_bars.filter(pl.col("symbol") == sym).is_empty()
                else td_dt.isoformat(),
                adv=None,
                jitter=self._rng.uniform(-1, 1) if self.config.slippage_jitter_bps > 0 else None,
            )
            bucket_id = bucket_map.get(sym) if bucket_map else None
            portfolio.apply_fill(fill, bucket_id=bucket_id)

        # Execute buys.
        for sym, dollar in sorted(buys, key=lambda x: -x[1]):  # largest buy first
            price = fill_prices.get(sym)
            if price is None or price <= 0:
                continue
            # Compute buy qty: deploy available cash up to the target dollar.
            # When target weight is ~100% (full allocation), use all available
            # cash (matching the independent computation's logic). Otherwise,
            # cap at the target dollar amount.
            target_weight = target_weights.get(sym, 0.0)
            if target_weight >= 0.99:
                budget = portfolio.cash
            else:
                budget = min(dollar, portfolio.cash)
            if budget <= 0:
                continue
            buy_qty = budget / (price + self.config.commission_per_share)
            for _ in range(3):  # converge
                est_comm = max(
                    buy_qty * self.config.commission_per_share,
                    self.config.commission_min,
                )
                new_qty = (budget - est_comm) / price
                if abs(new_qty - buy_qty) < 1e-9:
                    break
                buy_qty = new_qty
            if buy_qty <= 0:
                continue
            fill = costs.estimate_fill(
                symbol=sym,
                side=Side.BUY,
                qty=buy_qty,
                ref_price=price,
                arrival_price=portfolio.position(sym).last_price,
                ts=fill_bars.filter(pl.col("symbol") == sym)["ts"][0].isoformat()
                if not fill_bars.filter(pl.col("symbol") == sym).is_empty()
                else td_dt.isoformat(),
                adv=None,
                jitter=self._rng.uniform(-1, 1) if self.config.slippage_jitter_bps > 0 else None,
            )
            bucket_id = bucket_map.get(sym) if bucket_map else None
            portfolio.apply_fill(fill, bucket_id=bucket_id)

    def _flatten(
        self,
        portfolio: Portfolio,
        driver: BarDriver,
        all_symbols: list[str],
        td_dt: datetime,
        td: date,
        costs: CostModel,
        calendar: TradingCalendar,
    ) -> None:
        """Sell all positions (kill-switch flatten mode)."""
        held = [sym for sym, pos in portfolio.positions.items() if pos.qty > 0]
        if not held:
            return
        fill_bars = driver.fill_bar_for(held, td_dt)
        fill_prices = self._fill_prices(fill_bars, td)
        for sym in held:
            price = fill_prices.get(sym)
            if price is None or price <= 0:
                continue
            pos = portfolio.position(sym)
            if pos.qty <= 0:
                continue
            fill = costs.estimate_fill(
                symbol=sym,
                side=Side.SELL,
                qty=pos.qty,
                ref_price=price,
                arrival_price=pos.last_price,
                ts=fill_bars.filter(pl.col("symbol") == sym)["ts"][0].isoformat()
                if not fill_bars.filter(pl.col("symbol") == sym).is_empty()
                else td_dt.isoformat(),
            )
            portfolio.apply_fill(fill)

    @staticmethod
    def _close_prices(bars: pl.DataFrame, td: date) -> dict[str, float]:
        """Extract close prices for the given trading day from bars."""
        day_bars = bars.filter(pl.col("ts").dt.date() == td)
        if day_bars.is_empty():
            # Use the last available bar per symbol.
            return {
                row["symbol"]: float(row["close"])
                for row in bars.sort("ts").group_by("symbol").last().iter_rows(named=True)
            }
        return {
            row["symbol"]: float(row["close"])
            for row in day_bars.iter_rows(named=True)
        }

    @staticmethod
    def _fill_prices(fill_bars: pl.DataFrame, td: date) -> dict[str, float]:
        """Extract fill prices (open of the next bar) from fill_bars."""
        if fill_bars.is_empty():
            return {}
        return {
            row["symbol"]: float(row["open"])
            for row in fill_bars.iter_rows(named=True)
        }

    def _default_cost_model(self) -> IbkrTieredCostModel:
        """Build the default IBKR Tiered cost model from the config."""
        from personal_strat_pai.backtest.costs import (
            CommissionModel,
            FeeModel,
            SlippageModel,
        )

        commission = CommissionModel(
            rate_per_share=self.config.commission_per_share,
            min_per_order=self.config.commission_min,
            max_pct_of_notional=self.config.commission_max_pct,
        )
        fees = FeeModel(
            sec_fee_sell_bps=self.config.sec_fee_sell_bps,
            finra_taf_per_share_sell=self.config.finra_taf_per_share,
            finra_taf_max=self.config.finra_taf_max,
        )
        slippage = SlippageModel(
            kind=self.config.slippage_kind,
            half_spread_bps=self.config.slippage_half_spread_bps,
            sqrt_coeff=self.config.slippage_sqrt_coeff,
            jitter_bps=self.config.slippage_jitter_bps,
        )
        return IbkrTieredCostModel(commission=commission, fees=fees, slippage=slippage)
