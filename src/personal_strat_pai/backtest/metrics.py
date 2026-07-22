"""Performance metrics + tax-drag attribution (plan §10).

Computes trade-level and aggregate performance reports from the backtest
result:

  * **Equity-curve metrics:** total return, CAGR, volatility, Sharpe, Sortino,
    max drawdown, Calmar.
  * **Exposure metrics:** average gross/net exposure, turnover.
  * **Cost attribution:** total commission, fees, slippage; per-ticker breakdown.
  * **Tax-drag attribution:** realized ST/LT gains, harvested losses, and the
    tax drag = (ST gains × ST rate + LT gains × LT rate) / NAV. Uses the
    HIFO ledger's ST/LT classification from ``RealizedLot`` records.

The metrics module operates on the ``BacktestResult``'s equity curve and
trade list. It returns a dict suitable for JSON export (plan §10 acceptance
criterion: JSON metrics).
"""

from __future__ import annotations

import math

import polars as pl

from personal_strat_pai.backtest.portfolio import Portfolio

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "compute_cost_attribution",
    "compute_stats",
    "compute_tax_drag",
]

TRADING_DAYS_PER_YEAR: int = 252

# Brief §1: 38.8% ST (35% ordinary + 20% LTCG base + 3.8% NIIT),
# 23.8% LT (20% + 3.8% NIIT).
ST_TAX_RATE: float = 0.388
LT_TAX_RATE: float = 0.238


def compute_stats(
    equity_curve: pl.DataFrame,
    portfolio: Portfolio,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Compute aggregate performance stats from the equity curve.

    Returns a dict with: total_return, cagr, volatility, sharpe, sortino,
    max_drawdown, calmar, win_rate, n_periods, start_equity, end_equity,
    avg_gross_exposure, avg_net_exposure, turnover, total_commission,
    total_fees, total_slippage, total_costs.
    """
    if equity_curve.is_empty():
        return {"total_return": 0.0, "n_periods": 0}

    nav = equity_curve["nav"].cast(pl.Float64)
    n = nav.len()
    start_equity = float(nav[0])
    end_equity = float(nav[-1])

    # Daily returns.
    returns = nav.pct_change().drop_nulls()
    if returns.len() < 2:
        return {
            "total_return": (end_equity / start_equity - 1.0) if start_equity > 0 else 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "n_periods": float(n),
            "start_equity": start_equity,
            "end_equity": end_equity,
            "avg_gross_exposure": float(equity_curve["gross_exposure"].mean() or 0.0),  # type: ignore[arg-type]
            "avg_net_exposure": float(equity_curve["net_exposure"].mean() or 0.0),  # type: ignore[arg-type]
            "turnover": 0.0,
            "total_commission": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "total_costs": 0.0,
        }

    total_return = end_equity / start_equity - 1.0 if start_equity > 0 else 0.0
    years = n / trading_days
    cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0 if years > 0 and start_equity > 0 else 0.0

    # Annualized volatility.
    vol = float(returns.std(ddof=1)) * math.sqrt(trading_days) if returns.std(ddof=1) is not None else 0.0  # type: ignore[arg-type]
    if isinstance(vol, float) and math.isnan(vol):
        vol = 0.0

    # Sharpe ratio (annualized).
    mean_ret = float(returns.mean())  # type: ignore[arg-type]
    if isinstance(mean_ret, float) and math.isnan(mean_ret):
        mean_ret = 0.0
    daily_rf = risk_free / trading_days
    excess = returns - daily_rf  # scalar broadcast on Series
    excess_std_val = excess.std(ddof=1)
    excess_std = float(excess_std_val) if excess_std_val is not None else 0.0  # type: ignore[arg-type]
    if math.isnan(excess_std):
        excess_std = 0.0
    excess_mean = float(excess.mean())  # type: ignore[arg-type]
    if math.isnan(excess_mean):
        excess_mean = 0.0
    sharpe = (
        excess_mean / excess_std * math.sqrt(trading_days)
        if excess_std > 1e-12
        else 0.0
    )
    if math.isnan(sharpe):
        sharpe = 0.0

    # Sortino ratio (only penalizes downside deviation).
    downside = excess.filter(excess < 0)
    downside_std_val = downside.std(ddof=1) if downside.len() > 0 else None
    downside_std = float(downside_std_val) if downside_std_val is not None else 0.0  # type: ignore[arg-type]
    if math.isnan(downside_std):
        downside_std = 0.0
    sortino = (
        excess_mean / downside_std * math.sqrt(trading_days)
        if downside_std > 1e-12
        else 0.0
    )
    if math.isnan(sortino):
        sortino = 0.0

    # Max drawdown.
    cummax = nav.cum_max()
    drawdown = (cummax - nav) / cummax
    max_dd = float(drawdown.max()) if drawdown.max() is not None else 0.0  # type: ignore[arg-type]
    if isinstance(max_dd, float) and math.isnan(max_dd):
        max_dd = 0.0

    # Calmar.
    calmar = cagr / max_dd if max_dd > 1e-9 else 0.0

    # Win rate (fraction of positive-return days).
    win_rate = float(returns.filter(returns > 0).len()) / float(returns.len()) if returns.len() > 0 else 0.0

    # Turnover (sum of |buy_value + sell_value| / avg NAV per period).
    trades = portfolio.trades
    total_trade_value = sum(abs(t.qty * t.fill_price) for t in trades)
    avg_nav = float(nav.mean()) if nav.mean() is not None else end_equity  # type: ignore[arg-type]
    turnover = total_trade_value / avg_nav / max(years, 1e-9) if avg_nav > 0 else 0.0

    # Cost attribution.
    total_commission = sum(t.commission for t in trades)
    total_fees = sum(t.fees for t in trades)
    total_slippage = sum(t.slippage_cost for t in trades)
    total_costs = total_commission + total_fees + total_slippage

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_periods": float(n),
        "start_equity": start_equity,
        "end_equity": end_equity,
        "avg_gross_exposure": float(equity_curve["gross_exposure"].mean() or 0.0),  # type: ignore[arg-type]
        "avg_net_exposure": float(equity_curve["net_exposure"].mean() or 0.0),  # type: ignore[arg-type]
        "turnover": turnover,
        "total_commission": total_commission,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_costs": total_costs,
    }


def compute_tax_drag(
    portfolio: Portfolio,
    *,
    st_tax_rate: float = ST_TAX_RATE,
    lt_tax_rate: float = LT_TAX_RATE,
) -> dict[str, float]:
    """Compute tax-drag attribution from realized P&L (brief §1, plan §10).

    Uses the portfolio's trade records to compute realized ST/LT gains and
    the tax drag = (ST gains × ST rate + LT gains × LT rate) / NAV.

    In the full HIFO integration, the ST/LT split comes from the ledger's
    ``RealizedLot`` records. In the simple portfolio path, all realized gains
    are classified as ST (the reconciliation scenario uses short holding
    periods). The full HIFO-backed tax-drag attribution is wired when the
    ``LedgerRepo`` is integrated.
    """
    trades = portfolio.trades
    realized_st = sum(t.realized_pnl for t in trades if t.realized_pnl != 0)
    # In the simple path, we can't distinguish ST/LT without lot holding periods.
    # The HIFO ledger provides this; here we report total realized + an estimate.
    total_realized = realized_st
    tax_owed = max(total_realized, 0) * st_tax_rate  # conservative: all ST
    nav = portfolio.nav
    tax_drag = tax_owed / nav if nav > 0 else 0.0

    return {
        "realized_st_gains": realized_st,
        "realized_lt_gains": 0.0,  # populated by HIFO integration
        "total_realized_gains": total_realized,
        "harvested_losses": sum(abs(t.realized_pnl) for t in trades if t.realized_pnl < 0),
        "tax_owed": tax_owed,
        "tax_drag": tax_drag,
        "st_tax_rate": st_tax_rate,
        "lt_tax_rate": lt_tax_rate,
    }


def compute_cost_attribution(portfolio: Portfolio) -> dict[str, dict[str, float]]:
    """Per-ticker cost attribution (plan §10 trade-level report).

    Returns ``{ticker: {commission, fees, slippage, total_cost, n_trades,
    buy_value, sell_value}}``.
    """
    by_ticker: dict[str, dict[str, float]] = {}
    for t in portfolio.trades:
        sym = t.symbol
        if sym not in by_ticker:
            by_ticker[sym] = {
                "commission": 0.0,
                "fees": 0.0,
                "slippage": 0.0,
                "total_cost": 0.0,
                "n_trades": 0.0,
                "buy_value": 0.0,
                "sell_value": 0.0,
            }
        entry = by_ticker[sym]
        entry["commission"] += t.commission
        entry["fees"] += t.fees
        entry["slippage"] += t.slippage_cost
        entry["total_cost"] += t.total_cost
        entry["n_trades"] += 1
        if t.side == "BUY":
            entry["buy_value"] += t.qty * t.fill_price
        else:
            entry["sell_value"] += t.qty * t.fill_price
    return by_ticker
