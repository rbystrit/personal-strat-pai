"""IV proxy — HV for backtesting, IBKR for live/paper (CEO directive 2026-07-18).

Supersedes design decision D2 (self-built IV from an EOD options chain snapshot,
no OPRA). CEO feedback 2026-07-18 on RBY-4:

  * **Backtesting** — compute **HV (historical volatility)** as the IV proxy.
    OPRA is too expensive; the EOD-chain self-built IV path is dropped. HV is
    realized volatility from the daily bars we already ingest, so it costs zero
    extra data spend.
  * **Live/paper** — obtain IV / options data via **IBKR** (we already connect
    to IBKR for execution). The ``IbkrIvProvider`` is the parameterized swap
    target; it raises ``NotImplementedError`` until the IBKR market-data wiring
    lands (P0-3), so the composition root is the only place that changes.

The ``IvProvider`` protocol (unchanged shape) is the seam the sieve codes
against, so swapping HV -> IBKR is a one-line wiring change. ``IvSnapshot``
keeps its 30d/90d term-structure fields so downstream code (smoke detector,
stops) does not need to branch on the source:

  * ``HvIvProvider`` fills ``iv_30d``/``iv_90d`` with 30-/90-day realized vol
    (annualized), ``term_slope`` and ``backwardation`` from those.
  * Put-skew fields (``put_skew_30d``, ``put_skew_percentile_12m``,
    ``extreme_put_skew``) are NOT computable from HV — they require option
    prices. They are set to ``0.0``/``None``/``False`` here and are populated
    by ``IbkrIvProvider`` in live mode. ``config/strategy.yaml`` keeps
    ``put_skew_extreme_percentile`` (D7) for when the live IBKR IV provider is
    wired; it is unused by the HV backtest path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import sqrt
from typing import Protocol

import polars as pl

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "BarLoader",
    "HvIvProvider",
    "IbkrIvProvider",
    "IvProvider",
    "IvSnapshot",
    "compute_hv",
    "compute_hv_snapshot",
]

# 252 is the conventional US-equity trading-day count for annualizing realized vol.
TRADING_DAYS_PER_YEAR: int = 252


# --- IV snapshot (shape unchanged from D2; put-skew fields deferred to IBKR) --- #
@dataclass(frozen=True, slots=True)
class IvSnapshot:
    """Per-symbol daily IV snapshot (plan §6.2).

    For the HV backtest proxy: ``iv_30d``/``iv_90d`` are 30-/90-day realized
    vol (annualized decimal); put-skew fields are zeroed (not computable from
    bars). For the live IBKR provider: all fields populated from IBKR option
    data. Downstream code reads the snapshot uniformly.
    """

    symbol: str
    as_of: date
    iv_30d: float  # 30-day IV (HV proxy in backtest; ATM IV in live)
    iv_90d: float  # 90-day IV (HV proxy in backtest; ATM IV in live)
    term_slope: float  # iv_30d - iv_90d  (>0 => backwardation, plan §6.2)
    put_skew_30d: float  # 30D OTM-put IV - 30D ATM IV (0.0 under HV proxy; live via IBKR)
    put_skew_percentile_12m: (
        float | None
    )  # pct vs trailing 12m (None under HV proxy; live via IBKR)
    backwardation: bool  # iv_30d > iv_90d  (plan §6.2 smoke-detector trigger)
    extreme_put_skew: bool  # False under HV proxy; live via IBKR


# --- HV (realized volatility) --- #
def compute_hv(
    closes: pl.Series | list[float] | pl.DataFrame,
    window: int,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Realized volatility over a trailing ``window`` of daily closes (annualized).

    HV = sample-std of the trailing ``window`` log returns, annualized by
    ``sqrt(trading_days)``. Requires at least ``window + 1`` closes to yield
    ``window`` returns; returns ``None`` if there is not enough history (the
    provider surfaces this as a snapshot-time error, not here).

    Accepts a polars Series, a list of floats, or a 1-column DataFrame of
    closes sorted ascending by date (caller responsibility).
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if isinstance(closes, pl.DataFrame):
        if closes.width != 1:
            raise ValueError("compute_hv: pass a 1-column DataFrame, a Series, or a list")
        closes = closes.to_series()
    series = pl.Series("c", closes) if not isinstance(closes, pl.Series) else closes
    series = series.cast(pl.Float64).drop_nulls()
    if series.len() < window + 1:
        return None
    # Trailing `window` log returns need the last `window + 1` closes.
    tail = series.tail(window + 1)
    log_rets = (tail / tail.shift(1)).log()
    log_rets = log_rets.drop_nulls()
    if log_rets.len() < 2:
        return None
    std_val = log_rets.std(ddof=1)
    if std_val is None:
        return None
    # polars' std() stubs widen the scalar to `float | timedelta | None`; a
    # timedelta is impossible for log-returns but the guard satisfies the type
    # checker and defends against an unexpected dtype.
    if isinstance(std_val, timedelta):
        return None
    return float(std_val) * sqrt(trading_days)


def compute_hv_snapshot(
    bars: pl.DataFrame,
    symbol: str,
    as_of: date,
    *,
    hv_short: int = 30,
    hv_long: int = 90,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> IvSnapshot:
    """Build an ``IvSnapshot`` from a bar frame using HV as the IV proxy.

    ``bars`` must conform to ``BAR_SCHEMA`` (symbol, ts, open, high, low, close,
    volume) and span at least ``hv_long + 1`` daily closes ending at/ before
    ``as_of``. Short/long HV fill ``iv_30d``/``iv_90d``; put-skew fields are
    zeroed (not computable from bars — deferred to the live IBKR provider).
    Raises ``ValueError`` if there is not enough history for either window.
    """
    if hv_short <= 0 or hv_long <= 0 or hv_short >= hv_long:
        raise ValueError(f"require 0 < hv_short < hv_long, got {hv_short}, {hv_long}")
    if bars.is_empty():
        raise ValueError(f"no bars for {symbol!r} to compute HV as of {as_of}")
    sub = bars.filter(pl.col("symbol") == symbol).filter(pl.col("ts").dt.date() <= as_of).sort("ts")
    if sub.is_empty():
        raise ValueError(f"no bars for {symbol!r} on/before {as_of}")
    closes = sub["close"].cast(pl.Float64)
    hv_30d = compute_hv(closes, hv_short, trading_days=trading_days)
    hv_90d = compute_hv(closes, hv_long, trading_days=trading_days)
    if hv_30d is None or hv_90d is None:
        raise ValueError(
            f"insufficient history for {symbol!r} as of {as_of}: need >={hv_long + 1} "
            f"closes, got {closes.len()} (hv_30d={hv_30d}, hv_90d={hv_90d})"
        )
    term_slope = hv_30d - hv_90d
    return IvSnapshot(
        symbol=symbol,
        as_of=as_of,
        iv_30d=hv_30d,
        iv_90d=hv_90d,
        term_slope=term_slope,
        put_skew_30d=0.0,
        put_skew_percentile_12m=None,
        backwardation=term_slope > 0.0,
        extreme_put_skew=False,
    )


# --- Provider protocol (parameterized swap: HV backtest -> IBKR live) --- #
class IvProvider(Protocol):
    """IV source protocol (plan §6.2). The sieve calls this, never bars/IBKR directly.

    Swapping HV (backtest) -> IBKR (live) is a one-line wiring change in the
    composition root; the sieve is untouched (D2 parameterization preserved).
    """

    def get_snapshot(self, symbol: str, as_of: date) -> IvSnapshot: ...


class BarLoader(Protocol):
    """Loads the daily bars for a symbol up to ``as_of`` for the HV provider.

    Returns a ``BAR_SCHEMA`` frame sorted ascending by ts; the provider trims to
    the lookback it needs. Injected so the provider is testable with synthetic
    bars and swappable to ``BarRepo`` / ``BarStore`` in production.
    """

    def __call__(self, symbol: str, as_of: date) -> pl.DataFrame: ...


class HvIvProvider:
    """Backtest IV provider: HV (realized vol) from daily bars (CEO 2026-07-18).

    Default for backtesting. ``bar_loader`` injects the bar source
    (``BarRepo``/``BarStore`` in prod; a synthetic fixture in tests).
    ``hv_short``/``hv_long`` set the two realized-vol windows (30/90 days by
    default). Put-skew smoke-detector fields are zeroed — they require option
    data and are populated only by the live ``IbkrIvProvider``.
    """

    def __init__(
        self,
        bar_loader: BarLoader,
        *,
        hv_short: int = 30,
        hv_long: int = 90,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.bar_loader = bar_loader
        self.hv_short = hv_short
        self.hv_long = hv_long
        self.trading_days = trading_days

    def get_snapshot(self, symbol: str, as_of: date) -> IvSnapshot:
        bars = self.bar_loader(symbol, as_of)
        return compute_hv_snapshot(
            bars,
            symbol,
            as_of,
            hv_short=self.hv_short,
            hv_long=self.hv_long,
            trading_days=self.trading_days,
        )


class IbkrIvProvider:
    """Live/paper IV provider via IBKR (CEO 2026-07-18). NOT wired in P0-1.

    Parameterized swap target for live trading: obtains IV / options data via
    IBKR (contract details / option chain) and fills the full ``IvSnapshot``
    including put-skew smoke-detector fields. Raises ``NotImplementedError``
    until the IBKR market-data wiring lands (P0-3); the sieve codes against
    ``IvProvider``, so enabling IBKR IV is a one-line composition-root change.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "IbkrIvProvider is the live/paper IV source via IBKR (CEO directive "
            "2026-07-18), not wired until the IBKR market-data integration lands "
            "(P0-3). For backtesting use HvIvProvider (HV proxy)."
        )

    def get_snapshot(self, symbol: str, as_of: date) -> IvSnapshot:  # pragma: no cover
        raise NotImplementedError
