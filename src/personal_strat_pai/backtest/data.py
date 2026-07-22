"""Bar driver for backtesting — polars lazy scan_parquet (plan §10, D14).

The bar driver is the **data source** the backtester replaces for live trading.
Live mode reads from IBKR market data; backtest mode reads from parquet. The
key discipline (D14): the scan stays lazy until the per-day decision boundary,
then ``.collect()`` produces an eager frame for the signal computation and
pre-trade risk checks.

Design (plan §10 acceptance criteria):
  * ``BarDriver`` wraps a parquet source (a ``BarStore`` or a raw path/glob).
  * ``scan_range`` returns a ``LazyFrame`` with symbol/date predicate pushdown
    so partitioned parquet skips unneeded symbol/year partitions on read.
  * ``bars_through`` collects all bars up to and including ``as_of`` — the
    eager frame the signal sees at a rebalance boundary. No look-ahead: the
    signal only sees data at or before the decision timestamp.
  * ``fill_bar_for`` returns the next bar after ``as_of`` for execution
    (``trade_on="next_open"`` fills at the next bar's open price).
  * ``trading_days`` returns the sorted unique dates in the data — the
    engine's calendar.

The driver never leaks a ``LazyFrame`` across its API boundary: all public
methods that return data return eager ``pl.DataFrame`` (D14(b)).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from personal_strat_pai.data.polars_utils import (
    BAR_SCHEMA,
    EagerFrame,
    LazyFrame,
    collect_eager,
    to_utc_datetime,
)

__all__ = [
    "BarDriver",
    "ParquetBarDriver",
    "TradingCalendar",
]


class BarDriver(Protocol):
    """Bar data source protocol — backtest and live share the same interface.

    The backtester codes against this protocol; ``ParquetBarDriver`` is the
    backtest implementation (parquet via polars lazy scan). The live
    implementation (P0-4) reads from IBKR market data and produces the same
    eager frames — one code path (plan §10).
    """

    def bars_through(self, symbols: list[str], as_of: datetime) -> EagerFrame:
        """All bars for ``symbols`` up to and including ``as_of`` (eager)."""
        ...

    def fill_bar_for(
        self, symbols: list[str], after: datetime
    ) -> EagerFrame:
        """The first bar after ``after`` for each symbol (eager) — for fills."""
        ...

    def trading_days(self) -> list[date]:
        """Sorted unique trading dates in the data — the engine's calendar."""
        ...

    def all_symbols(self) -> list[str]:
        """All symbols present in the data."""
        ...


class ParquetBarDriver:
    """Parquet-backed bar driver — polars lazy scan_parquet (D14, plan §10).

    Wraps a parquet directory (hive-partitioned by ``symbol``) or a list of
    parquet files. The scan stays lazy with predicate pushdown; ``bars_through``
    and ``fill_bar_for`` collect eagerly at the decision boundary.

    ``trade_on="next_open"`` (plan §10): signals are decided at bar *t*'s
    close and fills execute at bar *t+1*'s open. The engine calls
    ``bars_through(t_close)`` for the signal, then ``fill_bar_for(t_close)``
    for the execution price (next bar's open).

    Reproducibility: the manifest captures the parquet data hash so a run can
    be verified against the exact bytes (plan §10 acceptance criterion).
    """

    def __init__(
        self,
        source: str | Path | list[str | Path],
        *,
        symbols: list[str] | None = None,
        start: str | datetime | date | None = None,
        end: str | datetime | date | None = None,
    ) -> None:
        self._source = source
        self._symbols = symbols
        self._start = start
        self._end = end
        # Cache the trading-day calendar on first access.
        self._calendar: list[date] | None = None
        self._all_symbols_cache: list[str] | None = None

    def _scan(self, symbols: list[str] | None = None) -> LazyFrame:
        """Lazy scan over the parquet source with predicate pushdown (D14(a))."""
        lf: LazyFrame = pl.scan_parquet(self._source, hive_partitioning=True)  # type: ignore[arg-type]
        preds: list[pl.Expr] = []
        syms = symbols or self._symbols
        if syms is not None:
            preds.append(pl.col("symbol").is_in(syms))
        start_dt = to_utc_datetime(self._start)
        end_dt = to_utc_datetime(self._end)
        if start_dt is not None:
            preds.append(pl.col("ts") >= pl.lit(start_dt))
        if end_dt is not None:
            preds.append(pl.col("ts") < pl.lit(end_dt))
        if preds:
            lf = lf.filter(pl.all_horizontal(preds))
        return lf.select(list(BAR_SCHEMA.names()))

    def bars_through(self, symbols: list[str], as_of: datetime) -> EagerFrame:
        """All bars for ``symbols`` up to and including ``as_of`` — eager (D14(b)).

        No look-ahead: filters ``ts <= as_of``. Returns an eager frame sorted
        by ``(symbol, ts)`` so the signal computation sees chronological order.
        """
        as_of_dt = to_utc_datetime(as_of) or as_of
        lf = self._scan(symbols).filter(pl.col("ts") <= pl.lit(as_of_dt))
        return collect_eager(lf).sort(["symbol", "ts"])

    def fill_bar_for(self, symbols: list[str], after: datetime) -> EagerFrame:
        """The first bar after ``after`` for each symbol — eager (for fills).

        For ``trade_on="next_open"``: the engine decides at bar *t*'s close
        and fills at bar *t+1*'s open. This returns that next bar so the
        cost model can fill at its open price. Returns one row per symbol
        (the first bar after ``after``).
        """
        after_dt = to_utc_datetime(after) or after
        lf = (
            self._scan(symbols)
            .filter(pl.col("ts") > pl.lit(after_dt))
            .sort("ts")
        )
        eager = collect_eager(lf)
        if eager.is_empty():
            return eager
        # First bar per symbol after the decision timestamp.
        return eager.group_by("symbol", maintain_order=True).first().sort("symbol")

    def trading_days(self) -> list[date]:
        """Sorted unique trading dates in the data — the engine's calendar.

        Eager, small aggregation — computed once and cached.
        """
        if self._calendar is not None:
            return self._calendar
        lf = self._scan()
        days = (
            lf.select(pl.col("ts").dt.date().unique().sort())
            .collect()
            .to_series()
            .to_list()
        )
        self._calendar = days
        return days

    def all_symbols(self) -> list[str]:
        """All symbols present in the data (sorted)."""
        if self._all_symbols_cache is not None:
            return self._all_symbols_cache
        lf = self._scan()
        syms = (
            lf.select(pl.col("symbol").unique().sort())
            .collect()
            .to_series()
            .to_list()
        )
        self._all_symbols_cache = syms
        return syms


class TradingCalendar:
    """Trading-day calendar utilities for the engine.

    Wraps the sorted list of trading dates from the bar driver. Provides
    month-end detection (last trading day of each month) — the rebalance
    cadence for the Flat Momentum Strategy (brief §3: "On the final trading
    day of each month at the closing bell").
    """

    def __init__(self, days: list[date]) -> None:
        self._days = sorted(days)
        self._day_set = set(self._days)

    @property
    def days(self) -> list[date]:
        """All trading days, sorted ascending."""
        return list(self._days)

    def is_trading_day(self, d: date) -> bool:
        return d in self._day_set

    def month_ends(self) -> list[date]:
        """Last trading day of each month — the rebalance dates (brief §3)."""
        if not self._days:
            return []
        ends: list[date] = []
        for d in self._days:
            if not ends or (d.year != ends[-1].year or d.month != ends[-1].month):
                ends.append(d)
            else:
                ends[-1] = d
        return ends

    def next_trading_day_after(self, d: date) -> date | None:
        """First trading day strictly after ``d``, or ``None`` if past the end."""
        for td in self._days:
            if td > d:
                return td
        return None
