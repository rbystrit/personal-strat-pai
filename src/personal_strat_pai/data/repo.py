"""Caching bar repo — no piece of data downloaded twice (CEO directive 2026-07-18).

Sits between a remote ``BarFetcher`` (databento primary) and the parquet
``BarStore`` (Object Storage backing). Every request is served from the store;
the fetcher is only ever called for the range that is NOT already cached:

  * **First time** (store empty for a symbol): bootstrap with the **max
    historical range available** — ``[bootstrap_start, end)``. ``bootstrap_start``
    defaults to a far-back date so the fetcher returns whatever history the
    dataset has; this is the backtesting corpus.
  * **Not the first time**: fetch **only** ``[last_stored_day + 1, end)`` — the
    forward gap from the latest day in object storage up to the requested date.
    A defensive front-gap fill (``[start, first_stored_day)``) also runs when the
    caller asks for a start older than the store's first day, so a missing front
    window is fetched once and never re-requested.

The merge into the store is idempotent (``BarStore.upsert_bars`` dedupes by
``(symbol, ts)`` keep-latest), so overlapping fetches never duplicate rows.

Backing store: local ``file://`` is fully exercised in CI with a synthetic
fetcher. The ``oci://`` Object Storage backend lands in P0-2 (creds); the
caching logic is identical — only the store's ``base_uri`` changes.

This module is the bars path. Rates (data/rates.py) and corp actions
(data/corp_actions.py) follow the same no-double-download pattern when their
live loaders land (P0-2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

import polars as pl

from personal_strat_pai.data.polars_utils import EagerFrame, assert_eager
from personal_strat_pai.data.store import BarKind, BarStore

__all__ = [
    "BOOTSTRAP_START",
    "BarFetcher",
    "BarRepo",
    "FetchRange",
]

# Far-back default for the bootstrap (first-time) pull. databento DBEQ-BARS-1D
# history begins ~2007-2008 for most of the ~45-ETF universe; asking from
# 2000-01-01 lets the fetcher return the max range each dataset actually has.
# Override per-repo for datasets with a known later inception.
BOOTSTRAP_START: date = date(2000, 1, 1)


@dataclass(frozen=True, slots=True)
class FetchRange:
    """A single (start, end] range to fetch from the remote for one symbol.

    ``start`` inclusive, ``end`` exclusive (ISO-8601 date conventions, matches
    ``BarStore.scan_bars``). Half-open so adjacent ranges tile without overlap.
    """

    symbol: str
    start: date
    end: date


class BarFetcher(Protocol):
    """Remote bar source protocol (databento primary; fake in tests).

    Implementations return a normalized ``EagerFrame`` conforming to
    ``BAR_SCHEMA`` for the requested symbols over ``[start, end)``. The caching
    repo NEVER calls this for a range already in the store.
    """

    def fetch_bars(
        self,
        symbols: list[str],
        start: date | str,
        end: date | str,
        *,
        kind: BarKind = "daily",
    ) -> EagerFrame: ...


def _to_date(value: str | datetime | date | None, *, default: date | None = None) -> date:
    """Coerce a bound to a naive date (UTC if datetime). Used for fetch-range math."""
    if value is None:
        if default is None:
            raise ValueError("a date bound is required here but got None")
        return default
    if isinstance(value, datetime):
        return value.astimezone(UTC).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(UTC).date() if parsed.tzinfo else parsed.date()


def _min_step(kind: BarKind) -> timedelta:
    """Smallest fetch step per kind — avoids re-downloading the last stored bar
    and avoids zero-width fetches when the store already covers ``[start, end)``."""
    return timedelta(days=1) if kind == "daily" else timedelta(minutes=1)


def _compute_fetch_ranges(
    symbols: list[str],
    start: date | None,
    end: date | None,
    coverage: dict[str, tuple[datetime, datetime]],
    *,
    bootstrap_start: date,
    kind: BarKind = "daily",
) -> list[FetchRange]:
    """Compute the missing ranges to fetch per symbol (no double download).

    Empty coverage for a symbol => bootstrap ``[bootstrap_start, end)``.
    Otherwise => forward gap ``[last + step, end)`` only when ``last + step < end``
    (so a store that already covers ``[start, end)`` triggers NO fetch), plus a
    defensive front-gap ``[max(start, bootstrap_start), first)`` only when that
    range is non-empty. Returns the disjoint, non-empty ranges to fetch.
    """
    if end is None:
        raise ValueError("BarRepo requires an `end` bound (no unbounded fetches).")
    step = _min_step(kind)
    ranges: list[FetchRange] = []
    for sym in symbols:
        if sym not in coverage:
            if bootstrap_start < end:
                ranges.append(FetchRange(sym, bootstrap_start, end))
            continue
        first_dt, last_dt = coverage[sym]
        first, last = first_dt.date(), last_dt.date()
        # Forward gap: fetch [last + step, end) — strictly beyond the last stored
        # bar so we never re-download it. Skipped when the store already reaches
        # up to (end - step), i.e. fully covers the requested [start, end).
        fwd_start = last + step
        if fwd_start < end:
            ranges.append(FetchRange(sym, fwd_start, end))
        # Front gap: fetch [max(start, bootstrap_start), first) — dates before the
        # store's first bar, capped at the max-history floor. Skipped when empty.
        if start is not None and start < first:
            front_start = max(start, bootstrap_start)
            if front_start < first:
                ranges.append(FetchRange(sym, front_start, first))
    return ranges


class BarRepo:
    """Caching bar repo (CEO directive 2026-07-18: no piece of data downloaded twice).

    Composes a ``BarStore`` (the cache / Object Storage backing) with a
    ``BarFetcher`` (the remote). ``get_bars`` returns a lazy scan over the
    requested range, fetching only the missing pieces first. Re-requesting an
    already-cached range is a pure store read — the fetcher is never called.
    """

    def __init__(
        self,
        store: BarStore,
        fetcher: BarFetcher,
        *,
        bootstrap_start: date = BOOTSTRAP_START,
    ) -> None:
        self.store = store
        self.fetcher = fetcher
        self.bootstrap_start = bootstrap_start

    def coverage(
        self,
        kind: BarKind = "daily",
        *,
        symbols: list[str] | None = None,
    ) -> dict[str, tuple[datetime, datetime]]:
        """Per-symbol ``(first_ts, last_ts)`` currently in the store."""
        return self.store.coverage(kind=kind, symbols=symbols)

    def missing_ranges(
        self,
        symbols: list[str],
        start: str | datetime | date | None,
        end: str | datetime | date | None,
        *,
        kind: BarKind = "daily",
    ) -> list[FetchRange]:
        """The ranges the fetcher would be called for — pure computation, no fetch.

        Exposed for observability and tests: assert no range overlaps the store's
        existing coverage. ``end`` is required (unbounded fetches are rejected).
        """
        if end is None:
            raise ValueError("BarRepo requires an `end` bound (no unbounded fetches).")
        cov = self.store.coverage(kind=kind, symbols=symbols)
        return _compute_fetch_ranges(
            symbols,
            _to_date(start, default=None) if start is not None else None,
            _to_date(end),
            cov,
            bootstrap_start=self.bootstrap_start,
            kind=kind,
        )

    def get_bars(
        self,
        symbols: list[str],
        start: str | datetime | date | None,
        end: str | datetime | date,
        *,
        kind: BarKind = "daily",
    ) -> pl.LazyFrame:
        """Return a lazy scan of ``[start, end)`` for ``symbols``, fetching only gaps.

        1. Compute the missing ranges vs the store's coverage.
        2. For each disjoint range, call the fetcher and upsert into the store
           (idempotent — overlapping re-fetches dedupe by ``(symbol, ts)``).
        3. Scan the (now-complete) store over the requested range and return it.

        The caller MUST ``collect_eager()`` at a strategy boundary (D14(b)).
        """
        if end is None:
            raise ValueError("BarRepo requires an `end` bound (no unbounded fetches).")
        start_date = _to_date(start, default=None) if start is not None else None
        end_date = _to_date(end)
        ranges = _compute_fetch_ranges(
            symbols,
            start_date,
            end_date,
            self.store.coverage(kind=kind, symbols=symbols),
            bootstrap_start=self.bootstrap_start,
            kind=kind,
        )
        for r in ranges:
            chunk = self.fetcher.fetch_bars([r.symbol], r.start, r.end, kind=kind)
            if chunk.is_empty():
                continue
            assert_eager(chunk, "BarFetcher.fetch_bars")
            self.store.upsert_bars(chunk, kind=kind)
        return self.store.scan_bars(kind=kind, symbols=symbols, start=start, end=end)
