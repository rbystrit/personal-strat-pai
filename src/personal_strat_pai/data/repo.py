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

The no-double-download math lives in ``data/caching.py`` (shared with
``FredRateRepo``, CEO 2026-07-19). This module wires it to bars: the ``kind``
parameter (daily / minute) selects the fetch step; everything else is the
generic ``compute_fetch_ranges`` logic.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol

import polars as pl

from personal_strat_pai.data.caching import (
    BOOTSTRAP_START,
    FetchRange,
    compute_fetch_ranges,
    to_date,
)
from personal_strat_pai.data.polars_utils import EagerFrame, assert_eager
from personal_strat_pai.data.store import BarKind, BarStore, Coverage

__all__ = [
    "BOOTSTRAP_START",
    "BarFetcher",
    "BarRepo",
    "FetchRange",
]


def _min_step(kind: BarKind) -> timedelta:
    """Smallest fetch step per kind — avoids re-downloading the last stored bar
    and avoids zero-width fetches when the store already covers ``[start, end)``."""
    return timedelta(days=1) if kind == "daily" else timedelta(minutes=1)


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
    ) -> dict[str, Coverage]:
        """Per-symbol ``(first_ts, last_ts)`` currently in the store."""
        return self.store.coverage(kind=kind, symbols=symbols)

    def missing_ranges(
        self,
        symbols: list[str],
        start: str | date | None,
        end: str | date | None,
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
        return compute_fetch_ranges(
            symbols,
            to_date(start) if start is not None else None,
            to_date(end),
            cov,
            bootstrap_start=self.bootstrap_start,
            step=_min_step(kind),
        )

    def get_bars(
        self,
        symbols: list[str],
        start: str | date | None,
        end: str | date,
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
        start_date = to_date(start) if start is not None else None
        end_date = to_date(end)
        ranges = compute_fetch_ranges(
            symbols,
            start_date,
            end_date,
            self.store.coverage(kind=kind, symbols=symbols),
            bootstrap_start=self.bootstrap_start,
            step=_min_step(kind),
        )
        for r in ranges:
            chunk = self.fetcher.fetch_bars([r.key], r.start, r.end, kind=kind)
            if chunk.is_empty():
                continue
            assert_eager(chunk, "BarFetcher.fetch_bars")
            self.store.upsert_bars(chunk, kind=kind)
        return self.store.scan_bars(kind=kind, symbols=symbols, start=start, end=end)
