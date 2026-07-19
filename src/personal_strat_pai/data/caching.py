"""Shared no-double-download fetch-range math (CEO directive 2026-07-18 / 2026-07-19).

Both ``BarRepo`` (data/repo.py) and ``FredRateRepo`` (data/rates.py) compose a
remote fetcher with a parquet store. The missing-range computation is identical:
given the store's per-key coverage and the requested ``[start, end)``, compute
the disjoint, non-empty ranges the fetcher must be called for so **no piece of
data is ever downloaded twice**:

  * Empty coverage for a key => bootstrap ``[bootstrap_start, end)`` (the max
    historical range the remote has for that key — the backtesting corpus).
  * Partial coverage => forward gap ``[last + step, end)`` only when non-empty;
    plus a defensive front-gap ``[max(start, bootstrap_start), first)`` when the
    caller asks for a start older than the store's first day (fetched once, then
    cached — never re-requested).
  * Full coverage => no fetch (re-request is a pure store read; fetcher is
    never called).

This is the single home for the no-double-download logic. ``BarRepo`` was the
first consumer (CEO 2026-07-18); ``FredRateRepo`` follows the same policy per
CEO 2026-07-19 ("For FRED, follow the same download once policy").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

__all__ = [
    "BOOTSTRAP_START",
    "FetchRange",
    "compute_fetch_ranges",
    "to_date",
]

# Far-back default for the bootstrap (first-time) pull. databento DBEQ-BARS-1D
# history begins ~2007-2008 for most of the ~45-ETF universe; FRED Treasury CMT
# series go back to 1962, TIPS to ~1997, SOFR to 2018-04. Asking from
# 2000-01-01 lets each remote return whatever history the dataset actually has;
# override per-repo for datasets with a known later inception.
BOOTSTRAP_START: date = date(2000, 1, 1)


@dataclass(frozen=True, slots=True)
class FetchRange:
    """A single ``[start, end)`` range to fetch from the remote for one key.

    ``start`` inclusive, ``end`` exclusive (ISO-8601 date conventions, matches
    ``BarStore.scan_bars`` / ``RateSeriesStore.scan_observations``). Half-open
    so adjacent ranges tile without overlap.

    ``key`` is the generic identifier (a bar ticker OR a FRED series id); the
    legacy ``symbol`` alias is preserved so existing ``BarRepo`` tests and
    callers don't churn.
    """

    key: str
    start: date
    end: date

    @property
    def symbol(self) -> str:
        """Legacy alias for ``key`` (bar callers read ``FetchRange.symbol``)."""
        return self.key


def to_date(value: str | datetime | date | None, *, default: date | None = None) -> date:
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


def compute_fetch_ranges(
    keys: list[str],
    start: date | None,
    end: date,
    coverage: dict[str, tuple[datetime, datetime]],
    *,
    bootstrap_start: date,
    step: timedelta = timedelta(days=1),
) -> list[FetchRange]:
    """Compute the missing ranges to fetch per key (no double download).

    Empty coverage for a key => bootstrap ``[bootstrap_start, end)``.
    Otherwise => forward gap ``[last + step, end)`` only when
    ``last + step < end`` (so a store that already covers ``[start, end)``
    triggers NO fetch), plus a defensive front-gap
    ``[max(start, bootstrap_start), first)`` only when that range is non-empty.
    Returns the disjoint, non-empty ranges to fetch.

    ``step`` is the smallest fetch step — avoids re-downloading the last stored
    observation and avoids zero-width fetches when the store already covers
    ``[start, end)``. Daily bars / FRED observations use ``timedelta(days=1)``;
    minute bars use ``timedelta(minutes=1)``.
    """
    if end is None:
        raise ValueError("caching repo requires an `end` bound (no unbounded fetches).")
    ranges: list[FetchRange] = []
    for key in keys:
        if key not in coverage:
            if bootstrap_start < end:
                ranges.append(FetchRange(key, bootstrap_start, end))
            continue
        first_dt, last_dt = coverage[key]
        first, last = first_dt.date(), last_dt.date()
        # Forward gap: fetch [last + step, end) — strictly beyond the last stored
        # observation so we never re-download it. Skipped when the store already
        # reaches up to (end - step), i.e. fully covers the requested [start, end).
        fwd_start = last + step
        if fwd_start < end:
            ranges.append(FetchRange(key, fwd_start, end))
        # Front gap: fetch [max(start, bootstrap_start), first) — dates before
        # the store's first observation, capped at the max-history floor. Skipped
        # when the caller didn't request an earlier start or the cap empties it.
        if start is not None and start < first:
            front_start = max(start, bootstrap_start)
            if front_start < first:
                ranges.append(FetchRange(key, front_start, first))
    return ranges
