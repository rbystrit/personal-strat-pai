"""Tests for data/caching.py — shared no-double-download fetch-range math.

The same ``compute_fetch_ranges`` drives both ``BarRepo`` (data/repo.py) and
``FredRateRepo`` (data/rates.py). This suite exercises the generic invariants
once, independent of any specific store/fetcher, so the policy is auditable in
one place:

  * Empty coverage => bootstrap ``[bootstrap_start, end)``.
  * Partial coverage => forward gap ``[last + step, end)`` only when non-empty;
    plus a defensive front-gap ``[max(start, bootstrap_start), first)``.
  * Full coverage => no fetch (re-request is a pure store read).
  * Front gap is capped at ``bootstrap_start`` (no unbounded back-fill).
  * ``end`` is required (unbounded fetches are rejected).
  * ``step`` controls the forward-gap offset (daily vs minute).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from personal_strat_pai.data.caching import (
    BOOTSTRAP_START,
    FetchRange,
    compute_fetch_ranges,
    to_date,
)


def _dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def test_to_date_parses_iso_string_naive():
    assert to_date("2024-01-15") == date(2024, 1, 15)


def test_to_date_parses_iso_string_tz_aware():
    assert to_date("2024-01-15T10:00:00+00:00") == date(2024, 1, 15)


def test_to_date_passes_through_date():
    d = date(2024, 1, 15)
    assert to_date(d) is d


def test_to_date_strips_tz_from_datetime():
    assert to_date(datetime(2024, 1, 15, 10, 0, tzinfo=UTC)) == date(2024, 1, 15)


def test_to_date_none_without_default_raises():
    with pytest.raises(ValueError):
        to_date(None)


def test_to_date_none_with_default_returns_default():
    assert to_date(None, default=date(2020, 1, 1)) == date(2020, 1, 1)


def test_bootstrap_when_coverage_empty():
    ranges = compute_fetch_ranges(
        ["SOFR"],
        start=date(2024, 1, 10),
        end=date(2024, 1, 31),
        coverage={},
        bootstrap_start=date(2000, 1, 1),
    )
    assert len(ranges) == 1
    assert ranges[0] == FetchRange(key="SOFR", start=date(2000, 1, 1), end=date(2024, 1, 31))


def test_bootstrap_skipped_when_bootstrap_start_ge_end():
    # bootstrap_start >= end => no fetch (range is empty / inverted).
    ranges = compute_fetch_ranges(
        ["SOFR"],
        start=None,
        end=date(2000, 1, 1),
        coverage={},
        bootstrap_start=date(2000, 1, 1),
    )
    assert ranges == []


def test_forward_gap_only_when_partially_cached():
    coverage = {"SOFR": (_dt(date(2020, 1, 1)), _dt(date(2024, 1, 20)))}
    ranges = compute_fetch_ranges(
        ["SOFR"],
        start=date(2024, 1, 10),
        end=date(2024, 1, 31),
        coverage=coverage,
        bootstrap_start=date(2000, 1, 1),
    )
    # Forward gap only: [2024-01-21, 2024-01-31). No front gap (start >= first).
    assert ranges == [FetchRange(key="SOFR", start=date(2024, 1, 21), end=date(2024, 1, 31))]


def test_no_fetch_when_fully_cached():
    coverage = {"SOFR": (_dt(date(2020, 1, 1)), _dt(date(2024, 1, 30)))}
    # Request [2024-01-10, 2024-01-31) — store already covers up to 2024-01-30,
    # so fwd_start = 2024-01-31 == end => no forward gap. No front gap either.
    ranges = compute_fetch_ranges(
        ["SOFR"],
        start=date(2024, 1, 10),
        end=date(2024, 1, 31),
        coverage=coverage,
        bootstrap_start=date(2000, 1, 1),
    )
    assert ranges == []


def test_front_gap_filled_when_start_before_first():
    coverage = {"SOFR": (_dt(date(2020, 1, 1)), _dt(date(2024, 1, 20)))}
    # Request start=2019-12-15 (before first=2020-01-01) AND end=2024-01-25.
    # Forward gap: [2024-01-21, 2024-01-25). Front gap: [2019-12-15, 2020-01-01).
    ranges = compute_fetch_ranges(
        ["SOFR"],
        start=date(2019, 12, 15),
        end=date(2024, 1, 25),
        coverage=coverage,
        bootstrap_start=date(2000, 1, 1),
    )
    starts = {(r.key, r.start, r.end) for r in ranges}
    assert (
        FetchRange(key="SOFR", start=date(2024, 1, 21), end=date(2024, 1, 25)).key,
        date(2024, 1, 21),
        date(2024, 1, 25),
    ) in starts
    assert ("SOFR", date(2019, 12, 15), date(2020, 1, 1)) in starts


def test_front_gap_capped_at_bootstrap_start():
    coverage = {"SOFR": (_dt(date(2020, 1, 1)), _dt(date(2024, 1, 20)))}
    # Ask for a start older than bootstrap_start — front gap must be capped.
    ranges = compute_fetch_ranges(
        ["SOFR"],
        start=date(1990, 1, 1),
        end=date(2024, 1, 25),
        coverage=coverage,
        bootstrap_start=date(2000, 1, 1),
    )
    front = next(r for r in ranges if r.start < date(2020, 1, 1))
    assert front.start == date(2000, 1, 1)  # capped, not 1990-01-01
    assert front.end == date(2020, 1, 1)


def test_step_controls_forward_gap_offset():
    # step=2 days => forward gap starts 2 days after the last stored ts.
    coverage = {"SOFR": (_dt(date(2024, 1, 1)), _dt(date(2024, 1, 20)))}
    ranges = compute_fetch_ranges(
        ["SOFR"],
        start=date(2024, 1, 10),
        end=date(2024, 1, 31),
        coverage=coverage,
        bootstrap_start=date(2000, 1, 1),
        step=timedelta(days=2),
    )
    assert ranges == [FetchRange(key="SOFR", start=date(2024, 1, 22), end=date(2024, 1, 31))]


def test_unbounded_end_rejected():
    with pytest.raises(ValueError, match="`end`"):
        compute_fetch_ranges(
            ["SOFR"],
            start=date(2024, 1, 1),
            end=None,  # type: ignore[arg-type]
            coverage={},
            bootstrap_start=date(2000, 1, 1),
        )


def test_multiple_keys_bootstrap_and_incremental_mixed():
    coverage = {"SOFR": (_dt(date(2020, 1, 1)), _dt(date(2024, 1, 20)))}
    # SOFR is cached; DGS10 is new -> bootstrap; DGS10 not in coverage.
    ranges = compute_fetch_ranges(
        ["SOFR", "DGS10"],
        start=date(2024, 1, 10),
        end=date(2024, 1, 31),
        coverage=coverage,
        bootstrap_start=date(2000, 1, 1),
    )
    by_key = {r.key: r for r in ranges}
    assert by_key["SOFR"].start == date(2024, 1, 21)  # forward gap
    assert by_key["SOFR"].end == date(2024, 1, 31)
    assert by_key["DGS10"].start == date(2000, 1, 1)  # bootstrap
    assert by_key["DGS10"].end == date(2024, 1, 31)


def test_fetch_range_symbol_alias_for_key():
    """Legacy ``FetchRange.symbol`` reads ``key`` (bar callers depend on it)."""
    r = FetchRange(key="XLB", start=date(2024, 1, 1), end=date(2024, 1, 31))
    assert r.symbol == "XLB"
    assert r.key == "XLB"


def test_default_bootstrap_start_is_far_back():
    # Sanity: the module-level default is a far-back date so first-time pulls
    # fetch the max historical range the remote has.
    assert BOOTSTRAP_START.year <= 2000
