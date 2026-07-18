"""Tests for data/repo.py — caching BarRepo: no piece of data downloaded twice (CEO 2026-07-18).

A recording fake fetcher stands in for databento so the no-double-download
guarantee is exercised with real parquet I/O in CI (no spend). Key invariants:

  * First request for a symbol bootstraps the max historical range available
    (``[bootstrap_start, end)``).
  * Subsequent requests fetch ONLY the forward gap ``[last_stored + 1, end)`` —
    never a range already in the store.
  * Re-requesting an already-cached range triggers ZERO fetcher calls.
  * Overlapping re-fetches never duplicate rows (upsert dedupes by (symbol, ts)).
  * A front gap (requested start < store first day) is fetched once, not re-fetched.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from personal_strat_pai.data.repo import BarRepo, FetchRange
from personal_strat_pai.data.store import BarStore


class FakeFetcher:
    """Records every (symbols, start, end, kind) call and returns synthetic bars.

    Stands in for ``DatabentoClient`` so the no-double-download logic is tested
    with real parquet I/O and zero spend. ``calls`` is the audit trail.
    """

    def __init__(self, bars_factory, *, seed: int = 7) -> None:
        self._bars_factory = bars_factory
        self.calls: list[tuple[list[str], date, date, str]] = []
        self._seed = seed

    def fetch_bars(
        self,
        symbols: list[str],
        start: date | str,
        end: date | str,
        *,
        kind: str = "daily",
    ) -> pl.DataFrame:
        s = start if isinstance(start, date) else date.fromisoformat(start)
        e = end if isinstance(end, date) else date.fromisoformat(end)
        self.calls.append(([s for s in symbols], s, e, kind))
        # Generate one row per calendar day in [s, e). Synthetic bars are dated
        # by calendar days so coverage math is deterministic in tests.
        start_dt = datetime(s.year, s.month, s.day, tzinfo=UTC)
        end_dt = datetime(e.year, e.month, e.day, tzinfo=UTC)
        return self._bars_factory(symbols, start_dt, end_dt, seed=self._seed)


@pytest.fixture
def repo(tmp_bars_dir, bars_range_factory) -> tuple[BarRepo, FakeFetcher]:
    store = BarStore(tmp_bars_dir)
    fetcher = FakeFetcher(bars_range_factory)
    return BarRepo(store, fetcher, bootstrap_start=date(2010, 1, 1)), fetcher


def _dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def test_bootstrap_pulls_max_range_on_first_request(repo):
    bar_repo, fetcher = repo
    end = date(2024, 1, 31)
    # First request with a narrow [start, end) — bootstrap pulls [bootstrap_start, end).
    lazy = bar_repo.get_bars(["XLB"], start="2024-01-29", end=end, kind="daily")
    out = lazy.collect().filter(pl.col("symbol") == "XLB").sort("ts")
    assert fetcher.calls == [(["XLB"], date(2010, 1, 1), end, "daily")]
    # The store now covers [bootstrap_start, end); the returned slice is the requested window.
    cov = bar_repo.coverage(kind="daily", symbols=["XLB"])
    assert cov["XLB"][0].date() == date(2010, 1, 1)
    assert cov["XLB"][1].date() >= end - timedelta(days=1)
    assert out["ts"].min().date() >= date(2024, 1, 29)
    assert out["ts"].max().date() < end


def test_incremental_fetch_only_forward_gap_no_double_download(repo):
    bar_repo, fetcher = repo
    # First request: bootstrap [2010-01-01, 2024-01-15).
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=date(2024, 1, 15), kind="daily").collect()
    first_call_count = len(fetcher.calls)
    assert first_call_count == 1
    first_cov = bar_repo.coverage(kind="daily", symbols=["XLB"])["XLB"]
    last_stored = first_cov[1].date()

    # Second request extending the end forward: fetch ONLY [last+1, new_end).
    new_end = last_stored + timedelta(days=10)
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=new_end, kind="daily").collect()
    assert len(fetcher.calls) == 2
    sym, s, e, kind = fetcher.calls[1]
    assert sym == ["XLB"]
    assert s == last_stored + timedelta(days=1), "must fetch only the forward gap"
    assert e == new_end
    assert kind == "daily"


def test_re_request_cached_range_triggers_zero_fetches(repo):
    bar_repo, fetcher = repo
    end = date(2024, 1, 20)
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=end, kind="daily").collect()
    assert len(fetcher.calls) == 1
    # Same range again -> pure store read, fetcher NOT called.
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=end, kind="daily").collect()
    assert len(fetcher.calls) == 1
    # Smaller subrange already cached -> also zero fetches.
    bar_repo.get_bars(["XLB"], start="2024-01-12", end=date(2024, 1, 18), kind="daily").collect()
    assert len(fetcher.calls) == 1


def test_overlapping_re_fetch_does_not_duplicate_rows(repo):
    bar_repo, _fetcher = repo
    end = date(2024, 1, 20)
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=end, kind="daily").collect()
    before = bar_repo.coverage(kind="daily", symbols=["XLB"])["XLB"]
    # Re-fetch an overlapping forward range (end pushed by 3 days).
    new_end = date(2024, 1, 23)
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=new_end, kind="daily").collect()
    after = bar_repo.coverage(kind="daily", symbols=["XLB"])["XLB"]
    assert after[1].date() >= new_end - timedelta(days=1)
    # No duplicate (symbol, ts) rows: count per day == 1.
    df = bar_repo.store.read_bars_eager(kind="daily", symbols=["XLB"])
    per_day = df.group_by("ts").len().sort("len", descending=True)
    assert per_day["len"].max() == 1
    # The front of coverage is unchanged (we did not rewrite the front).
    assert after[0] == before[0]


def test_front_gap_is_fetched_once(repo, bars_range_factory):
    bar_repo, fetcher = repo
    # Pre-populate the store with a NARROW recent range (not a bootstrap) so the
    # store's first day is recent and a front gap genuinely exists. (A bootstrap
    # always fills from bootstrap_start, so no front gap can occur after one.)
    narrow = bars_range_factory(
        ["XLB"],
        datetime(2024, 1, 10, tzinfo=UTC),
        datetime(2024, 1, 20, tzinfo=UTC),
        seed=7,
    )
    bar_repo.store.upsert_bars(narrow, kind="daily")
    first = bar_repo.coverage(kind="daily", symbols=["XLB"])["XLB"][0].date()
    assert first == date(2024, 1, 10)
    assert len(fetcher.calls) == 0  # the pre-populate bypassed the fetcher

    # Request a start BEFORE the store's first day -> front-gap fill, once.
    earlier_start = first - timedelta(days=5)
    bar_repo.get_bars(["XLB"], start=earlier_start, end=date(2024, 1, 20), kind="daily").collect()
    assert len(fetcher.calls) == 1
    sym, s, e, _kind = fetcher.calls[0]
    assert sym == ["XLB"]
    assert s == max(earlier_start, bar_repo.bootstrap_start)
    assert e == first  # exclusive end at the store's first day — no re-download
    # Coverage front now reaches the earlier start.
    cov = bar_repo.coverage(kind="daily", symbols=["XLB"])["XLB"]
    assert cov[0].date() <= earlier_start
    # Re-requesting the same earlier start -> zero new fetches (front gap now filled).
    bar_repo.get_bars(["XLB"], start=earlier_start, end=date(2024, 1, 20), kind="daily").collect()
    assert len(fetcher.calls) == 1


def test_front_gap_capped_at_bootstrap_start(repo, bars_range_factory):
    bar_repo, fetcher = repo
    # Pre-populate a narrow recent range so a front gap exists; ask for a start
    # older than bootstrap_start and assert the fetch is capped to bootstrap_start.
    narrow = bars_range_factory(
        ["XLB"],
        datetime(2024, 1, 10, tzinfo=UTC),
        datetime(2024, 1, 20, tzinfo=UTC),
        seed=7,
    )
    bar_repo.store.upsert_bars(narrow, kind="daily")
    first = bar_repo.coverage(kind="daily", symbols=["XLB"])["XLB"][0].date()
    very_old = date(1990, 1, 1)  # before bootstrap_start (2010-01-01)
    bar_repo.get_bars(["XLB"], start=very_old, end=date(2024, 1, 20), kind="daily").collect()
    sym, s, e, _kind = fetcher.calls[0]
    assert sym == ["XLB"]
    assert s == bar_repo.bootstrap_start  # capped, not 1990-01-01
    assert e == first


def test_missing_ranges_pure_computation_no_fetch(repo):
    bar_repo, fetcher = repo
    # Bootstrap once.
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=date(2024, 1, 20), kind="daily").collect()
    n_before = len(fetcher.calls)
    ranges = bar_repo.missing_ranges(["XLB"], start="2024-01-25", end=date(2024, 2, 5))
    # No fetch happened.
    assert len(fetcher.calls) == n_before
    assert len(ranges) == 1
    assert isinstance(ranges[0], FetchRange)
    assert ranges[0].symbol == "XLB"
    last_stored = bar_repo.coverage(kind="daily", symbols=["XLB"])["XLB"][1].date()
    assert ranges[0].start == last_stored + timedelta(days=1)
    assert ranges[0].end == date(2024, 2, 5)


def test_missing_ranges_empty_when_fully_cached(repo):
    bar_repo, _fetcher = repo
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=date(2024, 1, 20), kind="daily").collect()
    ranges = bar_repo.missing_ranges(["XLB"], start="2024-01-12", end=date(2024, 1, 18))
    assert ranges == []


def test_missing_ranges_bootstrap_for_new_symbol(repo):
    bar_repo, _fetcher = repo
    # XLB is cached, XLY is new -> XLY bootstraps, XLB is a no-op.
    bar_repo.get_bars(["XLB"], start="2024-01-10", end=date(2024, 1, 20), kind="daily").collect()
    ranges = bar_repo.missing_ranges(["XLB", "XLY"], start="2024-01-15", end=date(2024, 1, 25))
    # Only XLY (the new symbol) needs a fetch — XLB may also need a forward gap
    # if 2024-01-25 > its last stored day.
    symbols = [r.symbol for r in ranges]
    assert "XLY" in symbols
    xly_range = next(r for r in ranges if r.symbol == "XLY")
    assert xly_range.start == bar_repo.bootstrap_start
    assert xly_range.end == date(2024, 1, 25)


def test_unbounded_end_rejected(repo):
    bar_repo, _fetcher = repo
    with pytest.raises(ValueError, match="`end`"):
        bar_repo.missing_ranges(["XLB"], start="2024-01-10", end=None)  # type: ignore[arg-type]


def test_get_bars_returns_lazyframe(repo):
    bar_repo, _fetcher = repo
    lazy = bar_repo.get_bars(["XLB"], start="2024-01-10", end=date(2024, 1, 20), kind="daily")
    assert isinstance(lazy, pl.LazyFrame)
