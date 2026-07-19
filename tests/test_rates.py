"""Tests for data/rates.py — SOFR/OIS curve + real rates (D3; FRED primary for history).

Coverage:
  * SyntheticRatesProvider + RatesCurve slope logic (pre-existing).
  * DatabentoRatesProvider stub still gated on its API key (live forward curve, P0-3).
  * RateSeriesStore parquet round-trip: write/scan/upsert/coverage; upsert dedupes
    by (series, ts) keep-latest.
  * FredRateRepo no-double-download (CEO 2026-07-19): bootstrap max range first
    time; forward-gap-only fetch after; re-request = 0 fetches; overlapping
    re-fetch = no duplicate rows; front-gap fetched once then cached; front-gap
    capped at bootstrap_start.
  * FredRatesProvider builds a RatesCurve from the cache: SOFR flat across
    tenors, OIS proxy at matching tenors, real rates at 5/10/30y with nan
    elsewhere; falls back to DEFAULT_RISK_FREE_RATE for missing OIS tenors.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from personal_strat_pai.data.fred import (
    ALL_FRED_SERIES_IDS,
    FRED_OIS_SERIES,
    FRED_REAL_SERIES,
    FRED_SOFR_SERIES,
)
from personal_strat_pai.data.polars_utils import RATE_OBSERVATION_SCHEMA
from personal_strat_pai.data.rates import (
    CURVE_TENORS_YEARS,
    DEFAULT_RISK_FREE_RATE,
    DatabentoRatesProvider,
    FredRateRepo,
    FredRatesProvider,
    RatesCurve,
    SyntheticRatesProvider,
    risk_free_continuous,
)
from personal_strat_pai.data.store import RateSeriesStore


def test_synthetic_provider_returns_curve():
    p = SyntheticRatesProvider()
    curve = p.get_curve(date(2024, 1, 2))
    assert isinstance(curve, RatesCurve)
    assert curve.ois_slope_2y10y is not None
    # normal curve: 10y > 2y => positive slope
    assert curve.ois_slope_2y10y > 0.0


def test_ois_slope_inverted_when_10y_below_2y():
    curve = RatesCurve(
        as_of=date(2024, 1, 2),
        tenors_years=[2.0, 10.0],
        ois_rates=[0.05, 0.04],  # inverted: 10y < 2y
        sofr_rates=[0.05, 0.04],
    )
    assert curve.ois_slope_2y10y is not None
    assert curve.ois_slope_2y10y < 0.0


def test_ois_slope_none_when_tenors_missing():
    curve = RatesCurve(as_of=date(2024, 1, 2))
    assert curve.ois_slope_2y10y is None


def test_risk_free_continuous_uses_curve():
    p = SyntheticRatesProvider(ois=[0.04, 0.04, 0.04, 0.045, 0.05, 0.052, 0.055])
    r = risk_free_continuous(p, date(2024, 1, 2), tenor_years=1.0)
    assert r == pytest.approx(0.04)


def test_risk_free_continuous_falls_back_when_curve_empty():
    p = SyntheticRatesProvider(ois=[], sofr=[], tenors=[])
    r = risk_free_continuous(p, date(2024, 1, 2))
    assert r == DEFAULT_RISK_FREE_RATE


def test_databento_provider_requires_key():
    p = DatabentoRatesProvider(api_key=None)
    with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
        p.get_curve(date(2024, 1, 2))


# --- RateSeriesStore (parquet round-trip + upsert dedupe) --- #


def _obs(
    series_id: str,
    start: datetime,
    days: int,
    *,
    rate_start: float = 0.045,
    seed: int = 7,
) -> pl.DataFrame:
    """Deterministic synthetic FRED-like observations over [start, start+days) (daily)."""
    import random

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    rate = rate_start
    for d in range(days):
        ts = start + timedelta(days=d)
        # small daily drift so rates vary day to day
        rate = max(rate + rng.gauss(0.0, 0.0005), 1e-4)
        rows.append(
            {
                "series": series_id,
                "ts": ts,
                "rate": float(round(rate, 6)),
                "source": "FRED",
            }
        )
    return pl.DataFrame(rows, schema=RATE_OBSERVATION_SCHEMA)


@pytest.fixture
def tmp_rates_dir(tmp_path: Path) -> Path:
    d = tmp_path / "rates"
    d.mkdir()
    return d


def test_rate_series_store_write_scan_round_trip(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    df = _obs("SOFR", datetime(2024, 1, 2, tzinfo=UTC), days=10)
    store.write_observations(df)
    out = store.scan_observations(series=["SOFR"]).collect().sort("ts")
    assert out.height == 10
    assert list(out.columns) == list(RATE_OBSERVATION_SCHEMA.names())
    assert out["series"].unique().to_list() == ["SOFR"]


def test_rate_series_store_scan_predicate_pushdown(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    df = _obs("DGS10", datetime(2024, 1, 2, tzinfo=UTC), days=20)
    store.write_observations(df)
    out = (
        store.scan_observations(
            series=["DGS10"],
            start="2024-01-10",
            end="2024-01-15",
        )
        .collect()
        .sort("ts")
    )
    # [2024-01-10, 2024-01-15) -> 5 days
    assert out.height == 5
    assert out["ts"].min().date() >= date(2024, 1, 10)
    assert out["ts"].max().date() <= date(2024, 1, 14)


def test_rate_series_store_coverage_empty_when_no_data(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    assert store.coverage() == {}


def test_rate_series_store_coverage_per_series(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    df = _obs("SOFR", datetime(2024, 1, 2, tzinfo=UTC), days=10)
    store.write_observations(df)
    cov = store.coverage()
    assert "SOFR" in cov
    first, last = cov["SOFR"]
    assert first.date() == date(2024, 1, 2)
    assert last.date() == date(2024, 1, 11)


def test_rate_series_store_upsert_dedupes_by_series_ts(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    df1 = _obs("SOFR", datetime(2024, 1, 2, tzinfo=UTC), days=5, seed=1)
    store.write_observations(df1)
    # Re-fetch overlapping range with different rates (seed=2) -> keep-latest.
    df2 = _obs("SOFR", datetime(2024, 1, 4, tzinfo=UTC), days=5, seed=2)
    store.upsert_observations(df2)
    out = store.read_observations_eager(series=["SOFR"]).sort("ts")
    # No duplicate (series, ts) rows.
    per_day = out.group_by("ts").len().sort("len", descending=True)
    assert per_day["len"].max() == 1
    # Total rows = 5 (from df1) + 3 new days from df2 (df2 covers 01-04..01-08,
    # overlapping 01-04, 01-05, 01-06 with df1's 01-02..01-06). Unique dates:
    # 01-02, 01-03, 01-04, 01-05, 01-06, 01-07, 01-08 = 7 days.
    assert out.height == 7


def test_rate_series_store_upsert_is_idempotent_on_re_play(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    df = _obs("DGS2", datetime(2024, 1, 2, tzinfo=UTC), days=5, seed=3)
    store.upsert_observations(df)
    store.upsert_observations(df)  # same data again
    out = store.read_observations_eager(series=["DGS2"]).sort("ts")
    assert out.height == 5  # no duplicates


# --- FredRateRepo (no piece of data downloaded twice; CEO 2026-07-19) --- #


class FakeFredFetcher:
    """Records every (series_id, start, end) call and returns synthetic observations.

    Stands in for ``FredClient`` so the no-double-download logic is tested with
    real parquet I/O and zero spend. ``calls`` is the audit trail.
    """

    def __init__(self, obs_factory, *, seed: int = 11) -> None:
        self._obs_factory = obs_factory
        self.calls: list[tuple[str, date, date]] = []
        self._seed = seed

    def fetch_observations(
        self,
        series_id: str,
        start: date | str,
        end: date | str,
    ) -> pl.DataFrame:
        s = start if isinstance(start, date) else date.fromisoformat(start)
        e = end if isinstance(end, date) else date.fromisoformat(end)
        self.calls.append((series_id, s, e))
        start_dt = datetime(s.year, s.month, s.day, tzinfo=UTC)
        end_dt = datetime(e.year, e.month, e.day, tzinfo=UTC)
        if end_dt <= start_dt:
            return pl.DataFrame(schema=RATE_OBSERVATION_SCHEMA)
        days = (end_dt - start_dt).days
        return self._obs_factory(series_id, start_dt, days, seed=self._seed + hash(series_id) % 100)


@pytest.fixture
def fred_repo(tmp_rates_dir, obs_factory) -> tuple[FredRateRepo, FakeFredFetcher]:
    store = RateSeriesStore(tmp_rates_dir)
    fetcher = FakeFredFetcher(obs_factory)
    return FredRateRepo(store, fetcher, bootstrap_start=date(2010, 1, 1)), fetcher


@pytest.fixture
def obs_factory():
    """Factory fixture returning :func:`_obs` for the fake fetcher."""
    return _obs


def test_fred_repo_bootstrap_pulls_max_range_on_first_request(fred_repo):
    repo, fetcher = fred_repo
    end = date(2024, 1, 31)
    repo.get_observations(["SOFR"], start="2024-01-29", end=end).collect()
    # First request bootstraps [bootstrap_start, end) — one fetch call.
    assert fetcher.calls == [("SOFR", date(2010, 1, 1), end)]
    cov = repo.coverage(series=["SOFR"])["SOFR"]
    assert cov[0].date() == date(2010, 1, 1)
    assert cov[1].date() >= end - timedelta(days=1)


def test_fred_repo_incremental_fetch_only_forward_gap(fred_repo):
    repo, fetcher = fred_repo
    # First request: bootstrap [2010-01-01, 2024-01-15).
    repo.get_observations(["SOFR"], start="2024-01-10", end=date(2024, 1, 15)).collect()
    assert len(fetcher.calls) == 1
    last_stored = repo.coverage(series=["SOFR"])["SOFR"][1].date()

    # Second request extending end forward: fetch ONLY [last+1, new_end).
    new_end = last_stored + timedelta(days=10)
    repo.get_observations(["SOFR"], start="2024-01-10", end=new_end).collect()
    assert len(fetcher.calls) == 2
    sid, s, e = fetcher.calls[1]
    assert sid == "SOFR"
    assert s == last_stored + timedelta(days=1), "must fetch only the forward gap"
    assert e == new_end


def test_fred_repo_re_request_cached_range_triggers_zero_fetches(fred_repo):
    repo, fetcher = fred_repo
    end = date(2024, 1, 20)
    repo.get_observations(["SOFR"], start="2024-01-10", end=end).collect()
    assert len(fetcher.calls) == 1
    # Same range again -> pure store read, fetcher NOT called.
    repo.get_observations(["SOFR"], start="2024-01-10", end=end).collect()
    assert len(fetcher.calls) == 1
    # Smaller subrange already cached -> also zero fetches.
    repo.get_observations(["SOFR"], start="2024-01-12", end=date(2024, 1, 18)).collect()
    assert len(fetcher.calls) == 1


def test_fred_repo_overlapping_re_fetch_does_not_duplicate_rows(fred_repo):
    repo, _fetcher = fred_repo
    end = date(2024, 1, 20)
    repo.get_observations(["SOFR"], start="2024-01-10", end=end).collect()
    # Re-fetch an overlapping forward range (end pushed by 3 days).
    new_end = date(2024, 1, 23)
    repo.get_observations(["SOFR"], start="2024-01-10", end=new_end).collect()
    df = repo.store.read_observations_eager(series=["SOFR"])
    per_day = df.group_by("ts").len().sort("len", descending=True)
    assert per_day["len"].max() == 1  # no duplicate (series, ts) rows


def test_fred_repo_front_gap_fetched_once(fred_repo, obs_factory):
    repo, fetcher = fred_repo
    # Pre-populate the store with a NARROW recent range (not a bootstrap) so a
    # front gap genuinely exists.
    narrow = obs_factory("SOFR", datetime(2024, 1, 10, tzinfo=UTC), days=10, seed=21)
    repo.store.upsert_observations(narrow)
    first = repo.coverage(series=["SOFR"])["SOFR"][0].date()
    assert first == date(2024, 1, 10)
    assert len(fetcher.calls) == 0  # pre-populate bypassed the fetcher

    earlier_start = first - timedelta(days=5)
    repo.get_observations(["SOFR"], start=earlier_start, end=date(2024, 1, 20)).collect()
    assert len(fetcher.calls) == 1
    sid, s, e = fetcher.calls[0]
    assert sid == "SOFR"
    assert s == max(earlier_start, repo.bootstrap_start)
    assert e == first  # exclusive end at the store's first day
    # Re-requesting the same earlier start -> zero new fetches (front gap filled).
    repo.get_observations(["SOFR"], start=earlier_start, end=date(2024, 1, 20)).collect()
    assert len(fetcher.calls) == 1


def test_fred_repo_front_gap_capped_at_bootstrap_start(fred_repo, obs_factory):
    repo, fetcher = fred_repo
    narrow = obs_factory("SOFR", datetime(2024, 1, 10, tzinfo=UTC), days=10, seed=31)
    repo.store.upsert_observations(narrow)
    first = repo.coverage(series=["SOFR"])["SOFR"][0].date()
    very_old = date(1990, 1, 1)  # before bootstrap_start (2010-01-01)
    repo.get_observations(["SOFR"], start=very_old, end=date(2024, 1, 20)).collect()
    sid, s, e = fetcher.calls[0]
    assert sid == "SOFR"
    assert s == repo.bootstrap_start  # capped, not 1990-01-01
    assert e == first


def test_fred_repo_missing_ranges_pure_computation_no_fetch(fred_repo):
    repo, fetcher = fred_repo
    repo.get_observations(["SOFR"], start="2024-01-10", end=date(2024, 1, 20)).collect()
    n_before = len(fetcher.calls)
    ranges = repo.missing_ranges(["SOFR"], start="2024-01-25", end=date(2024, 2, 5))
    assert len(fetcher.calls) == n_before  # no fetch happened
    assert len(ranges) == 1
    last_stored = repo.coverage(series=["SOFR"])["SOFR"][1].date()
    assert ranges[0].start == last_stored + timedelta(days=1)
    assert ranges[0].end == date(2024, 2, 5)


def test_fred_repo_missing_ranges_empty_when_fully_cached(fred_repo):
    repo, _fetcher = fred_repo
    repo.get_observations(["SOFR"], start="2024-01-10", end=date(2024, 1, 20)).collect()
    ranges = repo.missing_ranges(["SOFR"], start="2024-01-12", end=date(2024, 1, 18))
    assert ranges == []


def test_fred_repo_missing_ranges_bootstrap_for_new_series(fred_repo):
    repo, _fetcher = fred_repo
    # SOFR is cached; DGS10 is new -> DGS10 bootstraps.
    repo.get_observations(["SOFR"], start="2024-01-10", end=date(2024, 1, 20)).collect()
    ranges = repo.missing_ranges(["SOFR", "DGS10"], start="2024-01-15", end=date(2024, 1, 25))
    by_key = {r.key: r for r in ranges}
    assert "DGS10" in by_key
    assert by_key["DGS10"].start == repo.bootstrap_start
    assert by_key["DGS10"].end == date(2024, 1, 25)


def test_fred_repo_unbounded_end_rejected(fred_repo):
    repo, _fetcher = fred_repo
    with pytest.raises(ValueError, match="`end`"):
        repo.missing_ranges(["SOFR"], start="2024-01-10", end=None)  # type: ignore[arg-type]


def test_fred_repo_get_observations_returns_lazyframe(fred_repo):
    repo, _fetcher = fred_repo
    lazy = repo.get_observations(["SOFR"], start="2024-01-10", end=date(2024, 1, 20))
    assert isinstance(lazy, pl.LazyFrame)


def test_fred_repo_multiple_series_bootstrap_independently(fred_repo):
    repo, fetcher = fred_repo
    # Request three series at once — each bootstraps independently on first call.
    repo.get_observations(
        ["SOFR", "DGS10", "DFII10"], start="2024-01-10", end=date(2024, 1, 20)
    ).collect()
    assert len(fetcher.calls) == 3
    sids = {c[0] for c in fetcher.calls}
    assert sids == {"SOFR", "DGS10", "DFII10"}
    # All bootstrapped from bootstrap_start.
    for _sid, s, _e in fetcher.calls:
        assert s == date(2010, 1, 1)


# --- FredRatesProvider (builds RatesCurve from the FRED cache) --- #


def _populate_curve_observations(store: RateSeriesStore, as_of: date) -> None:
    """Seed the store with one observation per catalog series on as_of."""
    rows: list[dict[str, Any]] = []
    ts = datetime(as_of.year, as_of.month, as_of.day, tzinfo=UTC)
    # Distinct rates per kind so the provider test can assert routing.
    # SOFR (overnight short rate).
    rows.append({"series": FRED_SOFR_SERIES, "ts": ts, "rate": 0.044, "source": "FRED"})
    # OIS proxy (Treasury CMT) — small tenor drift so tenors are distinguishable.
    for tenor, sid in FRED_OIS_SERIES.items():
        rows.append({"series": sid, "ts": ts, "rate": 0.045 + tenor * 0.0001, "source": "FRED"})
    # Real rates (TIPS) — lower than nominal.
    for tenor, sid in FRED_REAL_SERIES.items():
        rows.append({"series": sid, "ts": ts, "rate": 0.015 + tenor * 0.0001, "source": "FRED"})
    df = pl.DataFrame(rows, schema=RATE_OBSERVATION_SCHEMA)
    store.upsert_observations(df)


def test_fred_rates_provider_builds_curve_from_cache(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    as_of = date(2024, 1, 31)
    _populate_curve_observations(store, as_of)
    provider = FredRatesProvider(store)
    curve = provider.get_curve(as_of)
    assert isinstance(curve, RatesCurve)
    assert curve.as_of == as_of
    assert curve.tenors_years == CURVE_TENORS_YEARS
    # OIS rates populated at every tenor (DGSx covers 0.25..30y).
    assert all(r is not None and not math.isnan(r) for r in curve.ois_rates)
    # SOFR flat across tenors (overnight short rate).
    assert all(r == curve.sofr_rates[0] for r in curve.sofr_rates)
    # Real rates: 5/10/30y populated; other tenors are nan.
    real_by_tenor = dict(zip(curve.tenors_years, curve.real_rates, strict=True))
    assert not math.isnan(real_by_tenor[5.0])
    assert not math.isnan(real_by_tenor[10.0])
    assert not math.isnan(real_by_tenor[30.0])
    assert math.isnan(real_by_tenor[0.25])
    assert math.isnan(real_by_tenor[2.0])
    # OIS slope 2y10y is computable (both tenors populated).
    assert curve.ois_slope_2y10y is not None


def test_fred_rates_provider_uses_latest_on_or_before_as_of(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    # Plant an old DGS10 observation and a newer one; provider must use the newer.
    old = _obs("DGS10", datetime(2024, 1, 2, tzinfo=UTC), days=5, seed=41)
    new = _obs("DGS10", datetime(2024, 1, 25, tzinfo=UTC), days=5, seed=42)
    store.upsert_observations(old)
    store.upsert_observations(new)
    provider = FredRatesProvider(store)
    # as_of after the newest observation -> uses the newest (2024-01-29).
    curve = provider.get_curve(date(2024, 2, 15))
    tenor_to_idx = {t: i for i, t in enumerate(curve.tenors_years)}
    dgs10_rate = curve.ois_rates[tenor_to_idx[10.0]]
    # The newest DGS10 observation (seed=42, start 2024-01-25) should be used.
    newest = new.filter(pl.col("ts").dt.date() <= date(2024, 2, 15)).sort("ts")["rate"][-1]
    assert dgs10_rate == pytest.approx(float(newest))


def test_fred_rates_provider_falls_back_when_ois_missing(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    # Empty store -> all OIS tenors fall back to DEFAULT_RISK_FREE_RATE.
    provider = FredRatesProvider(store)
    curve = provider.get_curve(date(2024, 1, 31))
    assert all(r == DEFAULT_RISK_FREE_RATE for r in curve.ois_rates)
    # SOFR also falls back to the OIS 0.25y value (which is the default).
    assert curve.sofr_rates[0] == DEFAULT_RISK_FREE_RATE


def test_fred_rates_provider_sofr_falls_back_to_ois_when_sofr_missing(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    # Plant only DGS3MO (the 0.25y OIS proxy); no SOFR.
    dgs3mo = _obs("DGS3MO", datetime(2024, 1, 29, tzinfo=UTC), days=2, seed=51, rate_start=0.043)
    store.upsert_observations(dgs3mo)
    provider = FredRatesProvider(store)
    curve = provider.get_curve(date(2024, 1, 30))
    # SOFR falls back to ois_rates[0] (the 0.25y tenor = DGS3MO latest).
    latest_dgs3mo = float(dgs3mo.sort("ts")["rate"][-1])
    assert curve.sofr_rates[0] == pytest.approx(latest_dgs3mo)


def test_fred_rates_provider_real_rates_nan_at_tenors_without_tips(tmp_rates_dir):
    store = RateSeriesStore(tmp_rates_dir)
    as_of = date(2024, 1, 31)
    _populate_curve_observations(store, as_of)
    provider = FredRatesProvider(store)
    curve = provider.get_curve(as_of)
    # Tenors 0.25, 0.5, 1.0, 2.0, 20.0 have no TIPS series -> nan.
    real_by_tenor = dict(zip(curve.tenors_years, curve.real_rates, strict=True))
    for t in [0.25, 0.5, 1.0, 2.0, 20.0]:
        assert math.isnan(real_by_tenor[t]), f"tenor {t} should be nan"
    # 5/10/30 have TIPS -> populated.
    for t in [5.0, 10.0, 30.0]:
        assert not math.isnan(real_by_tenor[t]), f"tenor {t} should be populated"


def test_fred_rates_provider_covers_full_catalog_series(tmp_rates_dir):
    """The provider reads every series in the FRED catalog when building the curve."""
    store = RateSeriesStore(tmp_rates_dir)
    as_of = date(2024, 1, 31)
    _populate_curve_observations(store, as_of)
    # Every catalog series is in the store.
    cov = store.coverage()
    assert set(cov.keys()) == set(ALL_FRED_SERIES_IDS)
    provider = FredRatesProvider(store)
    curve = provider.get_curve(as_of)
    # Spot-check: DGS2 and DGS10 (the FIRF 2y/10y slope inputs) are populated.
    tenor_to_idx = {t: i for i, t in enumerate(curve.tenors_years)}
    assert curve.ois_rates[tenor_to_idx[2.0]] is not None
    assert curve.ois_rates[tenor_to_idx[10.0]] is not None
    assert curve.ois_slope_2y10y is not None
