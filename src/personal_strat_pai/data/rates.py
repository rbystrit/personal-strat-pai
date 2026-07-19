"""SOFR / OIS swap curve + real rates — FRED primary for history, databento for live (D3).

The Fixed-Income Regime Filter (FIRF, plan §6.3) monitors the slope of the SOFR
Forward Curve (2y vs 10y OIS swap spread) alongside real rates. If the curve
aggressively flattens or inverts while real rates surge, a macro circuit breaker
caps growth/duration buckets (Tech, Real Estate, Crypto) at 20% NAV.

Sources (CEO directive 2026-07-19, RBY-4 rejection 2 + "For FRED, follow the
same download once policy"):

  * **Historical backtest corpus** — **FRED** (free, https://api.stlouisfed.org).
    SOFR (overnight) + Treasury CMT yields as the OIS proxy + TIPS yields as
    real rates. Every observation is downloaded ONCE, cached in the parquet
    store (Object Storage in prod), and re-read on subsequent requests — the
    same no-double-download policy as bars (CEO 2026-07-18). See
    ``FredRateRepo`` / ``FredRatesProvider`` / ``data/fred.py``.
  * **Live forward curve** — **databento** (SOFR/OIS forward curve, plan §6.3
    D3). Remains the system of record once creds land (P0-3); integration-gated
    behind ``DATABENTO_API_KEY``. ``DatabentoRatesProvider`` keeps its stub.

For P0-1 the ``FredRatesProvider`` is wired but integration-gated (needs
``FRED_API_KEY``); ``SyntheticRatesProvider`` is a deterministic provider for
tests/dev so the IV proxy and FIRF can run without spend. The caching LOGIC
(``FredRateRepo``) is real and tested on ``file://`` with synthetic data — only
the live FRED calls + OCI Object Storage backing land with creds (P0-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from math import nan
from typing import Protocol

import polars as pl

from personal_strat_pai.data.caching import (
    BOOTSTRAP_START,
    FetchRange,
    compute_fetch_ranges,
    to_date,
)
from personal_strat_pai.data.fred import (
    ALL_FRED_SERIES_IDS,
    FRED_OIS_SERIES,
    FRED_REAL_SERIES,
    FRED_SOFR_SERIES,
    FredFetcher,
)
from personal_strat_pai.data.polars_utils import assert_eager
from personal_strat_pai.data.store import Coverage, RateSeriesStore

__all__ = [
    "DEFAULT_RISK_FREE_RATE",
    "DatabentoRatesProvider",
    "FredRateRepo",
    "FredRatesProvider",
    "RatesCurve",
    "RatesProvider",
    "SyntheticRatesProvider",
    "risk_free_continuous",
]

# Conservative default used when no curve is available (IV proxy, FIRF tests).
# ~5% continuous — a reasonable recent-regime placeholder; overridden by the live curve.
DEFAULT_RISK_FREE_RATE: float = 0.05

# Curve tenors (years) — matches FRED OIS proxy coverage (3M..30Y).
# SOFR (overnight) fills the short rate; OIS proxy (DGSx) fills each tenor;
# real rates (DFIIx) fill 5/10/30y and are `nan` elsewhere.
CURVE_TENORS_YEARS: list[float] = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]


@dataclass(frozen=True, slots=True)
class RatesCurve:
    """SOFR/OIS forward curve snapshot (plan §6.3)."""

    as_of: date
    tenors_years: list[float] = field(default_factory=list)  # e.g. [0.25, 0.5, 1, 2, 5, 10, 20]
    ois_rates: list[float] = field(default_factory=list)  # continuous annualized rates
    sofr_rates: list[float] = field(default_factory=list)  # continuous annualized rates
    real_rates: list[float] = field(default_factory=list)  # real rates per tenor (optional)

    @property
    def ois_slope_2y10y(self) -> float | None:
        """2y-10y OIS swap spread (plan §6.3 FIRF trigger). Positive = normal curve."""
        r2 = self._rate_at(2.0, self.ois_rates)
        r10 = self._rate_at(10.0, self.ois_rates)
        if r2 is None or r10 is None:
            return None
        return r10 - r2

    def _rate_at(self, tenor: float, rates: list[float]) -> float | None:
        if not self.tenors_years or not rates:
            return None
        try:
            idx = self.tenors_years.index(tenor)
        except ValueError:
            # linear interpolation between bracketing tenors
            idx_lo = max(i for i, t in enumerate(self.tenors_years) if t <= tenor)
            idx_hi = min(i for i, t in enumerate(self.tenors_years) if t >= tenor)
            if idx_lo == idx_hi:
                return rates[idx_lo]
            t_lo, t_hi = self.tenors_years[idx_lo], self.tenors_years[idx_hi]
            r_lo, r_hi = rates[idx_lo], rates[idx_hi]
            if t_hi == t_lo:
                return r_lo
            return r_lo + (r_hi - r_lo) * (tenor - t_lo) / (t_hi - t_lo)
        return rates[idx]


class RatesProvider(Protocol):
    """Rates source protocol (plan §6.3, D3)."""

    def get_curve(self, as_of: date) -> RatesCurve: ...


class SyntheticRatesProvider:
    """Deterministic rates provider for tests/dev (no databento spend)."""

    def __init__(
        self,
        ois: list[float] | None = None,
        sofr: list[float] | None = None,
        tenors: list[float] | None = None,
    ) -> None:
        self.tenors = tenors if tenors is not None else [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        self.ois = ois if ois is not None else [0.045, 0.045, 0.046, 0.045, 0.047, 0.050, 0.052]
        self.sofr = sofr if sofr is not None else [0.045, 0.045, 0.046, 0.045, 0.047, 0.050, 0.052]

    def get_curve(self, as_of: date) -> RatesCurve:
        return RatesCurve(
            as_of=as_of,
            tenors_years=list(self.tenors),
            ois_rates=list(self.ois),
            sofr_rates=list(self.sofr),
        )


class DatabentoRatesProvider:
    """databento SOFR/OIS provider (plan §6.3, D3). Integration-gated (needs API key)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def get_curve(self, as_of: date) -> RatesCurve:  # pragma: no cover - integration
        if not self.api_key:
            raise RuntimeError(
                "DatabentoRatesProvider needs DATABENTO_API_KEY (plan §6.3). "
                "Use SyntheticRatesProvider for tests/dev."
            )
        raise NotImplementedError(
            "databento SOFR/OIS timeseries fetch is wired in P0-1's integration test "
            "path; the production loader lands with the live ingest job. Use "
            "SyntheticRatesProvider until creds are set."
        )


def risk_free_continuous(provider: RatesProvider, as_of: date, tenor_years: float = 1.0) -> float:
    """The continuous risk-free rate used in Black-Scholes (IV proxy) / FIRF.

    Falls back to DEFAULT_RISK_FREE_RATE if the curve is unavailable.
    """
    curve = provider.get_curve(as_of)
    r = curve._rate_at(tenor_years, curve.ois_rates)
    return r if r is not None else DEFAULT_RISK_FREE_RATE


# --- FRED caching repo (no piece of data downloaded twice; CEO 2026-07-19) --- #
class FredRateRepo:
    """Caching FRED rate repo — no piece of data downloaded twice (CEO 2026-07-19).

    Composes a ``RateSeriesStore`` (the cache / Object Storage backing) with a
    ``FredFetcher`` (FRED primary; fake in tests). ``get_observations`` returns
    a lazy scan over the requested range, fetching only the missing pieces
    first. Re-requesting an already-cached range is a pure store read — the
    fetcher is never called.

    Same no-double-download pattern as ``BarRepo`` (data/repo.py); both use the
    shared ``compute_fetch_ranges`` (data/caching.py):

      * **First time** (store empty for a series): bootstrap with the max
        historical range available — ``[bootstrap_start, end)``. FRED returns
        whatever history each series has (SOFR from 2018-04, DGS from 1962,
        TIPS from ~1997); ``bootstrap_start`` defaults to 2000-01-01.
      * **Not the first time**: fetch only ``[last_stored + 1, end)`` — the
        forward gap from the latest day in storage up to the requested date.
        A defensive front-gap fill (``[start, first_stored)``) runs when the
        caller asks for a start older than the store's first day.
      * **Re-request**: zero fetcher calls (pure store read).
    """

    def __init__(
        self,
        store: RateSeriesStore,
        fetcher: FredFetcher,
        *,
        bootstrap_start: date = BOOTSTRAP_START,
    ) -> None:
        self.store = store
        self.fetcher = fetcher
        self.bootstrap_start = bootstrap_start

    def coverage(self, *, series: list[str] | None = None) -> dict[str, Coverage]:
        """Per-series ``(first_ts, last_ts)`` currently in the store."""
        return self.store.coverage(series=series)

    def missing_ranges(
        self,
        series_ids: list[str],
        start: str | date | None,
        end: str | date | None,
    ) -> list[FetchRange]:
        """The ranges the fetcher would be called for — pure computation, no fetch.

        Exposed for observability and tests. ``end`` is required (unbounded
        fetches are rejected).
        """
        if end is None:
            raise ValueError("FredRateRepo requires an `end` bound (no unbounded fetches).")
        cov = self.store.coverage(series=series_ids)
        return compute_fetch_ranges(
            series_ids,
            to_date(start) if start is not None else None,
            to_date(end),
            cov,
            bootstrap_start=self.bootstrap_start,
            step=timedelta(days=1),
        )

    def get_observations(
        self,
        series_ids: list[str],
        start: str | date | None,
        end: str | date,
    ) -> pl.LazyFrame:
        """Return a lazy scan of ``[start, end)`` for ``series_ids``, fetching only gaps.

        1. Compute the missing ranges vs the store's coverage.
        2. For each disjoint range, call the fetcher and upsert into the store
           (idempotent — overlapping re-fetches dedupe by ``(series, ts)``).
        3. Scan the (now-complete) store over the requested range and return it.

        The caller MUST ``collect_eager()`` at a strategy boundary (D14(b)).
        """
        if end is None:
            raise ValueError("FredRateRepo requires an `end` bound (no unbounded fetches).")
        start_date = to_date(start) if start is not None else None
        end_date = to_date(end)
        ranges = compute_fetch_ranges(
            series_ids,
            start_date,
            end_date,
            self.store.coverage(series=series_ids),
            bootstrap_start=self.bootstrap_start,
            step=timedelta(days=1),
        )
        for r in ranges:
            chunk = self.fetcher.fetch_observations(r.key, r.start, r.end)
            if chunk.is_empty():
                continue
            assert_eager(chunk, "FredFetcher.fetch_observations")
            self.store.upsert_observations(chunk)
        return self.store.scan_observations(series=series_ids, start=start, end=end)


class FredRatesProvider:
    """RatesProvider backed by the FRED cache (CEO 2026-07-19).

    Reads the latest observation for each FRED series on / before ``as_of`` from
    the ``RateSeriesStore`` and builds a ``RatesCurve``. Tenor mapping:

      * ``sofr_rates`` — overnight SOFR (``SOFR`` series) replicated across
        tenors (the short rate is flat; FIRF reads the slope from ``ois_rates``).
      * ``ois_rates`` — Treasury CMT yield (``DGSx``) at each matching tenor.
      * ``real_rates`` — TIPS yield (``DFIIx``) at 5/10/30y; ``nan`` elsewhere
        (those tenors have no TIPS series). The FIRF real-rates-surge trigger
        reads the 10y real rate (``DFII10``), which is always populated when
        the cache covers the as_of date.

    For backtesting, this is the primary rates provider. For live trading,
    ``DatabentoRatesProvider`` (forward curve) takes over once creds land
    (P0-3). Falls back to ``DEFAULT_RISK_FREE_RATE`` for any OIS tenor with no
    cached observation (e.g. as_of before the series inception).
    """

    def __init__(self, store: RateSeriesStore) -> None:
        self.store = store

    def get_curve(self, as_of: date) -> RatesCurve:
        as_of_end = as_of + timedelta(days=1)  # exclusive end => include as_of
        # Load only the series we need for the curve, latest observation per
        # series on/before as_of. Eager at the strategy boundary (D14(b)).
        df = self.store.read_observations_eager(start=None, end=as_of_end)
        latest = _latest_per_series_on_or_before(df, as_of)
        sofr = latest.get(FRED_SOFR_SERIES)
        # OIS proxy at each curve tenor; fall back to the default when a tenor
        # has no cached observation (e.g. as_of before the series inception).
        ois_rates: list[float] = []
        for tenor in CURVE_TENORS_YEARS:
            sid = FRED_OIS_SERIES.get(tenor)
            v = latest.get(sid) if sid is not None else None
            ois_rates.append(v if v is not None else DEFAULT_RISK_FREE_RATE)
        # SOFR flat across tenors (overnight short rate); fall back to the 0.25y
        # OIS proxy if SOFR has no observation (e.g. as_of before 2018-04-09).
        sofr_val = sofr if sofr is not None else ois_rates[0]
        sofr_rates: list[float] = [sofr_val for _ in CURVE_TENORS_YEARS]
        # Real rates: nan at tenors without a TIPS series; the FIRF reads the
        # 10y real rate (always populated when the cache covers as_of).
        real_rates: list[float] = []
        for tenor in CURVE_TENORS_YEARS:
            sid = FRED_REAL_SERIES.get(tenor)
            v = latest.get(sid) if sid is not None else None
            real_rates.append(v if v is not None else nan)
        return RatesCurve(
            as_of=as_of,
            tenors_years=list(CURVE_TENORS_YEARS),
            ois_rates=ois_rates,
            sofr_rates=sofr_rates,
            real_rates=real_rates,
        )


def _latest_per_series_on_or_before(df: pl.DataFrame, as_of: date) -> dict[str, float | None]:
    """Per-series latest observation rate on or before ``as_of`` (None if no data)."""
    if df.is_empty():
        return {sid: None for sid in ALL_FRED_SERIES_IDS}
    sub = df.filter(pl.col("ts").dt.date() <= as_of)
    if sub.is_empty():
        return {sid: None for sid in ALL_FRED_SERIES_IDS}
    latest = (
        sub.sort("ts")
        .group_by("series")
        .agg(pl.col("rate").last(), pl.col("ts").max().alias("last_ts"))
    )
    out: dict[str, float | None] = {sid: None for sid in ALL_FRED_SERIES_IDS}
    for row in latest.iter_rows(named=True):
        out[row["series"]] = float(row["rate"]) if row["rate"] is not None else None
    return out
