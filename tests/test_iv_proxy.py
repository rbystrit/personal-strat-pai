"""Tests for data/iv_proxy.py — HV IV proxy for backtesting; IBKR for live (CEO 2026-07-18).

Supersedes the D2 EOD-chain self-built IV approach. Coverage:
  * compute_hv on known series recovers the annualized realized vol.
  * compute_hv returns None / raises on insufficient history.
  * HvIvProvider builds an IvSnapshot from bars (30d/90d HV, term slope,
    backwardation) and zeroes the put-skew fields (deferred to IBKR live).
  * IbkrIvProvider raises NotImplementedError until the IBKR wiring lands (P0-3).
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from personal_strat_pai.data.iv_proxy import (
    TRADING_DAYS_PER_YEAR,
    HvIvProvider,
    IbkrIvProvider,
    IvSnapshot,
    compute_hv,
    compute_hv_snapshot,
)
from personal_strat_pai.data.polars_utils import BAR_SCHEMA


def _bars_with_daily_vol(
    symbol: str,
    start: datetime,
    days: int,
    *,
    daily_sigma: float,
    start_price: float = 100.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Deterministic bars with a target daily log-return std ``daily_sigma``.

    Uses a fixed-seed RNG to produce Gaussian daily returns with std
    ``daily_sigma``; HV over the trailing window should recover
    ``daily_sigma * sqrt(252)`` within sampling noise for a long enough window.
    """
    import random

    rng = random.Random(seed)
    rows: list[dict[str, float | int | str | datetime]] = []
    price = start_price
    for d in range(days):
        ts = start + timedelta(days=d)
        ret = rng.gauss(0.0, daily_sigma)
        open_ = price
        close = max(round(open_ * math.exp(ret), 4), 1e-3)
        high = max(open_, close) * 1.005
        low = min(open_, close) * 0.995
        rows.append(
            {
                "symbol": symbol,
                "ts": ts,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": 1_000_000,
            }
        )
        price = close
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def test_compute_hv_constant_series_returns_zero():
    # A flat series has zero variance -> HV == 0.0 (not None).
    closes = pl.Series("c", [100.0] * 32)
    hv = compute_hv(closes, window=30)
    assert hv is not None
    assert hv == 0.0


def test_compute_hv_insufficient_history_returns_none():
    closes = pl.Series("c", [100.0, 101.0, 99.0])  # 3 points -> < 30+1
    assert compute_hv(closes, window=30) is None


def test_compute_hv_recovers_annualized_vol_within_tolerance():
    # Plant daily returns with std 0.012; HV over 60d should recover ~0.012*sqrt(252).
    daily_sigma = 0.012
    bars = _bars_with_daily_vol(
        "XLB", datetime(2023, 1, 2, tzinfo=UTC), days=200, daily_sigma=daily_sigma, seed=1
    )
    closes = bars["close"].cast(pl.Float64)
    hv = compute_hv(closes, window=60)
    assert hv is not None
    expected = daily_sigma * math.sqrt(TRADING_DAYS_PER_YEAR)
    # Generous tolerance (sample std over 60 points); order-of-magnitude check.
    assert abs(hv - expected) < 0.05


def test_compute_hv_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        compute_hv(pl.Series("c", [1.0, 2.0]), window=0)


def test_compute_hv_accepts_list_input():
    hv = compute_hv([100.0] * 32, window=30)
    assert hv == 0.0


def test_compute_hv_snapshot_builds_term_structure():
    bars = _bars_with_daily_vol(
        "XLY", datetime(2023, 1, 2, tzinfo=UTC), days=200, daily_sigma=0.012, seed=2
    )
    as_of = (datetime(2023, 1, 2, tzinfo=UTC) + timedelta(days=199)).date()
    snap = compute_hv_snapshot(bars, "XLY", as_of, hv_short=30, hv_long=90)
    assert isinstance(snap, IvSnapshot)
    assert snap.symbol == "XLY"
    assert snap.as_of == as_of
    assert snap.iv_30d > 0.0
    assert snap.iv_90d > 0.0
    assert snap.term_slope == pytest.approx(snap.iv_30d - snap.iv_90d)
    assert snap.backwardation == (snap.iv_30d > snap.iv_90d)
    # Put-skew fields are NOT computable from HV -> deferred to IBKR live.
    assert snap.put_skew_30d == 0.0
    assert snap.put_skew_percentile_12m is None
    assert snap.extreme_put_skew is False


def test_compute_hv_snapshot_raises_on_insufficient_history():
    bars = _bars_with_daily_vol(
        "XLB", datetime(2023, 1, 2, tzinfo=UTC), days=10, daily_sigma=0.01, seed=3
    )
    as_of = (datetime(2023, 1, 2, tzinfo=UTC) + timedelta(days=9)).date()
    with pytest.raises(ValueError, match="insufficient history"):
        compute_hv_snapshot(bars, "XLB", as_of, hv_short=30, hv_long=90)


def test_compute_hv_snapshot_raises_on_empty_bars():
    empty = pl.DataFrame(schema=BAR_SCHEMA)
    with pytest.raises(ValueError, match="no bars"):
        compute_hv_snapshot(empty, "XLB", date(2024, 1, 2))


def test_compute_hv_snapshot_rejects_bad_window_order():
    bars = _bars_with_daily_vol(
        "XLB", datetime(2023, 1, 2, tzinfo=UTC), days=100, daily_sigma=0.01, seed=4
    )
    with pytest.raises(ValueError):
        compute_hv_snapshot(bars, "XLB", date(2023, 7, 1), hv_short=90, hv_long=30)


def test_compute_hv_snapshot_trims_to_as_of():
    # Bars after as_of must be excluded from the HV window.
    start = datetime(2023, 1, 2, tzinfo=UTC)
    bars = _bars_with_daily_vol("XLB", start, days=200, daily_sigma=0.012, seed=5)
    as_of = (start + timedelta(days=99)).date()  # only ~100 days available
    snap = compute_hv_snapshot(bars, "XLB", as_of, hv_short=30, hv_long=60)
    assert snap.iv_30d > 0.0
    assert snap.iv_90d > 0.0  # 60d window fits within 100 days


def test_hv_iv_provider_uses_injected_bar_loader():
    bars = _bars_with_daily_vol(
        "XLB", datetime(2023, 1, 2, tzinfo=UTC), days=200, daily_sigma=0.012, seed=6
    )
    as_of = (datetime(2023, 1, 2, tzinfo=UTC) + timedelta(days=199)).date()

    def loader(sym: str, d: date) -> pl.DataFrame:
        assert sym == "XLB"
        assert d == as_of
        return bars

    provider = HvIvProvider(loader, hv_short=30, hv_long=90)
    snap = provider.get_snapshot("XLB", as_of)
    assert snap.symbol == "XLB"
    assert snap.iv_30d > 0.0
    assert snap.put_skew_percentile_12m is None  # deferred to IBKR live


def test_ibkr_iv_provider_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="IBKR"):
        IbkrIvProvider()  # not wired until P0-3
