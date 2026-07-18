"""Shared test fixtures.

Synthetic data only — no live databento/yfinance/OCI calls in CI (plan §16).
Integration tests (live feeds) are opt-in via ``-m integration``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

# Default: never hit live feeds. Individual integration tests override.
os.environ.setdefault("DATABENTO_API_KEY", "")


@pytest.fixture
def tmp_bars_dir(tmp_path: Path) -> Path:
    """A temp directory used as a parquet bar store base."""
    d = tmp_path / "bars"
    d.mkdir()
    return d


def _synthetic_bars(
    symbols: list[str],
    start: datetime,
    days: int,
    *,
    seed: int = 42,
    start_price: float = 100.0,
) -> pl.DataFrame:
    """Deterministic synthetic OHLCV daily bars conforming to BAR_SCHEMA."""
    import random

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        price = start_price + (rng.random() * 20.0)
        for d in range(days):
            ts = start + timedelta(days=d)
            # realistic-ish daily move
            ret = rng.gauss(0.0, 0.012)
            open_ = price
            close = max(round(open_ * (1.0 + ret), 4), 1e-3)
            high = max(open_, close) * (1.0 + abs(rng.gauss(0.0, 0.004)))
            low = min(open_, close) * (1.0 - abs(rng.gauss(0.0, 0.004)))
            vol = max(int(rng.gauss(1_000_000, 200_000)), 1)
            rows.append(
                {
                    "symbol": sym,
                    "ts": ts,
                    "open": float(open_),
                    "high": float(round(high, 4)),
                    "low": float(round(low, 4)),
                    "close": float(close),
                    "volume": vol,
                }
            )
            price = close
    return pl.DataFrame(rows, schema=None)  # schema inferred then validated below


@pytest.fixture
def synthetic_bars() -> pl.DataFrame:
    """~3 symbols × 10 days of clean daily bars (UTC ts)."""
    from personal_strat_pai.data.polars_utils import BAR_SCHEMA

    start = datetime(2024, 1, 2, tzinfo=UTC)
    df = _synthetic_bars(["XLB", "XLY", "TLT"], start, days=10, seed=7)
    return df.cast({c: t for c, t in BAR_SCHEMA.items() if c in df.columns})


@pytest.fixture
def synthetic_bars_lazy_df(tmp_bars_dir: Path) -> pl.DataFrame:
    """Bars written to parquet (partitioned by symbol) — for lazy/eager property tests."""
    from personal_strat_pai.data.store import BarStore

    start = datetime(2024, 1, 2, tzinfo=UTC)
    df = _synthetic_bars(["XLB", "XLY", "TLT", "XLE", "XLF"], start, days=40, seed=11)
    store = BarStore(tmp_bars_dir)
    store.write_bars(df, kind="daily")
    return df


def make_synthetic_bars_range(
    symbols: list[str],
    start: datetime,
    end: datetime,
    *,
    seed: int = 42,
    start_price: float = 100.0,
) -> pl.DataFrame:
    """Synthetic OHLCV daily bars over ``[start, end)`` (date-wise, UTC).

    Used by the caching-repo tests: a deterministic bar generator standing in
    for a live ``BarFetcher`` so the no-double-download logic is exercised with
    real parquet I/O in CI (no databento spend).
    """
    from personal_strat_pai.data.polars_utils import BAR_SCHEMA

    if end <= start:
        return pl.DataFrame(schema=BAR_SCHEMA)
    days = (end - start).days
    df = _synthetic_bars(symbols, start, days, seed=seed, start_price=start_price)
    return df.cast({c: t for c, t in BAR_SCHEMA.items() if c in df.columns})


@pytest.fixture
def bars_range_factory():
    """Factory fixture returning :func:`make_synthetic_bars_range`."""
    return make_synthetic_bars_range
