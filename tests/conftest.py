"""Shared test fixtures.

Synthetic data only — no live databento/yfinance/OCI calls in CI (plan §16).
Integration tests (live feeds) are opt-in via ``-m integration``.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
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


def make_synthetic_option_chain(
    symbol: str,
    spot: float,
    as_of: date,
    *,
    iv_30d: float = 0.25,
    iv_90d: float = 0.22,
    r: float = 0.05,
    n_strikes: int = 11,
    strike_step: float = 5.0,
) -> pl.DataFrame:
    """Build a synthetic EOD option chain priced at known IV (for IV-proxy tests).

    The 30D expiry options are priced at iv_30d; 90D at iv_90d. OTM puts have
    extra skew (so put_skew_30d > 0). implied_vol should recover iv_30d/iv_90d.
    """
    from personal_strat_pai.data.iv_proxy import black_scholes_price

    rows: list[dict[str, Any]] = []
    half = n_strikes // 2
    for expiry_days, iv in ((30, iv_30d), (90, iv_90d)):
        expiry = as_of + timedelta(days=expiry_days)
        T = expiry_days / 365.0
        atm_strike = round(spot / strike_step) * strike_step
        for i in range(-half, half + 1):
            strike = atm_strike + i * strike_step
            for otype in ("C", "P"):
                # OTM puts get a skew bump (higher IV) to exercise the put-skew path.
                local_iv = iv
                if otype == "P" and strike < spot:
                    local_iv = iv + 0.02 * (1.0 - strike / spot) * 4.0
                price = black_scholes_price(spot, strike, T, r, local_iv, otype)
                # bid/ask around the mid with a tiny spread
                spread = max(price * 0.005, 0.01)
                rows.append(
                    {
                        "symbol": symbol,
                        "expiration": expiry,
                        "strike": float(strike),
                        "type": otype,
                        "bid": round(price - spread / 2, 4),
                        "ask": round(price + spread / 2, 4),
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.String,
            "expiration": pl.Date,
            "strike": pl.Float64,
            "type": pl.String,
            "bid": pl.Float64,
            "ask": pl.Float64,
        },
    )


@pytest.fixture
def option_chain_factory():
    """Factory fixture returning :func:`make_synthetic_option_chain`."""
    return make_synthetic_option_chain
