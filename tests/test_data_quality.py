"""Tests for data/quality.py — the §6.5 data-quality gate (RBY-4 acceptance)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from personal_strat_pai.data.polars_utils import BAR_SCHEMA
from personal_strat_pai.data.quality import (
    DataQualityError,
    check_monotonic_ts,
    check_no_null_ohlc,
    check_price_bounds,
    check_split_continuity,
    check_volume_nonneg,
    validate_bars,
)


def _bars(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def _clean_bars(n: int = 5, symbol: str = "XLB") -> pl.DataFrame:
    base = datetime(2024, 1, 2, tzinfo=UTC)
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbol": symbol,
                "ts": base + timedelta(days=i),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1_000_000 + i,
            }
        )
    return _bars(rows)


def test_clean_bars_pass_all_checks():
    df = _clean_bars()
    report = validate_bars(df)
    assert report.passed
    assert report.violations == []
    assert report.row_count == 5
    assert report.symbol_count == 1


def test_validate_bars_raises_on_fail_when_requested():
    bad = _clean_bars()
    bad = bad.with_columns(pl.lit(-5).alias("volume")).cast({"volume": pl.Int64})
    with pytest.raises(DataQualityError):
        validate_bars(bad, raise_on_fail=True)


def test_check_monotonic_ts_detects_duplicate():
    df = _clean_bars()
    # duplicate a row's ts for the same symbol
    bad = pl.concat([df, df.head(1)])
    out = check_monotonic_ts(bad.sort(["symbol", "ts"]))
    assert len(out) >= 1
    assert out[0].check == "monotonic_ts"


def test_check_no_null_ohlc_detects_null():
    df = _clean_bars()
    bad = df.with_columns(pl.lit(None).alias("high"))
    out = check_no_null_ohlc(bad.cast({"high": pl.Float64}))
    assert len(out) == 1
    assert out[0].check == "no_null_ohlc"
    assert out[0].symbol == "XLB"


def test_check_volume_nonneg_detects_negative():
    df = _clean_bars()
    bad = df.with_columns(pl.lit(-1).alias("volume")).cast({"volume": pl.Int64})
    out = check_volume_nonneg(bad)
    assert len(out) == 1
    assert out[0].check == "volume_nonneg"


def test_check_price_bounds_detects_low_gt_high():
    df = _clean_bars()
    # low > high on the first row => price_bounds violation
    bad = df.with_columns(
        pl.when(pl.col("ts") == pl.col("ts").min())
        .then(pl.lit(200.0))
        .otherwise(pl.col("low"))
        .alias("low")
    )
    out = check_price_bounds(bad)
    assert len(out) == 1
    assert out[0].check == "price_bounds"


def test_check_split_continuity_detects_huge_gap():
    df = _clean_bars(n=3)
    # close jumps 100% from row 1 to row 2 — no corp action recorded => flagged
    bad = df.with_columns(
        pl.when(pl.col("ts") == df["ts"][1])
        .then(pl.col("close") * 2.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    out = check_split_continuity(bad, gap_threshold=0.50)
    assert len(out) >= 1
    assert out[0].check == "split_continuity"


def test_check_split_continuity_passes_on_small_moves():
    out = check_split_continuity(_clean_bars(n=10), gap_threshold=0.50)
    assert out == []


def test_validate_bars_missing_column_raises():
    bad = pl.DataFrame({"symbol": ["XLB"], "ts": [datetime(2024, 1, 2, tzinfo=UTC)]})
    with pytest.raises(DataQualityError):
        validate_bars(bad)


def test_validate_bars_accepts_lazy_and_collects():
    df = _clean_bars().lazy()
    report = validate_bars(df)
    assert report.passed
