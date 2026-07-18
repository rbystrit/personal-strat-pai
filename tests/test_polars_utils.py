"""Tests for data/polars_utils.py — the D14 lazy/eager boundary helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from personal_strat_pai.data.polars_utils import (
    BAR_SCHEMA,
    CollectError,
    assert_eager,
    assert_lazy,
    collect_eager,
    scan_parquet,
    to_eager,
    to_lazy,
    write_parquet_eager,
)


def _sample_eager() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["XLB", "XLB"],
            "ts": [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.5, 100.5],
            "close": [101.0, 101.5],
            "volume": [1_000_000, 1_100_000],
        },
        schema=BAR_SCHEMA,
    )


def test_collect_eager_lazy_to_eager():
    lazy = _sample_eager().lazy()
    out = collect_eager(lazy)
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(out, _sample_eager())


def test_collect_eager_idempotent_on_eager():
    df = _sample_eager()
    out = collect_eager(df)
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(out, df)


def test_to_lazy_and_to_eager_roundtrip():
    df = _sample_eager()
    lazy = to_lazy(df)
    assert isinstance(lazy, pl.LazyFrame)
    back = to_eager(lazy)
    assert isinstance(back, pl.DataFrame)
    assert_frame_equal(back, df)


def test_assert_eager_rejects_lazyframe():
    lazy = _sample_eager().lazy()
    with pytest.raises(CollectError):
        assert_eager(lazy, "boundary")


def test_assert_eager_accepts_dataframe():
    assert_eager(_sample_eager(), "boundary")  # no raise


def test_assert_lazy_rejects_dataframe():
    with pytest.raises(CollectError):
        assert_lazy(_sample_eager(), "boundary")


def test_assert_lazy_accepts_lazyframe():
    assert_lazy(_sample_eager().lazy(), "boundary")  # no raise


def test_write_parquet_eager_rejects_lazyframe(tmp_path: Path):
    lazy = _sample_eager().lazy()
    with pytest.raises(CollectError):
        write_parquet_eager(lazy, tmp_path / "out.parquet")


def test_scan_parquet_predicate_pushdown(tmp_path: Path):
    df = pl.DataFrame(
        {
            "symbol": ["XLB"] * 3 + ["XLY"] * 3,
            "ts": [
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
                datetime(2024, 1, 4, tzinfo=UTC),
            ]
            * 2,
            "open": [100.0] * 6,
            "high": [101.0] * 6,
            "low": [99.0] * 6,
            "close": [100.5] * 6,
            "volume": [1_000_000] * 6,
        },
        schema=BAR_SCHEMA,
    )
    df.write_parquet(tmp_path, partition_by=["symbol"])
    lf = scan_parquet(tmp_path, symbols=["XLY"], start="2024-01-03")
    assert isinstance(lf, pl.LazyFrame)
    out = collect_eager(lf).sort("ts")
    assert set(out["symbol"].unique().to_list()) == {"XLY"}
    assert out["ts"].min() >= datetime(2024, 1, 3, tzinfo=UTC)
