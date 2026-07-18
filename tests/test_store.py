"""Tests for data/store.py — parquet I/O + SQLite read cache (RBY-4 exit gate)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from personal_strat_pai.data.polars_utils import BAR_SCHEMA
from personal_strat_pai.data.store import BarStore, OciBackendNotConfigured, SQLiteCache


def _bars(symbols: list[str], days: int = 5) -> pl.DataFrame:
    base = datetime(2024, 1, 2, tzinfo=UTC)
    rows = []
    for sym in symbols:
        for i in range(days):
            rows.append(
                {
                    "symbol": sym,
                    "ts": base.replace(month=1, day=2 + i),
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.5 + i,
                    "volume": 1_000_000 + i,
                }
            )
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def test_oci_backend_not_yet_wired(tmp_path: Path):
    # OciBackendNotConfigured is raised at construction (base_uri resolved in __init__).
    with pytest.raises(OciBackendNotConfigured):
        BarStore("oci://my-namespace/bars")


def test_write_then_scan_roundtrip(tmp_bars_dir: Path):
    store = BarStore(tmp_bars_dir)
    df = _bars(["XLB", "XLY"], days=5)
    written = store.write_bars(df, kind="daily")
    assert len(written) >= 1
    out = store.read_bars_eager(kind="daily").sort(["symbol", "ts"])
    assert out.shape == df.shape
    assert_frame_equal(
        out.select(sorted(out.columns)),
        df.select(sorted(df.columns)),
        check_dtypes=False,
    )


def test_partition_pruning_only_reads_requested_symbols(tmp_bars_dir: Path):
    store = BarStore(tmp_bars_dir)
    df = _bars(["XLB", "XLY", "TLT"], days=3)
    store.write_bars(df, kind="daily")
    out = store.read_bars_eager(kind="daily", symbols=["XLY"])
    assert set(out["symbol"].unique().to_list()) == {"XLY"}


def test_scan_returns_lazyframe(tmp_bars_dir: Path):
    store = BarStore(tmp_bars_dir)
    store.write_bars(_bars(["XLB"], days=3), kind="daily")
    lazy = store.scan_bars(kind="daily")
    assert isinstance(lazy, pl.LazyFrame)


def test_empty_store_scan_returns_empty_with_schema(tmp_path: Path):
    store = BarStore(tmp_path / "empty")
    assert isinstance(store.scan_bars(kind="daily"), pl.LazyFrame)
    out = store.read_bars_eager(kind="daily")
    assert out.is_empty()
    assert list(out.columns) == list(BAR_SCHEMA.names())


def test_schema_validation_rejects_missing_column(tmp_bars_dir: Path):
    store = BarStore(tmp_bars_dir)
    bad = pl.DataFrame({"symbol": ["XLB"], "ts": [datetime(2024, 1, 2, tzinfo=UTC)]})
    with pytest.raises(ValueError, match="missing column"):
        store.write_bars(bad)


def test_sqlite_cache_put_get_roundtrip(tmp_path: Path):
    cache = SQLiteCache(tmp_path / "cache.sqlite3")
    df = _bars(["XLB"], days=5)
    n = cache.put_bars(df)
    assert n == 5
    recent = cache.get_recent("XLB", n_days=3)
    assert recent.height == 3
    assert recent["symbol"].unique().to_list() == ["XLB"]
    # ascending ts order
    assert recent["ts"].is_sorted()


def test_sqlite_cache_get_recent_close(tmp_path: Path):
    cache = SQLiteCache(tmp_path / "cache.sqlite3")
    df = _bars(["XLB"], days=3)
    cache.put_bars(df)
    close = cache.get_recent_close("XLB")
    assert close is not None
    assert close == df["close"].max()


def test_sqlite_cache_missing_symbol_returns_empty(tmp_path: Path):
    cache = SQLiteCache(tmp_path / "cache.sqlite3")
    out = cache.get_recent("NOPE", n_days=5)
    assert out.is_empty()
    assert cache.get_recent_close("NOPE") is None
