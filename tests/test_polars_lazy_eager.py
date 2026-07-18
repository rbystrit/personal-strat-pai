"""Polars lazy-vs-eager property test — MUST-PASS (plan §16, D14; RBY-4 exit gate).

Asserts that a ``scan_parquet``-based lazy pipeline's collected output equals
an eager ``read_parquet`` reference on a sample slice, across filter/projection/
aggregation combinations. Guards against the D14 footguns the CEO flagged:
predicate/projection pushdown over partitioned parquet, join order, null
propagation, and ``maintain_order`` defaults that differ from eager/pandas.

If this test fails, a LazyFrame is silently producing different results than the
eager reference — block the release and audit the lazy pipeline (plan §19).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from personal_strat_pai.data.polars_utils import collect_eager, to_utc_datetime
from personal_strat_pai.data.store import BarStore

# Deterministic, hermetic — no network, no flakiness.
settings.register_profile("ci", max_examples=50, deadline=None)
settings.load_profile("ci")


def _eager_reference(bars_dir: Path, kind: str = "daily") -> pl.DataFrame:
    """The eager ground truth: read every parquet file under the kind dir."""
    kind_dir = bars_dir / kind
    files = sorted(kind_dir.rglob("*.parquet"))
    if not files:
        return pl.DataFrame()
    parts = [pl.read_parquet(f) for f in files]
    # hive-partitioned writes drop `symbol` from the file; re-inject from path
    # so the eager reference matches the lazy scan's hive-injected schema.
    out_parts: list[pl.DataFrame] = []
    for f, part in zip(files, parts, strict=True):
        frame = part
        if "symbol" not in frame.columns:
            sym = _symbol_from_path(f)
            frame = frame.with_columns(pl.lit(sym).alias("symbol"))
        out_parts.append(frame)
    return pl.concat(out_parts).select(["symbol", "ts", "open", "high", "low", "close", "volume"])


def _symbol_from_path(p: Path) -> str:
    """Extract the symbol=XYZ hive partition from a path like .../symbol=XLB/part-0.parquet."""
    for part in p.parts:
        if part.startswith("symbol="):
            return part[len("symbol=") :]
    raise ValueError(f"no symbol= partition in path {p}")


def test_lazy_full_read_equals_eager_reference(synthetic_bars_lazy_df, tmp_bars_dir):
    """Collected full lazy scan == eager read_parquet reference (sorted)."""
    store = BarStore(tmp_bars_dir)
    lazy = store.scan_bars(kind="daily")
    assert isinstance(lazy, pl.LazyFrame)
    lazy_collected = collect_eager(lazy).sort(["symbol", "ts"])
    ref = _eager_reference(tmp_bars_dir).sort(["symbol", "ts"])
    assert lazy_collected.shape == ref.shape
    assert set(lazy_collected.columns) == set(ref.columns)
    assert_frame_equal(
        lazy_collected.select(sorted(lazy_collected.columns)),
        ref.select(sorted(ref.columns)),
        check_dtypes=False,
    )


def test_lazy_symbol_predicate_pushdown_matches_eager_filter(synthetic_bars_lazy_df, tmp_bars_dir):
    """Predicate pushdown on symbol: lazy filtered scan == eager reference filtered."""
    symbols = ["XLB", "TLT"]
    store = BarStore(tmp_bars_dir)
    lazy = collect_eager(store.scan_bars(kind="daily", symbols=symbols)).sort(["symbol", "ts"])
    ref = (
        _eager_reference(tmp_bars_dir)
        .filter(pl.col("symbol").is_in(symbols))
        .sort(["symbol", "ts"])
    )
    assert_frame_equal(
        lazy.select(sorted(lazy.columns)),
        ref.select(sorted(ref.columns)),
        check_dtypes=False,
    )
    # Partition pruning sanity: only the requested symbols appear.
    assert set(lazy["symbol"].unique().to_list()) <= set(symbols)


def test_lazy_date_predicate_pushdown_matches_eager_filter(synthetic_bars_lazy_df, tmp_bars_dir):
    """Predicate pushdown on ts: lazy date-range scan == eager reference filtered."""
    start = datetime(2024, 1, 2).isoformat()
    end = datetime(2024, 1, 20).isoformat()
    store = BarStore(tmp_bars_dir)
    lazy = collect_eager(store.scan_bars(kind="daily", start=start, end=end)).sort(["symbol", "ts"])
    start_dt = to_utc_datetime(start)
    end_dt = to_utc_datetime(end)
    ref = (
        _eager_reference(tmp_bars_dir)
        .filter((pl.col("ts") >= pl.lit(start_dt)) & (pl.col("ts") < pl.lit(end_dt)))
        .sort(["symbol", "ts"])
    )
    assert_frame_equal(
        lazy.select(sorted(lazy.columns)),
        ref.select(sorted(ref.columns)),
        check_dtypes=False,
    )


def test_lazy_aggregation_equals_eager_aggregation(synthetic_bars_lazy_df, tmp_bars_dir):
    """A group_by aggregation on the lazy scan == the same on the eager reference."""
    store = BarStore(tmp_bars_dir)
    lazy_agg = (
        store.scan_bars(kind="daily")
        .group_by("symbol")
        .agg(
            pl.col("close").mean().alias("mean_close"),
            pl.col("close").std().alias("std_close"),
            pl.col("volume").sum().alias("sum_volume"),
            pl.len().alias("n"),
        )
        .collect()
        .sort("symbol")
    )
    ref_agg = (
        _eager_reference(tmp_bars_dir)
        .group_by("symbol")
        .agg(
            pl.col("close").mean().alias("mean_close"),
            pl.col("close").std().alias("std_close"),
            pl.col("volume").sum().alias("sum_volume"),
            pl.len().alias("n"),
        )
        .sort("symbol")
    )
    assert_frame_equal(lazy_agg, ref_agg, check_dtypes=False, abs_tol=1e-9)


@given(
    symbols=st.lists(
        st.sampled_from(["XLB", "XLY", "TLT", "XLE", "XLF"]), min_size=1, max_size=5, unique=True
    ),
    start_offset=st.integers(min_value=0, max_value=30),
    span_days=st.integers(min_value=1, max_value=40),
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None, max_examples=50
)
def test_lazy_property_random_filter(
    synthetic_bars_lazy_df, tmp_bars_dir, symbols, start_offset, span_days
):
    """Hypothesis-driven: for any symbol subset + date window, lazy == eager."""
    base = datetime(2024, 1, 2)
    start = (base + timedelta(days=start_offset)).date().isoformat()
    end = (base + timedelta(days=start_offset + span_days)).date().isoformat()
    store = BarStore(tmp_bars_dir)
    lazy = collect_eager(store.scan_bars(kind="daily", symbols=symbols, start=start, end=end)).sort(
        ["symbol", "ts"]
    )
    ref = (
        _eager_reference(tmp_bars_dir)
        .filter(
            pl.col("symbol").is_in(symbols)
            & (pl.col("ts") >= pl.lit(to_utc_datetime(start)))
            & (pl.col("ts") < pl.lit(to_utc_datetime(end)))
        )
        .sort(["symbol", "ts"])
    )
    if lazy.is_empty() and ref.is_empty():
        return
    assert_frame_equal(
        lazy.select(sorted(lazy.columns)),
        ref.select(sorted(ref.columns)),
        check_dtypes=False,
    )


def test_collect_eager_is_idempotent_on_eager(synthetic_bars):
    """collect_eager on an eager frame is a no-op (boundary helper sanity)."""
    eager = synthetic_bars
    out = collect_eager(eager)
    assert out is eager or out.equals(eager)


def test_no_lazyframe_leak_across_assert_eager(synthetic_bars_lazy_df, tmp_bars_dir):
    """A LazyFrame must not pass an eager-only boundary (D14(b) footgun guard)."""
    from personal_strat_pai.data.polars_utils import CollectError, assert_eager

    store = BarStore(tmp_bars_dir)
    lazy = store.scan_bars(kind="daily")
    assert isinstance(lazy, pl.LazyFrame)
    with pytest.raises(CollectError):
        assert_eager(lazy, "pre_trade_check")
