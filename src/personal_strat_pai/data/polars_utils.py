"""Lazy/eager boundary helpers + canonical schemas (plan §5, D14).

Discipline (CEO-flagged as tricky — plan §19):
  (a) LazyFrame + scan_parquet for large multi-symbol historical reads so predicate
      pushdown and projection pushdown skip unneeded rows/columns.
  (b) .collect() at well-defined strategy boundaries — NEVER let a LazyFrame leak
      across a module API or into a pre-trade check (eager there for predictability).
  (c) Be explicit about eager-vs-lazy in type signatures — pl.DataFrame and
      pl.LazyFrame are not interchangeable; silent auto-conversion is a footgun.
  (d) Watch lazy semantics that differ from eager/pandas: filter-pushdown over
      partitioned parquet, join order, null propagation in over() windows,
      maintain_order defaults.
  (e) Small config/state tables stay eager (no benefit, more readable).
  (f) Tests assert the lazy pipeline's collected output equals an eager reference
      on a small slice (tests/test_polars_lazy_eager.py — MUST-PASS).

This module is the single home for the boundary conventions. Other data-layer
modules import from here instead of calling pl.collect / pl.scan_parquet directly
so the boundary stays auditable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, overload

import polars as pl

__all__ = [
    "BAR_COLUMNS",
    "BAR_SCHEMA",
    "RATE_OBSERVATION_COLUMNS",
    "RATE_OBSERVATION_SCHEMA",
    "CollectError",
    "EagerFrame",
    "Frame",
    "LazyFrame",
    "assert_eager",
    "assert_lazy",
    "collect_eager",
    "scan_parquet",
    "to_eager",
    "to_lazy",
    "to_utc_datetime",
    "write_parquet_eager",
]

# --- Type aliases (D14(c): be explicit about eager-vs-lazy) --- #
EagerFrame = pl.DataFrame
LazyFrame = pl.LazyFrame
Frame = pl.DataFrame | pl.LazyFrame


# --- Canonical OHLCV bar schema (plan §6.1) --- #
# The schema every bar batch conforms to before it is written to parquet.
# ts is microsecond datetime (UTC). volume is int64. OHLC are float64.
BAR_SCHEMA: pl.Schema = pl.Schema(
    [
        ("symbol", pl.String),
        ("ts", pl.Datetime("us", "UTC")),
        ("open", pl.Float64),
        ("high", pl.Float64),
        ("low", pl.Float64),
        ("close", pl.Float64),
        ("volume", pl.Int64),
    ]
)
BAR_COLUMNS: tuple[str, ...] = tuple(BAR_SCHEMA.names())


# --- Canonical rate-observation schema (plan §6.3, FRED / databento) --- #
# One row per (series, ts) observation. `series` is the FRED series id (SOFR,
# DGS10, DFII10, ...) or the databento instrument id. `rate` is the continuous
# annualized rate as a DECIMAL (0.045 = 4.5%); FRED publishes percent, the
# normalizer converts at the boundary. `source` is "FRED" / "databento" so the
# origin is auditable after the merge.
RATE_OBSERVATION_SCHEMA: pl.Schema = pl.Schema(
    [
        ("series", pl.String),
        ("ts", pl.Datetime("us", "UTC")),
        ("rate", pl.Float64),
        ("source", pl.String),
    ]
)
RATE_OBSERVATION_COLUMNS: tuple[str, ...] = tuple(RATE_OBSERVATION_SCHEMA.names())


class CollectError(TypeError):
    """Raised when a LazyFrame leaks across a module API boundary (D14(b))."""


@overload
def collect_eager(df: LazyFrame) -> EagerFrame: ...
@overload
def collect_eager(df: EagerFrame) -> EagerFrame: ...
def collect_eager(df: Frame) -> EagerFrame:
    """Explicitly collect a LazyFrame to eager at a strategy-decision boundary.

    Idempotent on eager frames (returns the input unchanged). This is the ONLY
    place that calls ``LazyFrame.collect`` in the data layer — call it at module
    API boundaries so pre-trade checks and sieve decisions see predictable eager
    data (plan §6.1, §10, D14(b)).
    """
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, pl.DataFrame):
        return df
    raise CollectError(f"expected pl.DataFrame | pl.LazyFrame, got {type(df)!r}")


def assert_eager(df: object, name: str = "frame") -> None:
    """Guard a module API boundary against LazyFrame leaks (D14(b)).

    Use at the top of any function that must not accept a LazyFrame (pre-trade
    checks, small config/state tables, anything where deferred work is a footgun).
    """
    if isinstance(df, pl.LazyFrame):
        raise CollectError(
            f"{name}: received a pl.LazyFrame — call collect_eager() at the boundary "
            "first; a LazyFrame must not leak across this API (plan D14(b))."
        )


def assert_lazy(df: object, name: str = "frame") -> None:
    """Guard a boundary that requires a LazyFrame (large historical scans)."""
    if not isinstance(df, pl.LazyFrame):
        raise CollectError(
            f"{name}: expected a pl.LazyFrame for predicate/projection pushdown, "
            f"got {type(df)!r} (plan D14(a))."
        )


def to_lazy(df: Frame) -> LazyFrame:
    """Lift an eager frame to lazy for the large-read path. No-op on LazyFrame."""
    if isinstance(df, pl.LazyFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    raise CollectError(f"expected pl.DataFrame | pl.LazyFrame, got {type(df)!r}")


def to_eager(df: Frame) -> EagerFrame:
    """Alias for collect_eager — read clearer at interop boundaries."""
    return collect_eager(df)


def to_utc_datetime(value: str | datetime | date | None) -> datetime | None:
    """Coerce a start/end bound (str | datetime | date) to a tz-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    # str: parse ISO-8601
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def scan_parquet(
    source: str | Path | list[str] | list[Path],
    *,
    symbols: list[str] | None = None,
    start: str | datetime | date | None = None,
    end: str | datetime | date | None = None,
    ts_col: str = "ts",
) -> LazyFrame:
    """Lazy scan over parquet with optional symbol/date predicate pushdown (D14(a)).

    Predicates are pushed into the scan so partitioned parquet skips unneeded
    symbol/year partitions on read. ``start``/``end`` accept ISO-8601 strings,
    datetimes, or dates; ``start`` is inclusive, ``end`` is exclusive. Returns a
    LazyFrame — the caller MUST collect_eager() at a strategy boundary before
    passing it anywhere that requires eager data.
    """
    lf: LazyFrame = pl.scan_parquet(source)
    predicates: list[pl.Expr] = []
    if symbols is not None:
        predicates.append(pl.col("symbol").is_in(symbols))
    start_dt = to_utc_datetime(start)
    end_dt = to_utc_datetime(end)
    if start_dt is not None:
        predicates.append(pl.col(ts_col) >= pl.lit(start_dt))
    if end_dt is not None:
        predicates.append(pl.col(ts_col) < pl.lit(end_dt))
    if predicates:
        lf = lf.filter(pl.all_horizontal(predicates))
    return lf


def write_parquet_eager(
    df: EagerFrame,
    target: str | Path,
    *,
    partition_by: list[str] | None = None,
    compression: Literal["zstd", "snappy", "gzip"] = "zstd",
) -> list[str]:
    """Eager write at ingest (plan §6.1: write_parquet eager at ingest).

    Partitions by ``symbol`` (hive-style) so reads get symbol partition
    pruning. Validates the bar schema first; rejects LazyFrame input. Returns
    the list of written partition paths (globbed after write — polars'
    ``write_parquet`` returns None even when partitioning, so we inspect the
    target directory).
    """
    assert_eager(df, "write_parquet_eager")
    target_path = Path(target)
    df.write_parquet(
        target,
        partition_by=partition_by,
        compression=compression,
    )
    return sorted(str(p) for p in target_path.rglob("*.parquet"))
