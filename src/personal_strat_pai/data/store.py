"""Parquet I/O via polars + local SQLite read cache (plan §6.1, D14).

Write path:  eager ``write_parquet`` at ingest, partitioned by ``symbol``
(hive-style) so ``scan_parquet`` gets symbol-level partition pruning — the main
predicate-pushdown win for the ~45-ETF × multi-year dataset (D14(a)).

Read path:  lazy ``scan_parquet`` with optional symbol/date predicates pushed
into the scan. The caller MUST ``collect_eager()`` at a strategy boundary
before handing the result to anything eager (pre-trade checks, sieve decisions).

Backing store: local filesystem is fully implemented and tested in CI. The OCI
Object Storage backend is wired in P0-2 once creds/namespace are configured; the
interface (``base_uri`` accepting ``file://`` or ``oci://``) is ready for it.

No-double-download (CEO directive 2026-07-18): ``upsert_bars`` is the
idempotent merge path — new rows are merged into existing partitions by
``(symbol, ts)`` (keep-latest) so re-fetching an overlapping range never
duplicates rows. ``coverage`` returns the per-symbol ``(first_ts, last_ts)``
so the caching repo (data/repo.py) can compute the missing range to fetch
without re-downloading anything already in the store.

SQLite read cache: the podman primary's hot read path (plan §6.1 — last ~260
trading days for 200D SMA / 12m ROC). ``SQLiteCache`` is a working read-through
cache; the write-through wiring on ingest is P0-2/P0-4.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from personal_strat_pai.data.polars_utils import (
    BAR_COLUMNS,
    BAR_SCHEMA,
    assert_eager,
    collect_eager,
    to_utc_datetime,
)

__all__ = [
    "BarKind",
    "BarStore",
    "Coverage",
    "OciBackendNotConfigured",
    "SQLiteCache",
]

BarKind = Literal["daily", "minute"]
Coverage = tuple[datetime, datetime]


class OciBackendNotConfigured(NotImplementedError):
    """Raised when an ``oci://`` base_uri is used before the P0-2 OCI wiring lands."""


def _resolve_base(base_uri: str | Path) -> Path:
    """Resolve a base_uri to a local Path. ``oci://`` backends land in P0-2."""
    s = str(base_uri)
    if s.startswith("oci://"):
        raise OciBackendNotConfigured(
            "OCI Object Storage backend is wired in P0-2 (creds + namespace). "
            "Use a local 'file://' or plain path for P0-1."
        )
    if s.startswith("file://"):
        s = s[len("file://") :]
    return Path(s)


class BarStore:
    """Parquet-backed bar store (plan §6.1). Lazy read, eager write (D14)."""

    def __init__(self, base_uri: str | Path = "data/local/bars") -> None:
        self.base_uri = base_uri
        self._base = _resolve_base(base_uri)

    def _kind_dir(self, kind: BarKind) -> Path:
        return self._base / kind

    def write_bars(
        self,
        df: pl.DataFrame,
        kind: BarKind = "daily",
        *,
        compression: Literal["zstd", "snappy", "gzip"] = "zstd",
    ) -> list[str]:
        """Eager write at ingest (plan §6.1: write_parquet eager at ingest).

        Partitions by ``symbol`` (hive-style) so reads get symbol partition
        pruning. Validates the bar schema first; rejects LazyFrame input.
        """
        assert_eager(df, "BarStore.write_bars")
        self._validate_bar_schema(df)
        target = self._kind_dir(kind)
        target.mkdir(parents=True, exist_ok=True)
        df.write_parquet(
            target,
            partition_by=["symbol"],
            compression=compression,
        )
        # polars' write_parquet returns None even when partitioning; glob the
        # target directory for the written partition paths.
        return sorted(str(p) for p in target.rglob("*.parquet"))

    def scan_bars(
        self,
        kind: BarKind = "daily",
        *,
        symbols: list[str] | None = None,
        start: str | datetime | date | None = None,
        end: str | datetime | date | None = None,
    ) -> pl.LazyFrame:
        """Lazy scan over partitioned parquet with predicate pushdown (D14(a)).

        Returns a LazyFrame. The caller MUST ``collect_eager()`` at a strategy
        boundary. ``start`` inclusive, ``end`` exclusive (ISO-8601 strings,
        datetimes, or dates).
        """
        kind_dir = self._kind_dir(kind)
        if not kind_dir.exists():
            # Empty store -> return an empty lazy frame with the canonical schema
            # so downstream code sees the right columns.
            return pl.DataFrame(schema=BAR_SCHEMA).lazy()
        lf = pl.scan_parquet(kind_dir, hive_partitioning=True)
        predicates: list[pl.Expr] = []
        if symbols is not None:
            predicates.append(pl.col("symbol").is_in(symbols))
        start_dt = to_utc_datetime(start)
        end_dt = to_utc_datetime(end)
        if start_dt is not None:
            predicates.append(pl.col("ts") >= pl.lit(start_dt))
        if end_dt is not None:
            predicates.append(pl.col("ts") < pl.lit(end_dt))
        if predicates:
            lf = lf.filter(pl.all_horizontal(predicates))
        # Re-select canonical columns in canonical order (hive may add extras).
        return lf.select(list(BAR_COLUMNS))

    def read_bars_eager(
        self,
        kind: BarKind = "daily",
        *,
        symbols: list[str] | None = None,
        start: str | datetime | date | None = None,
        end: str | datetime | date | None = None,
    ) -> pl.DataFrame:
        """Eager read — collect at the boundary. Use for small slices / pre-trade checks."""
        return collect_eager(self.scan_bars(kind, symbols=symbols, start=start, end=end))

    def upsert_bars(
        self,
        df: pl.DataFrame,
        kind: BarKind = "daily",
        *,
        compression: Literal["zstd", "snappy", "gzip"] = "zstd",
    ) -> dict[str, int]:
        """Idempotent upsert — merge ``df`` into existing partitions by ``(symbol, ts)``.

        CEO directive 2026-07-18 (no piece of data downloaded twice): the caching
        repo fetches only the missing range and calls ``upsert_bars`` so re-fetching
        an overlapping range is a no-op (keep-latest on ``(symbol, ts)``), never a
        duplicate. Per-symbol: read the existing partition, concat with the new
        rows, unique by ``(symbol, ts)`` keeping the new (latest) rows, sort by ts,
        then rewrite the partition. Returns ``{symbol: rows_in_partition_after}``.
        """
        assert_eager(df, "BarStore.upsert_bars")
        self._validate_bar_schema(df)
        target = self._kind_dir(kind)
        target.mkdir(parents=True, exist_ok=True)
        result: dict[str, int] = {}
        for sym in df["symbol"].unique().sort().to_list():
            new_rows = df.filter(pl.col("symbol") == sym)
            existing = self._read_partition(kind, sym)
            if existing.is_empty() and new_rows.is_empty():
                result[sym] = 0
                continue
            merged = (
                pl.concat([existing, new_rows], how="vertical_relaxed")
                .unique(subset=["symbol", "ts"], keep="last")
                .sort(["ts"])
            )
            self._rewrite_partition(target, sym, merged, compression=compression)
            result[sym] = merged.height
        return result

    def coverage(
        self,
        kind: BarKind = "daily",
        *,
        symbols: list[str] | None = None,
    ) -> dict[str, Coverage]:
        """Per-symbol ``(first_ts, last_ts)`` from the store (CEO no-double-download).

        Eager, small aggregation — drives the caching repo's missing-range
        computation. Returns ``{}`` for an empty store. Datetimes are tz-aware UTC.
        """
        kind_dir = self._kind_dir(kind)
        if not kind_dir.exists():
            return {}
        lf = pl.scan_parquet(kind_dir, hive_partitioning=True)
        if symbols is not None:
            lf = lf.filter(pl.col("symbol").is_in(symbols))
        agg = (
            lf.group_by("symbol")
            .agg(
                pl.col("ts").min().alias("first_ts"),
                pl.col("ts").max().alias("last_ts"),
            )
            .collect()
        )
        out: dict[str, Coverage] = {}
        for row in agg.iter_rows(named=True):
            out[row["symbol"]] = (row["first_ts"], row["last_ts"])
        return out

    def _read_partition(self, kind: BarKind, symbol: str) -> pl.DataFrame:
        """Read a single symbol's partition eagerly (empty BAR_SCHEMA if missing).

        Scans from the kind root with a symbol filter (the proven ``scan_bars``
        pattern) so the hive ``symbol=<sym>`` key is reconstructed from the path
        and the leaf files — which drop the partition column when written via
        ``partition_by`` — are read correctly.
        """
        kind_dir = self._kind_dir(kind)
        part_dir = kind_dir / f"symbol={symbol}"
        if not kind_dir.exists() or not part_dir.exists() or not any(part_dir.rglob("*.parquet")):
            return pl.DataFrame(schema=BAR_SCHEMA)
        return collect_eager(
            pl.scan_parquet(kind_dir, hive_partitioning=True).filter(pl.col("symbol") == symbol)
        ).select(list(BAR_COLUMNS))

    @staticmethod
    def _rewrite_partition(
        kind_dir: Path,
        symbol: str,
        merged: pl.DataFrame,
        *,
        compression: Literal["zstd", "snappy", "gzip"],
    ) -> None:
        """Rewrite a single symbol's partition in the canonical hive layout.

        Removes the existing ``symbol=<sym>`` partition dir, then writes the
        merged frame with ``partition_by=["symbol"]`` so the leaf file drops the
        partition column (matching ``write_bars``); hive scanning reconstructs
        ``symbol`` from the path. Other symbols' partitions are untouched.
        """
        part_dir = kind_dir / f"symbol={symbol}"
        if part_dir.exists():
            shutil.rmtree(part_dir)
        merged.write_parquet(kind_dir, partition_by=["symbol"], compression=compression)

    @staticmethod
    def _validate_bar_schema(df: pl.DataFrame) -> None:
        for col, dtype in BAR_SCHEMA.items():
            if col not in df.columns:
                raise ValueError(f"bar schema missing column: {col!r}")
            if df.schema[col] != dtype and not _dtype_compatible(df.schema[col], dtype):
                raise ValueError(
                    f"bar schema column {col!r}: expected {dtype}, got {df.schema[col]}"
                )


class SQLiteCache:
    """Local read-through cache for the recent rolling window (plan §6.1).

    Holds the most-recent ~260 trading days per symbol for sub-minute Risk-Clock
    reads. Write-through on ingest lands in P0-2/P0-4; this provides the working
    read/write implementation now.
    """

    def __init__(self, path: str | Path = "data/local/cache.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recent_bars (
                    symbol   TEXT    NOT NULL,
                    ts       INTEGER NOT NULL,   -- microseconds since epoch (UTC)
                    open     REAL    NOT NULL,
                    high     REAL    NOT NULL,
                    low      REAL    NOT NULL,
                    close    REAL    NOT NULL,
                    volume   INTEGER NOT NULL,
                    PRIMARY KEY (symbol, ts)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_recent_bars_symbol_ts "
                "ON recent_bars(symbol, ts DESC)"
            )

    def put_bars(self, df: pl.DataFrame | pl.LazyFrame) -> int:
        """Upsert bars into the cache. Returns the number of rows written."""
        eager = collect_eager(df)
        assert_eager(eager, "SQLiteCache.put_bars")
        rows = eager.select(list(BAR_COLUMNS)).with_columns(
            pl.col("ts").dt.timestamp("us").alias("ts")
        )
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO recent_bars "
                "(symbol, ts, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows.iter_rows(),
            )
            return eager.height

    def get_recent(self, symbol: str, n_days: int = 260) -> pl.DataFrame:
        """Return the last ``n_days`` daily bars for ``symbol`` as a polars frame."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol, ts, open, high, low, close, volume "
                "FROM recent_bars WHERE symbol = ? "
                "ORDER BY ts DESC LIMIT ?",
                (symbol, n_days),
            ).fetchall()
        if not rows:
            return pl.DataFrame(schema=BAR_SCHEMA)
        # ts is int microseconds since epoch -> convert back to UTC datetime.
        # Schema order MUST match the row-tuple order (symbol, ts, ohlc, volume).
        df = pl.DataFrame(
            rows,
            schema={
                "symbol": pl.String,
                "ts": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
            orient="row",
        )
        df = df.with_columns(pl.from_epoch("ts", time_unit="us").dt.replace_time_zone("UTC"))
        return df.select(list(BAR_COLUMNS)).sort("ts")

    def get_recent_close(self, symbol: str) -> float | None:
        """Scalar latest close for ``symbol`` (Risk-Clock hot read)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT close FROM recent_bars WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return float(row[0]) if row else None


def _dtype_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    """Tolerant schema check: Datetime zones / int widths may differ across writes."""
    if actual == expected:
        return True
    # Datetime: accept any time zone / unit (parquet round-trips may normalize).
    if isinstance(actual, pl.Datetime) and isinstance(expected, pl.Datetime):
        return True
    # Int: accept i64 expected vs other int widths (parquet may down-cast on write).
    return isinstance(actual, _INT_TYPES) and isinstance(expected, _INT_TYPES)


_INT_TYPES: tuple[type, ...] = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)


# Re-export to_lazy for callers that build lazy pipelines off store output.
__all__.append("to_lazy")
