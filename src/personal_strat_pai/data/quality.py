"""Data-quality checks (plan §6.5).

Every bar batch is checked: monotonic timestamps, no null OHLC, volume >= 0,
price within sane bounds, split-adjustment continuity. Failures BLOCK the
consuming job and page via alerts — no silent data errors (priority #2).

Usage:
    report = validate_bars(df)             # returns QualityReport, does not raise
    report.raise_if_failed()               # raises DataQualityError -> blocks consumer
    # or
    validate_bars(df, raise_on_fail=True)  # convenience: raises directly

Checks are composable — each returns a list[Violation]; validate_bars aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from personal_strat_pai.data.polars_utils import BAR_COLUMNS, assert_eager, collect_eager

__all__ = [
    "DEFAULT_PRICE_BOUNDS",
    "DEFAULT_SPLIT_GAP_THRESHOLD",
    "DataQualityError",
    "QualityReport",
    "Violation",
    "check_monotonic_ts",
    "check_no_null_ohlc",
    "check_price_bounds",
    "check_split_continuity",
    "check_volume_nonneg",
    "validate_bars",
]

# Conservative defaults. Tunable via call-site; final values live in config/strategy.yaml.
DEFAULT_PRICE_BOUNDS: tuple[float, float] = (1e-6, 1e6)
# Flag a close-to-close move > 50% as a potential unexplained split/discontinuity.
# Real splits are reconciled by corp_actions.cross_validate before reaching here,
# so a gap this large without a recorded corp action is a data error (plan §6.4/§6.5).
DEFAULT_SPLIT_GAP_THRESHOLD: float = 0.50


@dataclass(frozen=True, slots=True)
class Violation:
    """A single data-quality violation."""

    check: str
    symbol: str | None
    detail: str
    row_count: int = 1
    sample: dict[str, Any] | None = None


@dataclass(slots=True)
class QualityReport:
    """Aggregate result of running all checks on a bar batch."""

    passed: bool
    violations: list[Violation] = field(default_factory=list)
    row_count: int = 0
    symbol_count: int = 0

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise DataQualityError(self)

    def summary(self) -> str:
        if self.passed:
            return f"OK: {self.row_count} rows, {self.symbol_count} symbols"
        checks = ", ".join(sorted({v.check for v in self.violations}))
        return f"FAIL: {len(self.violations)} violation(s) across [{checks}]"


class DataQualityError(ValueError):
    """Raised when a bar batch fails quality checks — blocks the consuming job."""

    def __init__(self, report: QualityReport) -> None:
        self.report = report
        super().__init__(f"data-quality failure: {report.summary()}")


def _ensure_columns(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(
            QualityReport(
                passed=False,
                violations=[Violation("schema", None, f"missing columns: {missing}")],
            )
        )
    return df


def check_monotonic_ts(df: pl.DataFrame) -> list[Violation]:
    """Per symbol, ts must be strictly increasing (plan §6.5: monotonic timestamps)."""
    out: list[Violation] = []
    bad = (
        df.sort(["symbol", "ts"])
        .group_by("symbol", maintain_order=True)
        .agg(
            pl.col("ts").diff().dt.total_microseconds().alias("_dt"),
            pl.col("ts").alias("_ts"),
        )
    )
    for row in bad.iter_rows(named=True):
        nonpos = row["_dt"]
        ts = row["_ts"]
        idxs = [i for i, v in enumerate(nonpos) if v is not None and v <= 0]
        for i in idxs:
            out.append(
                Violation(
                    check="monotonic_ts",
                    symbol=row["symbol"],
                    detail=f"ts not strictly increasing at {ts[i]}",
                    sample={"prev": ts[i - 1] if i > 0 else None, "curr": ts[i]},
                )
            )
    return out


def check_no_null_ohlc(df: pl.DataFrame) -> list[Violation]:
    """No nulls in open/high/low/close (plan §6.5)."""
    out: list[Violation] = []
    nulls = (
        df.group_by("symbol")
        .agg(
            pl.col("open").null_count().alias("open_nulls"),
            pl.col("high").null_count().alias("high_nulls"),
            pl.col("low").null_count().alias("low_nulls"),
            pl.col("close").null_count().alias("close_nulls"),
        )
        .filter(
            (pl.col("open_nulls") > 0)
            | (pl.col("high_nulls") > 0)
            | (pl.col("low_nulls") > 0)
            | (pl.col("close_nulls") > 0)
        )
    )
    for row in nulls.iter_rows(named=True):
        out.append(
            Violation(
                check="no_null_ohlc",
                symbol=row["symbol"],
                detail=(
                    f"null OHLC counts: open={row['open_nulls']} "
                    f"high={row['high_nulls']} low={row['low_nulls']} close={row['close_nulls']}"
                ),
            )
        )
    return out


def check_volume_nonneg(df: pl.DataFrame) -> list[Violation]:
    """volume >= 0 (plan §6.5)."""
    bad = df.filter(pl.col("volume") < 0)
    out: list[Violation] = []
    for row in bad.group_by("symbol").agg(pl.len().alias("n")).iter_rows(named=True):
        out.append(
            Violation(
                check="volume_nonneg",
                symbol=row["symbol"],
                detail=f"{row['n']} row(s) with volume < 0",
                row_count=int(row["n"]),
            )
        )
    return out


def check_price_bounds(
    df: pl.DataFrame, bounds: tuple[float, float] = DEFAULT_PRICE_BOUNDS
) -> list[Violation]:
    """Price within sane bounds + OHLC consistency (plan §6.5: price bounds).

    Checks: 0 < low <= high < upper; low <= open/close <= high; all within (lo, hi).
    """
    lo, hi = bounds
    out: list[Violation] = []
    invalid = df.filter(
        (pl.col("low") <= lo)
        | (pl.col("high") >= hi)
        | (pl.col("low") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
    )
    for row in invalid.group_by("symbol").agg(pl.len().alias("n")).iter_rows(named=True):
        out.append(
            Violation(
                check="price_bounds",
                symbol=row["symbol"],
                detail=f"{row['n']} row(s) fail OHLC consistency / bounds ({lo}, {hi})",
                row_count=int(row["n"]),
            )
        )
    return out


def check_split_continuity(
    df: pl.DataFrame, gap_threshold: float = DEFAULT_SPLIT_GAP_THRESHOLD
) -> list[Violation]:
    """Split-adjustment continuity (plan §6.5).

    Flags any close-to-close single-day return exceeding ``gap_threshold`` as a
    potential unexplained split/discontinuity. Real splits are reconciled by
    ``corp_actions.cross_validate`` BEFORE bars reach the store, so a gap this
    large without a recorded corp action is a data error.
    """
    out: list[Violation] = []
    rets = (
        df.sort(["symbol", "ts"])
        .group_by("symbol", maintain_order=True)
        .agg(
            pl.col("ts").alias("_ts"),
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("_ret"),
        )
    )
    for row in rets.iter_rows(named=True):
        ts = row["_ts"]
        rets_row = row["_ret"]
        for i, r in enumerate(rets_row):
            if r is None:
                continue
            if abs(r) > gap_threshold:
                out.append(
                    Violation(
                        check="split_continuity",
                        symbol=row["symbol"],
                        detail=f"close-to-close return {r:.4f} exceeds {gap_threshold:.2f}",
                        sample={"ts": ts[i], "return": float(r)},
                    )
                )
    return out


def validate_bars(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    price_bounds: tuple[float, float] = DEFAULT_PRICE_BOUNDS,
    split_gap_threshold: float = DEFAULT_SPLIT_GAP_THRESHOLD,
    raise_on_fail: bool = False,
) -> QualityReport:
    """Run all data-quality checks on a bar batch (plan §6.5).

    Accepts eager or lazy; lazy is collected once here (the quality gate is a
    boundary — eager by the time checks run). Returns a QualityReport. If
    ``raise_on_fail`` is True, raises DataQualityError on failure (blocks the
    consuming job).
    """
    eager = collect_eager(df)
    assert_eager(eager, "validate_bars")
    eager = _ensure_columns(eager)

    violations: list[Violation] = []
    violations += check_monotonic_ts(eager)
    violations += check_no_null_ohlc(eager)
    violations += check_volume_nonneg(eager)
    violations += check_price_bounds(eager, bounds=price_bounds)
    violations += check_split_continuity(eager, gap_threshold=split_gap_threshold)

    report = QualityReport(
        passed=len(violations) == 0,
        violations=violations,
        row_count=eager.height,
        symbol_count=eager["symbol"].n_unique(),
    )
    if raise_on_fail:
        report.raise_if_failed()
    return report
