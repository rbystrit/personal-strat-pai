"""Tests for backtest/data.py — bar driver and trading calendar."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from personal_strat_pai.backtest.data import ParquetBarDriver, TradingCalendar
from personal_strat_pai.data.polars_utils import BAR_SCHEMA


def _write_test_bars(tmp_path: Path) -> Path:
    """Write 3 symbols × 30 days of bars to parquet."""
    from personal_strat_pai.data.store import BarStore

    start = datetime(2024, 1, 2, tzinfo=UTC)
    rows = []
    for sym in ["AAA", "BBB", "CCC"]:
        for i in range(30):
            ts = start + timedelta(days=i)
            price = 100.0 + i * 0.5
            rows.append({
                "symbol": sym,
                "ts": ts,
                "open": float(price),
                "high": float(price + 0.5),
                "low": float(price - 0.5),
                "close": float(price),
                "volume": 1_000_000,
            })
    df = pl.DataFrame(rows)
    df = df.cast({c: t for c, t in BAR_SCHEMA.items() if c in df.columns})
    store = BarStore(tmp_path / "bars")
    store.write_bars(df, kind="daily")
    return tmp_path / "bars" / "daily"


class TestParquetBarDriver:
    def test_all_symbols(self, tmp_path):
        parquet_path = _write_test_bars(tmp_path)
        driver = ParquetBarDriver(parquet_path)
        syms = driver.all_symbols()
        assert sorted(syms) == ["AAA", "BBB", "CCC"]

    def test_trading_days(self, tmp_path):
        parquet_path = _write_test_bars(tmp_path)
        driver = ParquetBarDriver(parquet_path)
        days = driver.trading_days()
        assert len(days) == 30
        assert days[0] == date(2024, 1, 2)
        assert days[-1] == date(2024, 1, 31)

    def test_bars_through(self, tmp_path):
        parquet_path = _write_test_bars(tmp_path)
        driver = ParquetBarDriver(parquet_path)
        as_of = datetime(2024, 1, 10, tzinfo=UTC)
        bars = driver.bars_through(["AAA", "BBB"], as_of)
        assert isinstance(bars, pl.DataFrame)
        # Should have 9 days × 2 symbols = 18 rows
        assert bars.height == 18
        assert set(bars["symbol"].unique().to_list()) == {"AAA", "BBB"}
        # No look-ahead: all ts <= as_of
        assert bars["ts"].max() <= as_of

    def test_fill_bar_for(self, tmp_path):
        parquet_path = _write_test_bars(tmp_path)
        driver = ParquetBarDriver(parquet_path)
        after = datetime(2024, 1, 10, tzinfo=UTC)
        fill_bars = driver.fill_bar_for(["AAA", "BBB"], after)
        # Should return the first bar after Jan 10 = Jan 11
        assert fill_bars.height == 2  # one per symbol
        assert set(fill_bars["symbol"].unique().to_list()) == {"AAA", "BBB"}
        # The open price should be for Jan 11 (day 9, price = 100 + 9*0.5 = 104.5)
        aaa = fill_bars.filter(pl.col("symbol") == "AAA")
        assert aaa["open"][0] == 104.5

    def test_symbol_filter(self, tmp_path):
        parquet_path = _write_test_bars(tmp_path)
        driver = ParquetBarDriver(parquet_path, symbols=["AAA"])
        bars = driver.bars_through(["AAA"], datetime(2024, 1, 5, tzinfo=UTC))
        assert set(bars["symbol"].unique().to_list()) == {"AAA"}

    def test_lazy_scan_no_leak(self, tmp_path):
        """The driver never leaks a LazyFrame across its API (D14(b))."""
        parquet_path = _write_test_bars(tmp_path)
        driver = ParquetBarDriver(parquet_path)
        bars = driver.bars_through(["AAA"], datetime(2024, 1, 5, tzinfo=UTC))
        assert isinstance(bars, pl.DataFrame)
        assert not isinstance(bars, pl.LazyFrame)


class TestTradingCalendar:
    def test_month_ends(self):
        days = [
            date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 31),
            date(2024, 2, 1), date(2024, 2, 15), date(2024, 2, 29),
            date(2024, 3, 1), date(2024, 3, 15), date(2024, 3, 28),
        ]
        cal = TradingCalendar(days)
        ends = cal.month_ends()
        assert ends == [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 28)]

    def test_is_trading_day(self):
        cal = TradingCalendar([date(2024, 1, 2), date(2024, 1, 3)])
        assert cal.is_trading_day(date(2024, 1, 2))
        assert not cal.is_trading_day(date(2024, 1, 4))

    def test_next_trading_day_after(self):
        cal = TradingCalendar([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)])
        assert cal.next_trading_day_after(date(2024, 1, 2)) == date(2024, 1, 3)
        assert cal.next_trading_day_after(date(2024, 1, 4)) is None

    def test_empty_calendar(self):
        cal = TradingCalendar([])
        assert cal.month_ends() == []
        assert cal.days == []
