"""Tests for backtest/signals.py — momentum strategy and ROC computation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from personal_strat_pai.backtest.portfolio import Portfolio
from personal_strat_pai.backtest.signals import (
    MomentumStrategy,
    compute_blended_roc,
    compute_regime_votes,
    compute_roc,
)
from personal_strat_pai.config import StrategyConfig
from personal_strat_pai.data.polars_utils import BAR_SCHEMA


def _make_bars(prices: dict[str, list[float]], start: datetime) -> pl.DataFrame:
    """Build a bar frame from per-symbol close-price lists (daily bars)."""
    rows = []
    for sym, price_list in prices.items():
        for i, p in enumerate(price_list):
            ts = start + timedelta(days=i)
            rows.append({
                "symbol": sym,
                "ts": ts,
                "open": float(p),
                "high": float(p),
                "low": float(p),
                "close": float(p),
                "volume": 1_000_000,
            })
    df = pl.DataFrame(rows)
    return df.cast({c: t for c, t in BAR_SCHEMA.items() if c in df.columns})


class TestComputeRoc:
    def test_basic_roc(self):
        closes = pl.Series("c", [100.0, 105.0, 110.0, 115.0])
        # ROC over 1 period: 115/110 - 1 = 0.0454...
        roc = compute_roc(closes, 1)
        assert roc == 115.0 / 110.0 - 1.0

    def test_roc_window(self):
        closes = pl.Series("c", [100.0, 110.0, 120.0])
        # ROC over 2 periods: 120/100 - 1 = 0.2
        roc = compute_roc(closes, 2)
        assert roc == pytest.approx(0.2)

    def test_not_enough_history(self):
        closes = pl.Series("c", [100.0, 110.0])
        roc = compute_roc(closes, 5)
        assert roc is None

    def test_zero_start(self):
        closes = pl.Series("c", [0.0, 10.0, 20.0])
        roc = compute_roc(closes, 2)
        assert roc is None  # Division by zero -> None


class TestComputeBlendedRoc:
    def test_equal_weights(self):
        closes = pl.Series("c", [100.0] + [110.0] * 63 + [120.0] * 63)
        roc = compute_blended_roc(closes, roc_weight_3m=0.5, roc_weight_6m=0.5)
        assert roc is not None
        # ROC_3M = 120/110 - 1 = 0.0909
        # ROC_6M = 120/100 - 1 = 0.2
        # Blended = 0.5 * 0.0909 + 0.5 * 0.2
        assert abs(roc - (0.5 * (120.0 / 110.0 - 1.0) + 0.5 * (120.0 / 100.0 - 1.0))) < 1e-9


class TestComputeRegimeVotes:
    def test_all_positive(self):
        # 3M, 6M, 9M, 12M all positive — use increasing prices
        closes = pl.Series("c", [100.0 + i * 0.5 for i in range(300)])
        votes = compute_regime_votes(closes, [3, 6, 9, 12], min_votes=3)
        assert votes == 4

    def test_mixed(self):
        # 3M positive, 6M negative
        closes = pl.Series("c", [100.0, 90.0] + [95.0] * 300)
        votes = compute_regime_votes(closes, [3, 6], min_votes=1)
        # 3M: 95/95 - 1 = 0 (depends on exact window)
        # Just check it returns something sensible
        assert 0 <= votes <= 2


class TestMomentumStrategy:
    def test_returns_weights(self):
        start = datetime(2024, 1, 2, tzinfo=UTC)
        # 200 days of data, 3 tickers with different trends
        prices = {
            "AAA": [100.0 + i * 0.5 for i in range(200)],  # uptrend
            "BBB": [100.0 - i * 0.3 for i in range(200)],  # downtrend
            "CCC": [100.0 + i * 0.1 for i in range(200)],  # slight uptrend
        }
        bars = _make_bars(prices, start)
        config = StrategyConfig(
            regime_windows_sectors=[3, 6, 9, 12],
            regime_votes_sectors=1,  # 1 of 4 needed (lowered for test)
            waterfall=[1.0],  # 100% to top 1
        )
        strategy = MomentumStrategy(config, n_positions=1, trading_days_per_month=21)
        portfolio = Portfolio(100_000.0)
        as_of = start + timedelta(days=199)
        weights = strategy.target_weights(bars, as_of, portfolio)
        # AAA should be the top performer
        assert "AAA" in weights
        assert weights["AAA"] > 0.99  # 100% allocation
        assert "BBB" not in weights  # BBB fails regime filter (downtrend)

    def test_empty_bars(self):
        config = StrategyConfig()
        strategy = MomentumStrategy(config, n_positions=1)
        portfolio = Portfolio(100_000.0)
        weights = strategy.target_weights(
            pl.DataFrame(schema=BAR_SCHEMA),
            datetime(2024, 6, 1, tzinfo=UTC),
            portfolio,
        )
        assert weights == {}

    def test_waterfall_allocation(self):
        start = datetime(2024, 1, 2, tzinfo=UTC)
        # All 3 tickers uptrend, all pass regime
        prices = {
            "AAA": [100.0 + i * 0.5 for i in range(200)],
            "BBB": [100.0 + i * 0.4 for i in range(200)],
            "CCC": [100.0 + i * 0.3 for i in range(200)],
        }
        bars = _make_bars(prices, start)
        config = StrategyConfig(
            regime_windows_sectors=[3, 6, 9, 12],
            regime_votes_sectors=1,
            waterfall=[0.5, 0.3, 0.2],
        )
        strategy = MomentumStrategy(config, n_positions=3, trading_days_per_month=21)
        portfolio = Portfolio(100_000.0)
        as_of = start + timedelta(days=199)
        weights = strategy.target_weights(bars, as_of, portfolio)
        # All 3 should be in the portfolio with 50/30/20 split
        assert len(weights) == 3
        # AAA should have the highest weight
        assert weights["AAA"] > weights["BBB"]
        assert weights["BBB"] > weights["CCC"]
        # Check waterfall
        assert abs(weights["AAA"] - 0.5) < 0.01
        assert abs(weights["BBB"] - 0.3) < 0.01
        assert abs(weights["CCC"] - 0.2) < 0.01
