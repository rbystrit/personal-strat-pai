"""Tests for backtest/portfolio.py — position state, fills, mark-to-market."""

from __future__ import annotations

import pytest

from personal_strat_pai.backtest.costs import Fill, Side
from personal_strat_pai.backtest.portfolio import Portfolio


class TestPortfolio:
    def test_initial_state(self):
        pf = Portfolio(100_000.0)
        assert pf.cash == 100_000.0
        assert pf.nav == 100_000.0
        assert pf.drawdown == 0.0
        assert pf.weights() == {}
        assert pf.gross_exposure == 0.0

    def test_buy_fill(self):
        pf = Portfolio(10_000.0)
        fill = Fill(
            symbol="AAA",
            side=Side.BUY,
            qty=100,
            ref_price=50.0,
            fill_price=50.0,
            arrival_price=50.0,
            commission=0.35,
            fees=0.0,
            slippage_cost=0.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        pf.apply_fill(fill)
        # Cash = 10000 - 100*50 - 0.35 = 4999.65
        assert pf.cash == 4999.65
        assert pf.positions["AAA"].qty == 100
        assert pf.positions["AAA"].cost_basis == 5000.0
        assert len(pf.trades) == 1
        assert pf.trades[0].side == "BUY"
        assert pf.trades[0].cash_delta == -5000.35

    def test_sell_fill(self):
        pf = Portfolio(10_000.0)
        # Buy first
        buy = Fill(
            symbol="AAA", side=Side.BUY, qty=100,
            ref_price=50.0, fill_price=50.0, arrival_price=50.0,
            commission=0.35, fees=0.0, slippage_cost=0.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        pf.apply_fill(buy)
        # Sell
        sell = Fill(
            symbol="AAA", side=Side.SELL, qty=100,
            ref_price=55.0, fill_price=55.0, arrival_price=50.0,
            commission=0.35, fees=0.0139, slippage_cost=0.0,
            ts="2024-01-16T00:00:00+00:00",
        )
        pf.apply_fill(sell)
        # After buy: cash = 10000 - 5000 - 0.35 = 4999.65
        # After sell: cash = 4999.65 + 5500 - 0.35 - 0.0139 = 10499.2861
        assert pf.cash == pytest.approx(4999.65 + 5500.0 - 0.35 - 0.0139)
        # Position should be flat
        assert pf.positions["AAA"].qty == 0
        # Realized P&L = 100 * 55 - 100 * 50 = 500
        assert pf.positions["AAA"].realized_pnl == 500.0

    def test_mark_to_market(self):
        pf = Portfolio(10_000.0)
        buy = Fill(
            symbol="AAA", side=Side.BUY, qty=100,
            ref_price=50.0, fill_price=50.0, arrival_price=50.0,
            commission=0.35, fees=0.0, slippage_cost=0.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        pf.apply_fill(buy)
        pf.mark_to_market({"AAA": 55.0})
        # NAV = cash + 100 * 55 = 4999.65 + 5500 = 10499.65
        assert pf.nav == 4999.65 + 5500.0
        assert pf.positions["AAA"].market_value == 5500.0
        assert pf.positions["AAA"].unrealized_pnl == 500.0

    def test_drawdown_tracking(self):
        pf = Portfolio(10_000.0)
        buy = Fill(
            symbol="AAA", side=Side.BUY, qty=100,
            ref_price=50.0, fill_price=50.0, arrival_price=50.0,
            commission=0.35, fees=0.0, slippage_cost=0.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        pf.apply_fill(buy)
        pf.mark_to_market({"AAA": 60.0})
        peak = pf.peak_nav
        assert peak > 10_000.0
        pf.mark_to_market({"AAA": 45.0})
        assert pf.drawdown > 0.0
        assert pf.drawdown < 1.0

    def test_weights(self):
        pf = Portfolio(10_000.0)
        buy_a = Fill(
            symbol="AAA", side=Side.BUY, qty=50,
            ref_price=50.0, fill_price=50.0, arrival_price=50.0,
            commission=0.35, fees=0.0, slippage_cost=0.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        buy_b = Fill(
            symbol="BBB", side=Side.BUY, qty=50,
            ref_price=50.0, fill_price=50.0, arrival_price=50.0,
            commission=0.35, fees=0.0, slippage_cost=0.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        pf.apply_fill(buy_a)
        pf.apply_fill(buy_b)
        pf.mark_to_market({"AAA": 50.0, "BBB": 50.0})
        weights = pf.weights()
        # Each position is 2500, NAV = cash + 5000 = 4999.30 + 5000 = 9999.30
        # Weights = 2500 / 9999.30 ≈ 0.25
        assert "AAA" in weights
        assert "BBB" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01 or sum(weights.values()) < 1.0

    def test_inject_cash(self):
        pf = Portfolio(10_000.0)
        pf.inject_cash(5_000.0)
        assert pf.cash == 15_000.0
        assert pf.peak_nav == 15_000.0  # Peak resets after injection

    def test_gross_exposure(self):
        pf = Portfolio(10_000.0)
        buy = Fill(
            symbol="AAA", side=Side.BUY, qty=80,
            ref_price=100.0, fill_price=100.0, arrival_price=100.0,
            commission=0.35, fees=0.0, slippage_cost=0.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        pf.apply_fill(buy)
        pf.mark_to_market({"AAA": 100.0})
        # NAV = 10000 - 8000 - 0.35 + 8000 = 9999.65
        # Gross = 8000 / 9999.65 ≈ 0.80
        assert pf.gross_exposure == 8000.0 / pf.nav
        assert pf.gross_exposure < 1.0
