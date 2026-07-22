"""Tests for backtest/costs.py — IBKR Tiered cost model."""

from __future__ import annotations

import pytest

from personal_strat_pai.backtest.costs import (
    CommissionModel,
    FeeModel,
    IbkrTieredCostModel,
    Side,
    SlippageModel,
)


class TestCommissionModel:
    def test_per_share_basic(self):
        cm = CommissionModel(rate_per_share=0.0035, min_per_order=0.35)
        # 100 shares @ $50 = $5000 notional.
        # 100 * 0.0035 = $0.35, which is >= min ($0.35) and < 1% of $5000 ($50).
        assert cm.compute(100, 50.0) == pytest.approx(0.35)

    def test_per_share_above_min(self):
        cm = CommissionModel(rate_per_share=0.0035, min_per_order=0.35)
        # 200 shares @ $50 = $10000 notional.
        # 200 * 0.0035 = $0.70
        assert cm.compute(200, 50.0) == pytest.approx(0.70)

    def test_per_share_below_min(self):
        cm = CommissionModel(rate_per_share=0.0035, min_per_order=0.35)
        # 10 shares @ $50 = $500 notional.
        # 10 * 0.0035 = $0.035, min applies -> $0.35
        assert cm.compute(10, 50.0) == pytest.approx(0.35)

    def test_per_share_capped_at_max_pct(self):
        cm = CommissionModel(rate_per_share=0.0035, min_per_order=0.35, max_pct_of_notional=0.01)
        # 10000 shares @ $1 = $10000 notional.
        # 10000 * 0.0035 = $35, 1% of $10000 = $100. $35 < $100, no cap.
        assert cm.compute(10_000, 1.0) == pytest.approx(35.0)
        # 100000 shares @ $1 = $100000 notional.
        # 100000 * 0.0035 = $350, 1% of $100000 = $1000. $350 < $1000, no cap.
        assert cm.compute(100_000, 1.0) == pytest.approx(350.0)

    def test_zero_commission(self):
        cm = CommissionModel(kind="zero")
        assert cm.compute(100, 50.0) == 0.0

    def test_bps_commission(self):
        cm = CommissionModel(kind="bps", bps_rate=0.5)
        # 100 shares @ $50 = $5000 notional. 0.5 bps = 0.005% = $0.25
        assert cm.compute(100, 50.0) == pytest.approx(0.25)

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="unknown commission kind"):
            CommissionModel(kind="invalid")


class TestFeeModel:
    def test_no_fees_on_buy(self):
        fm = FeeModel()
        assert fm.compute(100, 50.0, Side.BUY) == 0.0

    def test_sec_fee_on_sell(self):
        fm = FeeModel(sec_fee_sell_bps=0.0278)
        # 100 shares @ $50 = $5000 sell principal.
        # SEC fee = 5000 * 0.0278 / 10000 = $0.0139
        fees = fm.compute(100, 50.0, Side.SELL)
        assert fees == pytest.approx(0.0139 + 100 * 0.000166)

    def test_finra_taf_capped(self):
        fm = FeeModel(finra_taf_per_share_sell=0.000166, finra_taf_max=8.30)
        # 100000 shares -> 100000 * 0.000166 = $16.60, capped at $8.30
        fees = fm.compute(100_000, 50.0, Side.SELL)
        # SEC fee + capped TAF
        sec = 100_000 * 50.0 * 0.0278 / 10_000.0
        assert fees == pytest.approx(sec + 8.30)


class TestSlippageModel:
    def test_none_slippage(self):
        sm = SlippageModel(kind="none")
        fill_price, _cost = sm.compute(100, 50.0, Side.BUY)
        assert fill_price == 50.0
        assert _cost == 0.0

    def test_half_spread_buy(self):
        sm = SlippageModel(kind="half_spread", half_spread_bps=5.0)
        fill_price, _cost = sm.compute(100, 50.0, Side.BUY)
        # Buy fills above: 50 * (1 + 5/10000) = 50.025
        assert fill_price == pytest.approx(50.025)
        assert _cost == pytest.approx(100 * 0.025)

    def test_half_spread_sell(self):
        sm = SlippageModel(kind="half_spread", half_spread_bps=5.0)
        fill_price, _cost = sm.compute(100, 50.0, Side.SELL)
        # Sell fills below: 50 * (1 - 5/10000) = 49.975
        assert fill_price == pytest.approx(49.975)
        assert _cost == pytest.approx(100 * 0.025)

    def test_sqrt_slippage_with_adv(self):
        sm = SlippageModel(kind="sqrt", half_spread_bps=5.0, sqrt_coeff=1.0)
        # 100 shares, adv = 1,000,000 -> sqrt(100/1e6) = 0.01 -> 0.01 bps
        # Total: 5 + 0.01 = 5.01 bps
        fill_price, _cost = sm.compute(100, 50.0, Side.BUY, adv=1_000_000)
        assert fill_price == pytest.approx(50.0 * (1 + 5.01 / 10_000))

    def test_sqrt_slippage_without_adv_falls_back(self):
        sm = SlippageModel(kind="sqrt", half_spread_bps=5.0)
        # No ADV -> just half-spread
        fill_price, _cost = sm.compute(100, 50.0, Side.BUY, adv=None)
        assert fill_price == pytest.approx(50.0 * (1 + 5 / 10_000))

    def test_jitter(self):
        sm = SlippageModel(kind="half_spread", half_spread_bps=5.0, jitter_bps=2.0)
        fill_price, _ = sm.compute(100, 50.0, Side.BUY, jitter=1.0)
        # 5 + 2 = 7 bps
        assert fill_price == pytest.approx(50.0 * (1 + 7 / 10_000))


class TestIbkrTieredCostModel:
    def test_buy_fill(self):
        cm = IbkrTieredCostModel()
        fill = cm.estimate_fill(
            symbol="AAA",
            side=Side.BUY,
            qty=100,
            ref_price=50.0,
            arrival_price=50.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        assert fill.symbol == "AAA"
        assert fill.side == Side.BUY
        assert fill.qty == 100
        assert fill.commission == pytest.approx(0.35)  # 100*0.0035 = 0.35 >= min
        assert fill.fees == 0.0  # No fees on buy
        assert fill.fill_price == pytest.approx(50.025)  # Default slippage: half_spread 5bps
        assert fill.total_cost > 0

    def test_sell_fill_with_fees(self):
        cm = IbkrTieredCostModel(
            slippage=SlippageModel(kind="none"),
        )
        fill = cm.estimate_fill(
            symbol="AAA",
            side=Side.SELL,
            qty=100,
            ref_price=50.0,
            arrival_price=48.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        assert fill.side == Side.SELL
        assert fill.fill_price == 50.0  # No slippage
        assert fill.commission == pytest.approx(0.35)
        # SEC fee + FINRA TAF
        assert fill.fees > 0
        assert fill.arrival_price == 48.0

    def test_fill_properties(self):
        cm = IbkrTieredCostModel(slippage=SlippageModel(kind="none"))
        fill = cm.estimate_fill(
            symbol="AAA",
            side=Side.BUY,
            qty=100,
            ref_price=50.0,
            arrival_price=50.0,
            ts="2024-01-15T00:00:00+00:00",
        )
        assert fill.notional == pytest.approx(5000.0)
        assert fill.total_cost == fill.commission + fill.fees + fill.slippage_cost
        assert fill.slippage_bps == 0.0  # No slippage
