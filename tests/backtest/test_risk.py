"""Tests for backtest/risk.py — NAV cap, beta ceiling, kill switch, DD stops."""

from __future__ import annotations

import pytest

from personal_strat_pai.backtest.risk import (
    DrawdownStopHit,
    KillSwitchAction,
    RiskGuard,
    RiskLimits,
)


class SimpleBetaProvider:
    """Fixed beta per ticker for testing."""

    def __init__(self, betas: dict[str, float]) -> None:
        self._betas = betas

    def get_beta(self, symbol: str) -> float:
        return self._betas.get(symbol, 1.0)


class TestRiskGuard:
    def test_no_constraints_breach(self):
        guard = RiskGuard(RiskLimits())
        result = guard.check({"AAA": 0.3, "BBB": 0.3, "CCC": 0.3}, drawdown=0.0)
        assert result.weights == {"AAA": 0.3, "BBB": 0.3, "CCC": 0.3}
        assert not result.kill_switch_active
        assert not result.halted
        assert len(result.events) == 0

    def test_nav_cap_clips_weight(self):
        guard = RiskGuard(RiskLimits(nav_cap=0.30))
        result = guard.check({"AAA": 0.50, "BBB": 0.30}, drawdown=0.0)
        assert result.weights["AAA"] == 0.30
        assert result.weights["BBB"] == 0.30
        assert any(e.kind == "nav_cap" for e in result.events)

    def test_nav_cap_firf(self):
        guard = RiskGuard(RiskLimits(
            nav_cap=0.40,
            nav_cap_firf=0.20,
            firf_capped_buckets=frozenset({10}),  # Tech bucket
        ))
        result = guard.check(
            {"XLK": 0.35, "XLB": 0.35},
            drawdown=0.0,
            bucket_map={"XLK": 10, "XLB": 1},
        )
        assert result.weights["XLK"] == 0.20  # FIRF cap
        assert result.weights["XLB"] == 0.35  # Normal cap

    def test_kill_switch_halts_on_drawdown(self):
        guard = RiskGuard(RiskLimits(
            kill_switch_drawdown=0.20,
            kill_switch_action=KillSwitchAction.HALT_ENTRIES,
        ))
        result = guard.check({"AAA": 0.5}, drawdown=0.25)
        assert result.kill_switch_active
        assert result.halted
        assert any(e.kind == "kill_switch" for e in result.events)

    def test_kill_switch_latches(self):
        guard = RiskGuard(RiskLimits(
            kill_switch_drawdown=0.20,
            kill_switch_action=KillSwitchAction.HALT_ENTRIES,
        ))
        # Trip the switch
        guard.check({"AAA": 0.5}, drawdown=0.25)
        assert guard.kill_switch_active
        # Drawdown recovers but switch stays latched
        result = guard.check({"AAA": 0.5}, drawdown=0.05)
        assert result.kill_switch_active
        assert result.halted

    def test_kill_switch_reset(self):
        guard = RiskGuard(RiskLimits(
            kill_switch_drawdown=0.20,
            kill_switch_action=KillSwitchAction.HALT_ENTRIES,
        ))
        guard.check({"AAA": 0.5}, drawdown=0.25)
        assert guard.kill_switch_active
        guard.reset_kill_switch()
        assert not guard.kill_switch_active
        result = guard.check({"AAA": 0.5}, drawdown=0.05)
        assert not result.kill_switch_active

    def test_kill_switch_flatten(self):
        guard = RiskGuard(RiskLimits(
            kill_switch_drawdown=0.20,
            kill_switch_action=KillSwitchAction.FLATTEN,
        ))
        result = guard.check({"AAA": 0.5, "BBB": 0.5}, drawdown=0.25)
        assert result.weights == {}
        assert result.halted

    def test_gross_exposure_cap(self):
        guard = RiskGuard(RiskLimits(max_gross_exposure=1.0, nav_cap=1.0))
        result = guard.check({"AAA": 0.6, "BBB": 0.6}, drawdown=0.0)
        # Gross = 1.2 > 1.0, scaled by 1/1.2
        assert sum(abs(w) for w in result.weights.values()) <= 1.0 + 1e-9
        assert any(e.kind == "gross_exposure" for e in result.events)

    def test_beta_ceiling(self):
        guard = RiskGuard(RiskLimits(beta_ceiling=1.0, nav_cap=1.0))
        betas = SimpleBetaProvider({"AAA": 1.5, "BBB": 0.5})
        result = guard.check(
            {"AAA": 0.5, "BBB": 0.5},
            drawdown=0.0,
            beta_provider=betas,
        )
        # Portfolio beta = 0.5*1.5 + 0.5*0.5 = 1.0, at ceiling, no scaling
        assert result.weights["AAA"] == 0.5
        assert result.weights["BBB"] == 0.5

    def test_beta_ceiling_breach(self):
        guard = RiskGuard(RiskLimits(beta_ceiling=0.8, nav_cap=1.0))
        betas = SimpleBetaProvider({"AAA": 1.5, "BBB": 0.5})
        result = guard.check(
            {"AAA": 0.5, "BBB": 0.5},
            drawdown=0.0,
            beta_provider=betas,
        )
        # Portfolio beta = 1.0 > 0.8, scale = 0.8/1.0 = 0.8
        assert result.weights["AAA"] < 0.5  # Scaled down
        assert result.weights["BBB"] < 0.5  # All longs scaled proportionally
        assert any(e.kind == "beta_ceiling" for e in result.events)

    def test_dd_stop_raises(self):
        guard = RiskGuard(RiskLimits(account_dd_stop=0.15, kill_switch_drawdown=0.99))
        with pytest.raises(DrawdownStopHit, match="Account drawdown stop"):
            guard.check({"AAA": 1.0}, drawdown=0.20)

    def test_dd_stop_not_breached(self):
        guard = RiskGuard(RiskLimits(account_dd_stop=0.20))
        result = guard.check({"AAA": 1.0}, drawdown=0.15)
        assert not result.halted
