"""Tests for config.py — universe/strategy/risk_limits (D7 defaults, D8 CEO-SET; RBY-4 acceptance)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from personal_strat_pai.config import (
    CONFIG_DIR,
    PIMCO_BLOCKLIST,
    RiskLimitsConfig,
    StrategyConfig,
    UniverseConfig,
    load_risk_limits,
    load_strategy,
    load_universe,
)


def test_load_universe_has_13_buckets():
    cfg = load_universe()
    assert isinstance(cfg, UniverseConfig)
    assert len(cfg.buckets) == 13
    ids = [b.id for b in cfg.buckets]
    assert sorted(ids) == list(range(1, 14))


def test_universe_39_etfs_no_duplicates():
    cfg = load_universe()
    tickers = cfg.all_tickers()
    assert len(tickers) == 39
    assert len(set(tickers)) == 39  # no ETF in two buckets


def test_universe_includes_parking_lot():
    cfg = load_universe()
    all_with_parking = cfg.all_tickers_with_parking()
    for t in ("SUB", "VTES", "SGOV"):
        assert t in all_with_parking


def test_universe_crypto_bucket_flagged():
    cfg = load_universe()
    crypto = [b for b in cfg.buckets if b.is_crypto]
    assert len(crypto) == 1
    assert crypto[0].id == 13
    assert set(crypto[0].etfs) == {"IBIT", "ETHA", "GSOL"}


def test_universe_firf_capped_buckets():
    """plan §6.3: Tech, Real Estate, Crypto are FIRF-capped (20% NAV)."""
    cfg = load_universe()
    firf = {b.name for b in cfg.buckets if b.firf_capped}
    assert firf == {"Real Estate", "Technology", "Crypto"}


def test_universe_rejects_pimco_ticker(tmp_path: Path):
    buckets = []
    for i in range(1, 13):
        buckets.append(
            {"id": i, "name": f"b{i}", "etf_a": f"A{i}", "etf_b": f"B{i}", "etf_c": f"C{i}"}
        )
    # bucket 13 uses BOND — a PIMCO ticker (brief §2 Anti-PIMCO rule).
    buckets.append({"id": 13, "name": "bad", "etf_a": "BOND", "etf_b": "X13", "etf_c": "Y13"})
    p = tmp_path / "u.yaml"
    p.write_text(yaml.safe_dump({"buckets": buckets}))
    with pytest.raises(ValueError, match="PIMCO"):
        load_universe(p)


def test_universe_rejects_duplicate_etf(tmp_path: Path):
    buckets = [
        {"id": 1, "name": "b1", "etf_a": "XLB", "etf_b": "B", "etf_c": "C"},
        {"id": 2, "name": "b2", "etf_a": "XLB", "etf_b": "D", "etf_c": "E"},
    ]
    for i in range(3, 14):
        buckets.append(
            {"id": i, "name": f"b{i}", "etf_a": f"A{i}", "etf_b": f"B{i}", "etf_c": f"C{i}"}
        )
    p = tmp_path / "u.yaml"
    p.write_text(yaml.safe_dump({"buckets": buckets}))
    with pytest.raises(ValueError, match="appears in bucket"):
        load_universe(p)


def test_strategy_d7_defaults_are_conservative():
    cfg = load_strategy()
    assert isinstance(cfg, StrategyConfig)
    # D7 conservative defaults (plan §9.2)
    assert cfg.delta_h_st == 0.08
    assert cfg.delta_h_lt == 0.04
    assert cfg.stop_iv_rank_low == 0.05
    assert cfg.stop_iv_rank_high == 0.08
    assert cfg.stop_crypto == 0.15
    assert cfg.roc_weight_3m == 0.5
    assert cfg.roc_weight_6m == 0.5
    assert cfg.nav_cap == 0.40
    assert cfg.nav_cap_firf == 0.20
    assert cfg.beta_ceiling == 1.35
    assert cfg.waterfall == [0.5, 0.3, 0.2]
    assert cfg.put_skew_extreme_percentile == 90.0


def test_strategy_rejects_bad_roc_weights(tmp_path: Path):
    bad = {"roc_weight_3m": 0.3, "roc_weight_6m": 0.3}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="roc_weight"):
        load_strategy(p)


def test_strategy_rejects_bad_waterfall(tmp_path: Path):
    bad = {"waterfall": [0.3, 0.3, 0.3]}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="waterfall"):
        load_strategy(p)


def test_risk_limits_d8_ceo_set_values_are_none_in_phase0():
    cfg = load_risk_limits()
    assert isinstance(cfg, RiskLimitsConfig)
    assert cfg.monthly_cash_injection == 10_000.0  # D8 literal
    # D8 CEO-SET placeholders (plan §18) — None until Phase 1
    assert cfg.starting_capital is None
    assert cfg.account_dd_stop_pct is None
    assert cfg.per_ticker_max_shares is None


def test_risk_limits_require_ceo_set_values_raises_when_none():
    cfg = load_risk_limits()
    with pytest.raises(ValueError, match="CEO-SET"):
        cfg.require_ceo_set_values()


def test_risk_limits_require_ceo_set_values_passes_when_set():
    cfg = RiskLimitsConfig(
        starting_capital=100_000.0,
        account_dd_stop_pct=0.20,
        per_ticker_max_shares={"XLB": 1000},
    )
    cfg.require_ceo_set_values()  # no raise


def test_yaml_files_have_ceo_set_markers():
    """RBY-4 acceptance: config/risk_limits.yaml carries # CEO-SET markers on D8 values."""
    text = (CONFIG_DIR / "risk_limits.yaml").read_text()
    assert "CEO-SET" in text
    for field in ("starting_capital", "account_dd_stop_pct", "per_ticker_max_shares"):
        assert field in text


def test_pimco_blocklist_contains_bond():
    assert "BOND" in PIMCO_BLOCKLIST
