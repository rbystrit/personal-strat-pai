"""Tests for data/iv_proxy.py — self-built IV from EOD chain, no OPRA (D2; RBY-4 acceptance)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from personal_strat_pai.data.iv_proxy import (
    EXTREME_PUT_SKEW_PERCENTILE,
    EodChainIvProvider,
    IvSnapshot,
    OpraIvProvider,
    black_scholes_price,
    compute_iv_from_chain,
    implied_vol,
    rolling_put_skew_percentile,
)


def test_implied_vol_recovers_known_iv():
    """The core D2 check: BS inversion recovers the IV used to price the option."""
    S, K, T, r, sigma = 100.0, 100.0, 30 / 365.0, 0.05, 0.25
    price = black_scholes_price(S, K, T, r, sigma, "C")
    recovered = implied_vol(price, S, K, T, r, "C")
    assert abs(recovered - sigma) < 1e-6


def test_implied_vol_put_recovers_known_iv():
    S, K, T, r, sigma = 100.0, 95.0, 30 / 365.0, 0.05, 0.30
    price = black_scholes_price(S, K, T, r, sigma, "P")
    recovered = implied_vol(price, S, K, T, r, "P")
    assert abs(recovered - sigma) < 1e-6


def test_implied_vol_rejects_zero_price():
    with pytest.raises(ValueError):
        implied_vol(0.0, 100.0, 100.0, 30 / 365.0, 0.05, "C")


def test_implied_vol_rejects_outside_no_arb_bounds():
    # call price > S is outside no-arb bounds
    with pytest.raises(ValueError, match="no-arb"):
        implied_vol(150.0, 100.0, 100.0, 30 / 365.0, 0.05, "C")


def test_compute_iv_from_chain_recovers_term_structure(option_chain_factory):
    """compute_iv_from_chain recovers iv_30d/iv_90d from a synthetic chain."""
    as_of = date(2024, 1, 2)
    chain = option_chain_factory("XLB", spot=100.0, as_of=as_of, iv_30d=0.25, iv_90d=0.22, r=0.05)
    snap = compute_iv_from_chain(chain, spot=100.0, r=0.05, as_of=as_of, symbol="XLB")
    assert isinstance(snap, IvSnapshot)
    # ATM IV should be close to the planted 30d/90d values (within skew averaging noise)
    assert abs(snap.iv_30d - 0.25) < 0.02
    assert abs(snap.iv_90d - 0.22) < 0.02
    assert snap.term_slope == pytest.approx(snap.iv_30d - snap.iv_90d)
    assert snap.backwardation is True  # iv_30d > iv_90d
    assert snap.put_skew_30d > 0.0  # OTM puts have skew bump


def test_compute_iv_from_chain_empty_raises():
    with pytest.raises(ValueError, match="no options"):
        compute_iv_from_chain(
            pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "expiration": pl.Date,
                    "strike": pl.Float64,
                    "type": pl.String,
                    "bid": pl.Float64,
                    "ask": pl.Float64,
                }
            ),
            spot=100.0,
            r=0.05,
            as_of=date(2024, 1, 2),
            symbol="XLB",
        )


def test_eod_chain_iv_provider_with_history_extreme_skew(option_chain_factory):
    as_of = date(2024, 1, 2)
    chain = option_chain_factory("XLB", spot=100.0, as_of=as_of, iv_30d=0.25, iv_90d=0.22)
    # History of low put-skew values so the current (higher) skew is >= 90th pct.
    history = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]

    def chain_loader(sym: str, d: date) -> pl.DataFrame:
        return chain

    def spot_loader(sym: str, d: date) -> float:
        return 100.0

    provider = EodChainIvProvider(
        chain_loader=chain_loader,
        spot_loader=spot_loader,
        risk_free_rate=0.05,
        put_skew_history={"XLB": history},
    )
    snap = provider.get_snapshot("XLB", as_of)
    assert snap.put_skew_percentile_12m is not None
    assert snap.put_skew_percentile_12m >= EXTREME_PUT_SKEW_PERCENTILE
    assert snap.extreme_put_skew is True


def test_opra_provider_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        OpraIvProvider()  # not wired in v1 (D2 parameterized swap target)


def test_rolling_put_skew_percentile_empty_history_neutral():
    assert rolling_put_skew_percentile([], 0.05) == 50.0


def test_rolling_put_skew_percentile_ranks_current():
    history = [0.01, 0.02, 0.03, 0.04]  # 4 values
    # current = 0.04 (max) => 100th pct
    assert rolling_put_skew_percentile(history, 0.04) == 100.0
    # current = 0.01 (min) => 25th pct
    assert rolling_put_skew_percentile(history, 0.01) == 25.0
