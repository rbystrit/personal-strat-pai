"""Tests for data/rates.py — SOFR/OIS curve (D3)."""

from __future__ import annotations

from datetime import date

import pytest

from personal_strat_pai.data.rates import (
    DEFAULT_RISK_FREE_RATE,
    DatabentoRatesProvider,
    RatesCurve,
    SyntheticRatesProvider,
    risk_free_continuous,
)


def test_synthetic_provider_returns_curve():
    p = SyntheticRatesProvider()
    curve = p.get_curve(date(2024, 1, 2))
    assert isinstance(curve, RatesCurve)
    assert curve.ois_slope_2y10y is not None
    # normal curve: 10y > 2y => positive slope
    assert curve.ois_slope_2y10y > 0.0


def test_ois_slope_inverted_when_10y_below_2y():
    curve = RatesCurve(
        as_of=date(2024, 1, 2),
        tenors_years=[2.0, 10.0],
        ois_rates=[0.05, 0.04],  # inverted: 10y < 2y
        sofr_rates=[0.05, 0.04],
    )
    assert curve.ois_slope_2y10y is not None
    assert curve.ois_slope_2y10y < 0.0


def test_ois_slope_none_when_tenors_missing():
    curve = RatesCurve(as_of=date(2024, 1, 2))
    assert curve.ois_slope_2y10y is None


def test_risk_free_continuous_uses_curve():
    p = SyntheticRatesProvider(ois=[0.04, 0.04, 0.04, 0.045, 0.05, 0.052, 0.055])
    r = risk_free_continuous(p, date(2024, 1, 2), tenor_years=1.0)
    assert r == pytest.approx(0.04)


def test_risk_free_continuous_falls_back_when_curve_empty():
    p = SyntheticRatesProvider(ois=[], sofr=[], tenors=[])
    r = risk_free_continuous(p, date(2024, 1, 2))
    assert r == DEFAULT_RISK_FREE_RATE


def test_databento_provider_requires_key():
    p = DatabentoRatesProvider(api_key=None)
    with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
        p.get_curve(date(2024, 1, 2))
