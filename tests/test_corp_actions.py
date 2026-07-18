"""Tests for data/corp_actions.py — splits/dividends + cross-validation (§6.4)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from personal_strat_pai.data.corp_actions import (
    CorpAction,
    CorpActionMismatch,
    CorpActionType,
    UnvalidatedCorpAction,
    apply_split_to_lots,
    cross_validate,
)


def _split(symbol: str, ex_date: date, ratio: float, source="databento") -> CorpAction:
    return CorpAction(
        symbol=symbol, ex_date=ex_date, type=CorpActionType.SPLIT, source=source, ratio=ratio
    )


def _div(symbol: str, ex_date: date, amount: float, source="databento") -> CorpAction:
    return CorpAction(
        symbol=symbol, ex_date=ex_date, type=CorpActionType.DIVIDEND, source=source, amount=amount
    )


def test_split_action_validates_ratio():
    with pytest.raises(ValueError):
        _split("XLB", date(2024, 1, 2), ratio=0.0)
    with pytest.raises(ValueError):
        _split("XLB", date(2024, 1, 2), ratio=-1.0)


def test_dividend_action_validates_amount():
    with pytest.raises(ValueError):
        CorpAction(
            symbol="XLB", ex_date=date(2024, 1, 2), type=CorpActionType.DIVIDEND, source="databento"
        )


def test_cross_validate_agree_returns_databento_authoritative():
    yf = _split("XLB", date(2024, 1, 2), 2.0, source="yfinance")
    db = _split("XLB", date(2024, 1, 2), 2.0, source="databento")
    out = cross_validate(yf, db)
    assert out is db


def test_cross_validate_mismatch_raises():
    yf = _split("XLB", date(2024, 1, 2), 2.0, source="yfinance")
    db = _split("XLB", date(2024, 1, 2), 3.0, source="databento")
    with pytest.raises(CorpActionMismatch):
        cross_validate(yf, db)


def test_cross_validate_date_mismatch_raises():
    yf = _split("XLB", date(2024, 1, 2), 2.0, source="yfinance")
    db = _split("XLB", date(2024, 1, 3), 2.0, source="databento")
    with pytest.raises(CorpActionMismatch):
        cross_validate(yf, db)


def test_cross_validate_unvalidated_yfinance_only_raises():
    """A yfinance-only action with no databento counterpart must NOT touch the ledger (§6.4)."""
    yf = _split("XLB", date(2024, 1, 2), 2.0, source="yfinance")
    with pytest.raises(UnvalidatedCorpAction):
        cross_validate(yf, None)


def test_cross_validate_dividend_agree():
    yf = _div("XLB", date(2024, 1, 2), 0.50, source="yfinance")
    db = _div("XLB", date(2024, 1, 2), 0.50, source="databento")
    out = cross_validate(yf, db)
    assert out is db


def test_cross_validate_dividend_amount_mismatch_raises():
    yf = _div("XLB", date(2024, 1, 2), 0.50, source="yfinance")
    db = _div("XLB", date(2024, 1, 2), 0.60, source="databento")
    with pytest.raises(CorpActionMismatch):
        cross_validate(yf, db)


def test_apply_split_to_lots_qty_times_ratio_basis_div_ratio():
    """plan §6.4: qty × ratio, basis ÷ ratio."""
    lots = pl.DataFrame({"symbol": ["XLB", "XLB"], "qty": [100.0, 50.0], "basis": [50.0, 60.0]})
    out = apply_split_to_lots(lots, ratio=2.0)
    assert out["qty"].to_list() == [200.0, 100.0]
    assert out["basis"].to_list() == [25.0, 30.0]


def test_apply_split_rejects_bad_ratio():
    lots = pl.DataFrame({"symbol": ["XLB"], "qty": [100.0], "basis": [50.0]})
    with pytest.raises(ValueError):
        apply_split_to_lots(lots, ratio=0.0)


def test_apply_split_rejects_missing_column():
    lots = pl.DataFrame({"symbol": ["XLB"], "qty": [100.0]})
    with pytest.raises(ValueError, match="missing column"):
        apply_split_to_lots(lots, ratio=2.0)
