"""Tests for data/yfinance.py — pure normalizer (live calls are @pytest.mark.integration)."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from personal_strat_pai.data.polars_utils import BAR_SCHEMA
from personal_strat_pai.data.yfinance import normalize_yf_bars


def test_normalize_yf_bars_empty_returns_empty():
    out = normalize_yf_bars(pl.DataFrame())
    assert out.is_empty()
    assert list(out.columns) == list(BAR_SCHEMA.names())


def test_normalize_yf_bars_single_symbol_path():
    raw = pl.DataFrame(
        {
            "Date": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Adj Close": [100.5, 101.5],
            "Volume": [1_000_000, 1_100_000],
        }
    )
    out = normalize_yf_bars(raw, symbol="XLB")
    assert list(out.columns) == list(BAR_SCHEMA.names())
    assert "adj_close" not in out.columns
    assert out["symbol"].unique().to_list() == ["XLB"]
    assert out.schema["ts"] == pl.Datetime("us", "UTC")
    assert out.schema["volume"] == pl.Int64


# The multi-symbol (MultiIndex-column) normalization path is validated in the
# integration test against a real ``yfinance.download`` — current polars does
# not construct tuple-column DataFrames from the dict constructor, so the
# defensive NotImplementedError branch in normalize_yf_bars is exercised there.
