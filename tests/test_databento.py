"""Tests for data/databento.py — pure normalizer (live calls are @pytest.mark.integration)."""

from __future__ import annotations

import os
from datetime import datetime

import polars as pl
import pytest

from personal_strat_pai.data.databento import (
    DatabentoClient,
    normalize_bars,
    normalize_option_chain,
)
from personal_strat_pai.data.polars_utils import BAR_SCHEMA


def test_normalize_bars_empty_returns_empty_with_schema():
    out = normalize_bars(pl.DataFrame())
    assert out.is_empty()
    assert list(out.columns) == list(BAR_SCHEMA.names())


def test_normalize_bars_maps_ts_event_and_casts():
    raw = pl.DataFrame(
        {
            "symbol": ["XLB", "XLB"],
            "ts_event": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1_000_000, 1_100_000],
        }
    )
    out = normalize_bars(raw)
    assert list(out.columns) == list(BAR_SCHEMA.names())
    assert out.schema["ts"] == pl.Datetime("us", "UTC")
    assert out.schema["volume"] == pl.Int64
    assert out["symbol"].to_list() == ["XLB", "XLB"]


def test_normalize_bars_maps_ticker_to_symbol():
    raw = pl.DataFrame(
        {
            "ticker": ["XLY"],
            "ts_event": [datetime(2024, 1, 2)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000],
        }
    )
    out = normalize_bars(raw)
    assert "symbol" in out.columns
    assert "ticker" not in out.columns
    assert out["symbol"].to_list() == ["XLY"]


def test_databento_client_requires_key():
    client = DatabentoClient(api_key=None)
    # accessing .api_key property raises when no key and no env var
    old = os.environ.pop("DATABENTO_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
            _ = client.api_key
    finally:
        if old is not None:
            os.environ["DATABENTO_API_KEY"] = old


def test_normalize_option_chain_empty_returns_empty():
    out = normalize_option_chain(pl.DataFrame())
    assert out.is_empty()
