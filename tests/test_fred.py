"""Tests for data/fred.py — FRED normalizer + FredClient surface (live calls @integration).

Coverage:
  * ``normalize_observations`` parses FRED's raw frame (date string, percent
    string value, '.' for missing) to the canonical RATE_OBSERVATION_SCHEMA.
  * Missing observations ('.') are dropped.
  * Percent -> decimal conversion (4.55 -> 0.0455).
  * ``series`` and ``source`` columns are attached.
  * ``FredClient`` requires ``FRED_API_KEY`` and exposes the ``FredFetcher``
    protocol surface (``fetch_observations``).
  * The FRED series catalog covers SOFR + Treasury CMT (OIS proxy) + TIPS (real).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import polars as pl
import pytest

from personal_strat_pai.data.fred import (
    ALL_FRED_SERIES_IDS,
    FRED_OIS_SERIES,
    FRED_REAL_SERIES,
    FRED_SOFR_SERIES,
    FredClient,
    normalize_observations,
)
from personal_strat_pai.data.polars_utils import RATE_OBSERVATION_COLUMNS, RATE_OBSERVATION_SCHEMA


def test_normalize_empty_returns_empty_with_schema():
    out = normalize_observations(pl.DataFrame(), series_id="SOFR")
    assert out.is_empty()
    assert list(out.columns) == list(RATE_OBSERVATION_COLUMNS)
    assert out.schema == RATE_OBSERVATION_SCHEMA


def test_normalize_parses_date_and_converts_percent_to_decimal():
    raw = pl.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "value": ["4.55", "5.20"],
            "realtime_start": ["2024-01-02", "2024-01-03"],
            "realtime_end": ["9999-12-31", "9999-12-31"],
        }
    )
    out = normalize_observations(raw, series_id="DGS10")
    assert list(out.columns) == list(RATE_OBSERVATION_COLUMNS)
    assert out.schema["ts"] == pl.Datetime("us", "UTC")
    assert out.schema["rate"] == pl.Float64
    assert out["series"].to_list() == ["DGS10", "DGS10"]
    assert out["source"].to_list() == ["FRED", "FRED"]
    # 4.55% -> 0.0455 ; 5.20% -> 0.052
    assert out["rate"].to_list() == pytest.approx([0.0455, 0.052])
    assert out["ts"].to_list() == [
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    ]


def test_normalize_drops_missing_observations():
    # FRED uses '.' for missing values (holidays, pre-inception dates).
    raw = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [".", "4.55", "."],
            "realtime_start": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "realtime_end": ["9999-12-31", "9999-12-31", "9999-12-31"],
        }
    )
    out = normalize_observations(raw, series_id="SOFR")
    assert out.height == 1
    assert out["ts"].to_list() == [datetime(2024, 1, 2, tzinfo=UTC)]
    assert out["rate"].to_list() == pytest.approx([0.0455])


def test_normalize_handles_float_value_column():
    # Some SDK paths may already have value as float (not str).
    raw = pl.DataFrame(
        {
            "date": ["2024-01-02"],
            "value": [4.55],
        }
    )
    out = normalize_observations(raw, series_id="DGS2")
    assert out["rate"].to_list() == pytest.approx([0.0455])
    assert out["series"].to_list() == ["DGS2"]


def test_normalize_rejects_frame_without_value_column():
    raw = pl.DataFrame({"date": ["2024-01-02"]})
    with pytest.raises(ValueError, match="value"):
        normalize_observations(raw, series_id="SOFR")


def test_fred_client_requires_key():
    client = FredClient(api_key=None)
    old = os.environ.pop("FRED_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            _ = client.api_key
    finally:
        if old is not None:
            os.environ["FRED_API_KEY"] = old


def test_fred_client_exposes_fred_fetcher_surface():
    client = FredClient(api_key="dummy-key")
    assert hasattr(client, "fetch_observations")
    assert callable(client.fetch_observations)


def test_catalog_covers_sofr_ois_proxy_and_real_rates():
    # OIS proxy tenors cover 3M..30Y (8 tenors).
    assert len(FRED_OIS_SERIES) == 8
    assert FRED_OIS_SERIES[2.0] == "DGS2"
    assert FRED_OIS_SERIES[10.0] == "DGS10"
    # Real rates cover 5/10/30y (3 TIPS series).
    assert FRED_REAL_SERIES == {5.0: "DFII5", 10.0: "DFII10", 30.0: "DFII30"}
    # SOFR is the overnight short-rate series.
    assert FRED_SOFR_SERIES == "SOFR"
    # ALL_FRED_SERIES_IDS is the unique, sorted set of all catalog series —
    # OIS and real share 5/10/30y tenors but are DIFFERENT series, so all 12
    # series ids appear (1 SOFR + 8 DGS + 3 DFII = 12).
    expected = sorted({FRED_SOFR_SERIES, *FRED_OIS_SERIES.values(), *FRED_REAL_SERIES.values()})
    assert expected == ALL_FRED_SERIES_IDS
    assert len(ALL_FRED_SERIES_IDS) == 12
