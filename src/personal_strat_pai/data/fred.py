"""FRED (Federal Reserve Economic Data) — historical SOFR / OIS / real rates.

CEO directive 2026-07-19 (RBY-4 rejection 2 + "For FRED, follow the same
download once policy"): FRED is the primary source for **historical** SOFR /
OIS / real rates used in backtesting. The FIRF (plan §6.3) reads the curve at
each backtest date; the no-double-download caching policy (CEO 2026-07-18)
applies in full: every FRED observation is fetched once, cached in the parquet
store (Object Storage in prod), and re-read on subsequent requests.

Series catalog (mapped to the FIRF curve's tenor structure):

  * **SOFR** (overnight secured financing rate) — ``SOFR`` (since 2018-04-09).
  * **OIS proxy** = Treasury constant-maturity yields (daily, percent). FRED
    does not publish OIS swap rates directly; Treasury CMT yields are the
    standard risk-free proxy used in backtesting (plan §6.3 note). Databento's
    SOFR/OIS forward curve remains the live system of record (P0-3, D3).
    Series: ``DGS3MO``, ``DGS6MO``, ``DGS1``, ``DGS2``, ``DGS5``, ``DGS10``,
    ``DGS20``, ``DGS30``.
  * **Real rates** (TIPS yields, daily, percent) — plan §6.3 FIRF "real rates
    surge" trigger. Series: ``DFII5``, ``DFII10``, ``DFII30``.

All live calls are gated behind ``FRED_API_KEY`` (free at
https://api.stlouisfed.org) and ``@pytest.mark.integration`` in tests. The pure
normalization helper (``normalize_observations``) is unit-tested with synthetic
raw frames so the data layer is exercised in CI without spend.

FRED publishes rates as percent (e.g. ``4.55`` = 4.55%); this module normalizes
to **decimal** (``0.0455``) at the boundary so downstream code (Black-Scholes,
FIRF, IV proxy) never has to convert.

``FredClient`` implements the ``FredFetcher`` protocol (data/rates.py
``FredRateRepo``) via ``fetch_observations`` so the caching repo can fetch only
the missing ranges without re-downloading.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any, Protocol

import polars as pl

from personal_strat_pai.data.polars_utils import RATE_OBSERVATION_SCHEMA, EagerFrame

__all__ = [
    "ALL_FRED_SERIES_IDS",
    "FRED_API_BASE",
    "FRED_OIS_SERIES",
    "FRED_REAL_SERIES",
    "FRED_SOFR_SERIES",
    "FredClient",
    "FredFetcher",
    "normalize_observations",
]

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

# SOFR (overnight secured financing rate) — the short end of the curve.
FRED_SOFR_SERIES: str = "SOFR"

# OIS proxy = Treasury constant-maturity yields (daily, percent). FRED does not
# publish OIS swap rates directly; Treasury CMT yields are the standard
# risk-free proxy used in backtesting (plan §6.3 note). Databento's SOFR/OIS
# forward curve remains the live system of record (P0-3, D3).
FRED_OIS_SERIES: dict[float, str] = {
    0.25: "DGS3MO",
    0.5: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    5.0: "DGS5",
    10.0: "DGS10",
    20.0: "DGS20",
    30.0: "DGS30",
}

# Real rates (TIPS yields, daily, percent) — plan §6.3 FIRF "real rates surge" trigger.
# Note: real-rate tenors (5/10/30y) overlap OIS tenors — they're a SEPARATE series
# per tenor (DFIIx), not a replacement. The provider routes each observation to the
# right curve list by series id, so the tenor overlap is intentional and correct.
FRED_REAL_SERIES: dict[float, str] = {
    5.0: "DFII5",
    10.0: "DFII10",
    30.0: "DFII30",
}

# All FRED series ids the caching repo pulls in one bootstrap. Built from the
# three separate dicts (SOFR + OIS proxy + real) so tenor overlap between OIS
# and real at 5/10/30y does NOT drop either series.
ALL_FRED_SERIES_IDS: list[str] = sorted(
    {FRED_SOFR_SERIES, *FRED_OIS_SERIES.values(), *FRED_REAL_SERIES.values()}
)


def _require_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED needs FRED_API_KEY (free at https://api.stlouisfed.org). "
            "Set it in .env or OCI Vault. Tests use synthetic data; live calls "
            "are @pytest.mark.integration."
        )
    return key


def normalize_observations(raw: pl.DataFrame, *, series_id: str) -> pl.DataFrame:
    """Normalize a raw FRED observations frame to the canonical schema.

    Pure & unit-tested. FRED returns ``{date: 'YYYY-MM-DD', value: '4.55'}`` per
    observation (value as a percent STRING; ``'.'`` means missing). This helper:

      * parses the date to a UTC datetime,
      * casts the value to float64 and converts percent -> decimal,
      * drops missing observations (value == ``'.'``),
      * attaches the series id as the ``series`` column and ``"FRED"`` as the
        ``source`` column,
      * selects canonical columns in order: ``(series, ts, rate, source)``.

    The input is expected to already be polars (the FRED client converts the
    JSON response at the boundary).
    """
    if raw.is_empty():
        return pl.DataFrame(schema=RATE_OBSERVATION_SCHEMA)
    out = raw
    if "date" in out.columns and "ts" not in out.columns:
        out = out.rename({"date": "ts"})
    # Parse ts (YYYY-MM-DD string) -> UTC datetime.
    if out.schema["ts"] != pl.Datetime("us", "UTC"):
        out = out.with_columns(
            pl.col("ts")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime("us"), format="%Y-%m-%d", strict=False)
            .dt.replace_time_zone("UTC")
            .alias("ts")
        )
    # Coerce value -> float; FRED uses '.' for missing. Drop missing rows.
    if "value" not in out.columns:
        raise ValueError("FRED raw frame missing 'value' column")
    out = out.with_columns(pl.col("value").cast(pl.Utf8).alias("value_str"))
    out = out.filter(pl.col("value_str") != ".")
    out = out.with_columns(pl.col("value_str").cast(pl.Float64, strict=False).alias("rate_pct"))
    out = out.filter(pl.col("rate_pct").is_not_null())
    # Percent -> decimal.
    out = out.with_columns((pl.col("rate_pct") / 100.0).alias("rate"))
    out = out.with_columns(
        pl.lit(series_id).alias("series"),
        pl.lit("FRED").alias("source"),
    )
    return out.select(list(RATE_OBSERVATION_SCHEMA.names())).cast(RATE_OBSERVATION_SCHEMA)


class FredFetcher(Protocol):
    """Rate-fetcher protocol — fetch FRED observations for one series over a range.

    Implementations return a normalized ``EagerFrame`` conforming to
    ``RATE_OBSERVATION_SCHEMA`` for the requested series over ``[start, end)``.
    The caching repo NEVER calls this for a range already in the store.
    """

    def fetch_observations(
        self,
        series_id: str,
        start: date | str,
        end: date | str,
    ) -> EagerFrame: ...


class FredClient:
    """FRED API client. Live calls integration-gated behind the API key.

    Implements the ``FredFetcher`` protocol via ``fetch_observations`` so the
    caching repo (``FredRateRepo``) can fetch only missing ranges without
    re-downloading. Uses stdlib ``urllib`` — no extra dependency.

    FRED's ``observation_end`` is inclusive; we pass ``end`` as-is and rely on
    the canonical ``[start, end)`` convention downstream (the upsert dedupes by
    ``(series, ts)`` so the at-most-one-day overshoot is harmless and useful for
    the next forward-gap computation).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = FRED_API_BASE,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        return _require_key(self._api_key)

    def fetch_observations(
        self,
        series_id: str,
        start: date | str,
        end: date | str,
    ) -> EagerFrame:  # pragma: no cover - integration
        """``FredFetcher`` protocol impl — GET /series/observations over [start, end)."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": _to_iso(start),
            "observation_end": _to_iso(end),
        }
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return _payload_to_frame(payload, series_id)


def _payload_to_frame(payload: dict[str, Any], series_id: str) -> pl.DataFrame:
    """Convert the FRED JSON payload to a normalized polars frame at the boundary."""
    obs = payload.get("observations", [])
    if not obs:
        return pl.DataFrame(schema=RATE_OBSERVATION_SCHEMA)
    # FRED observation fields: date (str), value (str), realtime_start, realtime_end.
    raw = pl.DataFrame(
        obs,
        schema={
            "date": pl.Utf8,
            "value": pl.Utf8,
            "realtime_start": pl.Utf8,
            "realtime_end": pl.Utf8,
        },
    )
    return normalize_observations(raw, series_id=series_id)


def _to_iso(d: date | str | datetime) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()
