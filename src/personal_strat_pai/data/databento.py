"""databento primary data feed (plan §6.1, §6.3, §6.4; D3).

Primary source for:
  - historical + live bars (daily + minute) across the ~45-ETF universe (DBEQ),
  - SOFR/OIS forward curve (D3, delegated to data/rates.py),
  - corporate actions / splits (§6.4, system of record).

IV (plan §6.2) is NOT sourced from databento here. Per CEO directive 2026-07-18
(superseding design decision D2): for **backtesting**, IV is proxied by HV
(historical volatility) computed from the bars we already ingest — no options
data spend (OPRA is too expensive; the EOD-chain self-built IV path was
dropped). For **live/paper**, IV/options come via IBKR (data/iv_proxy.py
``IbkrIvProvider``). So this module no longer fetches or normalizes option
chains.

All live calls are gated behind ``DATABENTO_API_KEY`` and ``@pytest.mark.integration``
in tests. The pure normalization helper (``normalize_bars``) is unit-tested
with synthetic raw frames so the data layer is exercised in CI without spend.

databento returns pandas DataFrames at the SDK boundary; this module converts to
polars at the boundary with ``pl.from_pandas`` (plan §5, D14 interop rule).

``DatabentoClient`` implements the ``BarFetcher`` protocol (data/repo.py) so the
caching repo can compose it without re-downloading cached ranges (CEO directive
2026-07-18: no piece of data downloaded twice).
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import databento
import pandas as pd
import polars as pl

from personal_strat_pai.data.corp_actions import CorpAction
from personal_strat_pai.data.polars_utils import BAR_SCHEMA, EagerFrame
from personal_strat_pai.data.store import BarKind

__all__ = [
    "DATABENTO_DBEQ_DAILY",
    "DATABENTO_DBEQ_MINUTE",
    "DatabentoClient",
    "normalize_bars",
]

DATABENTO_DBEQ_DAILY = "DBEQ-BARS-1D"
DATABENTO_DBEQ_MINUTE = "DBEQ-BARS-1M"


def _require_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError(
            "databento needs DATABENTO_API_KEY (plan §6.1). Set it in .env or "
            "OCI Vault. Tests use synthetic data; live calls are @pytest.mark.integration."
        )
    return key


def normalize_bars(raw: pl.DataFrame, *, bar_kind: str = "daily") -> pl.DataFrame:
    """Normalize a raw databento bar frame to the canonical BAR_SCHEMA.

    Pure & unit-tested. Maps databento's ``ts_event``/``open``/... fields to the
    canonical schema, casts types, and selects canonical columns in order. The
    input is expected to already be polars (databento pandas -> pl.from_pandas
    happens at the SDK boundary in ``DatabentoClient``).
    """
    if raw.is_empty():
        return pl.DataFrame(schema=BAR_SCHEMA)
    rename_map: dict[str, str] = {}
    if "ts_event" in raw.columns and "ts" not in raw.columns:
        rename_map["ts_event"] = "ts"
    if "symbol" in raw.columns and "ticker" not in raw.columns:
        pass  # already symbol
    elif "ticker" in raw.columns:
        rename_map["ticker"] = "symbol"
    out = raw.rename(rename_map)
    # Ensure datetime tz-aware UTC
    if "ts" in out.columns and out.schema["ts"] != pl.Datetime("us", "UTC"):
        out = out.with_columns(pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    # Cast OHLCV
    out = out.with_columns(
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    )
    return out.select(list(BAR_SCHEMA.names()))


class DatabentoClient:
    """databento client (plan §6.1). Live calls integration-gated behind the API key.

    Implements the ``BarFetcher`` protocol (data/repo.py) via ``fetch_bars`` so
    the caching repo can fetch only missing ranges without re-downloading.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        daily_dataset: str = DATABENTO_DBEQ_DAILY,
        minute_dataset: str = DATABENTO_DBEQ_MINUTE,
    ) -> None:
        self._api_key = api_key
        self.daily_dataset = daily_dataset
        self.minute_dataset = minute_dataset

    @property
    def api_key(self) -> str:
        return _require_key(self._api_key)

    def _historical(self) -> Any:  # pragma: no cover - integration
        return databento.Historical(key=self.api_key)

    def fetch_bars(
        self,
        symbols: list[str],
        start: date | str,
        end: date | str,
        *,
        kind: BarKind = "daily",
    ) -> EagerFrame:  # pragma: no cover - integration
        """``BarFetcher`` protocol impl — dispatch to daily/minute by kind."""
        dataset = self.daily_dataset if kind == "daily" else self.minute_dataset
        raw = self._fetch_range(dataset, symbols, start, end)
        return normalize_bars(_to_polars(raw), bar_kind=kind)

    def get_daily_bars(
        self, symbols: list[str], start: date | str, end: date | str
    ) -> EagerFrame:  # pragma: no cover - integration
        raw = self._fetch_range(self.daily_dataset, symbols, start, end)
        return normalize_bars(_to_polars(raw), bar_kind="daily")

    def get_minute_bars(
        self, symbols: list[str], start: date | str, end: date | str
    ) -> EagerFrame:  # pragma: no cover - integration
        raw = self._fetch_range(self.minute_dataset, symbols, start, end)
        return normalize_bars(_to_polars(raw), bar_kind="minute")

    def get_corp_actions(
        self, symbols: list[str], start: date | str, end: date | str
    ) -> list[CorpAction]:  # pragma: no cover - integration
        raise NotImplementedError(
            "databento corporate-actions fetch (plan §6.4) lands with the "
            "overnight corp-action routine (P0-2/P0-4). Use synthetic actions "
            "for unit tests."
        )

    def _fetch_range(
        self, dataset: str, symbols: list[str], start: date | str, end: date | str
    ) -> object:  # pragma: no cover - integration
        hist = self._historical()
        # databento's timeseries.get_range returns a pandas DataFrame.
        # Exact kwargs (schema, stype_in, symbols) validated in the integration test.
        return hist.timeseries.get_range(
            dataset=dataset,
            symbols=symbols,
            start=_to_iso(start),
            end=_to_iso(end),
        )


def _to_polars(raw: object) -> pl.DataFrame:
    """Convert the databento SDK return (pandas) to polars at the boundary (D14)."""
    if isinstance(raw, pl.DataFrame):
        return raw
    if isinstance(raw, pd.DataFrame):
        return pl.from_pandas(raw)
    raise TypeError(f"unexpected databento return type: {type(raw)!r}")


def _to_iso(d: date | str | datetime) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.isoformat()
    return d.isoformat()
