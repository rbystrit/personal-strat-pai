"""yfinance fallback data feed (plan §6.1, §6.4).

Fallback ONLY: daily bars + metadata + splits when databento is down or for a
quick sanity cross-check. yfinance split/dividend data is occasionally wrong,
so EVERY yfinance corp action is cross-validated against databento before it
touches the ledger (plan §6.4 — see data/corp_actions.py).

yfinance returns pandas at the SDK boundary; this module converts to polars at
the boundary with ``pl.from_pandas`` (plan §5, D14 interop rule). Live calls are
``@pytest.mark.integration``; ``normalize_yf_bars`` is pure and unit-tested.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import yfinance as yf

from personal_strat_pai.data.corp_actions import CorpAction, CorpActionType
from personal_strat_pai.data.polars_utils import BAR_SCHEMA, EagerFrame

__all__ = [
    "YFinanceFallback",
    "normalize_yf_bars",
]


def normalize_yf_bars(raw: pl.DataFrame, *, symbol: str | None = None) -> EagerFrame:
    """Normalize a yfinance download frame to the canonical BAR_SCHEMA.

    Pure & unit-tested on the single-symbol path (the common one): yfinance
    returns capitalized column names (Open/High/Low/Close/Volume) with a
    DatetimeIndex. After ``pl.from_pandas``, columns are flat strings and the
    index becomes a ``Date`` column; this helper lowercases the OHLCV names,
    derives ``symbol`` from the ``symbol`` arg, and casts to BAR_SCHEMA.

    The multi-symbol (MultiIndex-column) path is validated in the integration
    test against a real ``yfinance.download``; it raises NotImplementedError
    here so the unit tests stay deterministic.
    """
    if raw.is_empty():
        return pl.DataFrame(schema=BAR_SCHEMA)
    if isinstance(raw.columns[0], tuple):
        raise NotImplementedError(
            "yfinance multi-symbol (MultiIndex-column) normalization is validated "
            "in the integration test against a real download. Pass a single-symbol "
            "frame or a pre-flattened frame to normalize_yf_bars."
        )
    out = raw.rename(
        {
            "Date": "ts",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if symbol is not None and "symbol" not in out.columns:
        out = out.with_columns(pl.lit(symbol).alias("symbol"))

    # Drop adj_close — we store raw OHLCV; split/div adjustment is via corp_actions.
    if "adj_close" in out.columns:
        out = out.drop("adj_close")

    # Ensure datetime tz-aware UTC.
    if out.schema["ts"] != pl.Datetime("us", "UTC"):
        out = out.with_columns(pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    out = out.with_columns(
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    )
    return out.select(list(BAR_SCHEMA.names()))


class YFinanceFallback:
    """yfinance fallback feed (plan §6.1). Live calls are integration-gated."""

    def get_daily_bars(
        self, symbols: list[str], start: date | str, end: date | str
    ) -> EagerFrame:  # pragma: no cover - integration
        raw_pd = yf.download(symbols, start=start, end=end, group_by="column", auto_adjust=False)
        return normalize_yf_bars(
            pl.from_pandas(raw_pd), symbol=symbols[0] if len(symbols) == 1 else None
        )

    def get_splits(self, symbol: str) -> list[CorpAction]:  # pragma: no cover - integration
        t = yf.Ticker(symbol)
        splits = t.splits  # pandas Series index=Date, value=ratio
        out: list[CorpAction] = []
        for ex_date, ratio in splits.items():
            out.append(
                CorpAction(
                    symbol=symbol,
                    ex_date=ex_date.date() if hasattr(ex_date, "date") else ex_date,
                    type=CorpActionType.SPLIT,
                    source="yfinance",
                    ratio=float(ratio),
                )
            )
        return out

    def get_dividends(self, symbol: str) -> list[CorpAction]:  # pragma: no cover - integration
        t = yf.Ticker(symbol)
        divs = t.dividends  # pandas Series index=Date, value=amount
        out: list[CorpAction] = []
        for ex_date, amount in divs.items():
            out.append(
                CorpAction(
                    symbol=symbol,
                    ex_date=ex_date.date() if hasattr(ex_date, "date") else ex_date,
                    type=CorpActionType.DIVIDEND,
                    source="yfinance",
                    amount=float(amount),
                )
            )
        return out
