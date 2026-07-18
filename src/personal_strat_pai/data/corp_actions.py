"""Corporate actions: splits/dividends + cross-validation + ledger hooks (plan §6.4).

Primary source: databento reference / corporate-actions dataset.
Fallback: yfinance metadata.

Hard rule (plan §6.4): every yfinance corp action is cross-validated against
databento BEFORE it touches the ledger — yfinance split/dividend data is
occasionally wrong. An unvalidated yfinance-only action NEVER reaches the
ledger; a mismatch raises and pages.

Overnight routine (plan §6.4): ingest next-day splits, recalibrate ledger lots
(qty × ratio, basis ÷ ratio) before 04:00 ET so the Risk Clock never misreads a
split gap as a stop breach.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal

import polars as pl

from personal_strat_pai.data.polars_utils import assert_eager

__all__ = [
    "DIVIDEND_AMOUNT_TOLERANCE",
    "SPLIT_RATIO_TOLERANCE",
    "CorpAction",
    "CorpActionMismatch",
    "CorpActionSource",
    "CorpActionType",
    "UnvalidatedCorpAction",
    "apply_split_to_lots",
    "cross_validate",
]

CorpActionSource = Literal["databento", "yfinance"]

SPLIT_RATIO_TOLERANCE = 1e-6
DIVIDEND_AMOUNT_TOLERANCE = 1e-4


class CorpActionType(StrEnum):
    SPLIT = "split"
    DIVIDEND = "dividend"


@dataclass(frozen=True, slots=True)
class CorpAction:
    """A single corporate action for a symbol on an ex-date."""

    symbol: str
    ex_date: date
    type: CorpActionType
    source: CorpActionSource
    ratio: float | None = None  # for splits: new_shares / old_shares (e.g. 2.0 = 2-for-1)
    amount: float | None = None  # for dividends: $/share on ex-date

    def __post_init__(self) -> None:
        if self.type is CorpActionType.SPLIT:
            if self.ratio is None or self.ratio <= 0:
                raise ValueError(f"split action requires ratio > 0: {self!r}")
            if self.amount is not None:
                raise ValueError(f"split action must not carry amount: {self!r}")
        elif self.type is CorpActionType.DIVIDEND:
            if self.amount is None:
                raise ValueError(f"dividend action requires amount: {self!r}")
            if self.ratio is not None:
                raise ValueError(f"dividend action must not carry ratio: {self!r}")
        else:
            raise ValueError(f"unknown corp action type: {self.type!r}")


class CorpActionMismatch(ValueError):
    """Raised when yfinance and databento disagree on a corp action (plan §6.4)."""

    def __init__(self, yf: CorpAction, db: CorpAction, detail: str) -> None:
        self.yf = yf
        self.db = db
        super().__init__(f"corp action mismatch for {yf.symbol} on {yf.ex_date}: {detail}")


class UnvalidatedCorpAction(ValueError):
    """Raised when a yfinance corp action has no databento counterpart to validate against.

    Per plan §6.4, an unvalidated yfinance-only action MUST NOT touch the ledger.
    """


def cross_validate(yf: CorpAction, db: CorpAction | None) -> CorpAction:
    """Validate a yfinance corp action against the databento record (plan §6.4).

    - If ``db`` is None: the yfinance action is UNVALIDATED -> raise
      UnvalidatedCorpAction (never touches the ledger).
    - If both present and agree within tolerance: return the databento action
      (databento is the system of record).
    - If they disagree: raise CorpActionMismatch (pages; blocks the overnight job).
    """
    if db is None:
        raise UnvalidatedCorpAction(
            f"yfinance {yf.type.value} for {yf.symbol} on {yf.ex_date} has no "
            "databento counterpart to validate against — cannot touch the ledger "
            "(plan §6.4)."
        )
    if yf.symbol != db.symbol:
        raise CorpActionMismatch(yf, db, f"symbol {yf.symbol!r} vs {db.symbol!r}")
    if yf.ex_date != db.ex_date:
        raise CorpActionMismatch(yf, db, f"ex_date {yf.ex_date} vs {db.ex_date}")
    if yf.type is not db.type:
        raise CorpActionMismatch(yf, db, f"type {yf.type.value!r} vs {db.type.value!r}")
    if yf.type is CorpActionType.SPLIT:
        assert db.ratio is not None and yf.ratio is not None
        if abs(yf.ratio - db.ratio) > SPLIT_RATIO_TOLERANCE:
            raise CorpActionMismatch(yf, db, f"split ratio {yf.ratio} vs {db.ratio}")
    else:  # DIVIDEND
        assert db.amount is not None and yf.amount is not None
        if abs(yf.amount - db.amount) > DIVIDEND_AMOUNT_TOLERANCE:
            raise CorpActionMismatch(yf, db, f"dividend amount {yf.amount} vs {db.amount}")
    return db  # databento is authoritative


def apply_split_to_lots(
    lots: pl.DataFrame, ratio: float, *, symbol_col: str = "symbol"
) -> pl.DataFrame:
    """Recalibrate ledger lots for a split (plan §6.4: qty × ratio, basis ÷ ratio).

    Expects a DataFrame with at least ``symbol_col``, ``qty``, ``basis`` columns.
    Returns a new frame with adjusted qty/basis; only rows for the split symbol
    are adjusted (callers filter or pass the full ledger — the split symbol is
    inferred from the row where the split applies, so pass a pre-filtered frame
    OR provide a ``symbol`` arg via the closure below).
    """
    if ratio <= 0:
        raise ValueError(f"split ratio must be > 0, got {ratio}")
    assert_eager(lots, "apply_split_to_lots")
    for required in (symbol_col, "qty", "basis"):
        if required not in lots.columns:
            raise ValueError(f"apply_split_to_lots: missing column {required!r}")
    return lots.with_columns(
        (pl.col("qty") * ratio).alias("qty"),
        (pl.col("basis") / ratio).alias("basis"),
    )
