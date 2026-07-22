"""HIFO tax-lot ledger (plan §8, §9; brief §1).

Highest-In, First-Out specific tax-lot identification, enforced globally. On a
sell, lots are selected **highest-cost-first** to maximize harvested losses (a
higher cost basis => a larger realized loss when the market price has fallen,
which is what we want for tax-loss harvesting). Realized gain/loss is split
into **short-term** (holding period < 365 days) vs **long-term** (>= 365 days)
**at the lot level** — each closed lot's gain is classified by *that lot's*
holding period, not by an aggregate holding-period blend. This matters because
the brief's tax hurdles (38.8% ST / 23.8% LT) are applied per-lot at
realization; an aggregate blend would mis-state the tax drag.

Pure functions, property-tested with hypothesis (plan §16 acceptance criteria):

  * **qty conservation** — the sum of per-lot closed qty equals the requested
    sell qty (or the total available qty if the sell is capped).
  * **proceeds = Σ per-lot proceeds** — no proceeds are created or lost across
    the lot selection.
  * **no lot is double-counted** — each open lot is selected at most once per
    sell; a lot's `closed_qty` cannot exceed its `qty`.

The atomicity boundary lives in ``state/nosql.py``: the ledger calls
``update_if_condition`` per lot to close it under a "status == open and
qty == expected_qty" fence. Two concurrent sellers against the same lot
cannot both close it. This module is the pure-math selection + realization;
the NoSQL write path is in ``LedgerRepo`` below, which composes the pure
functions with the conditional-write fence.

ST/LT boundary (brief §1): holding period < 365 days = ST; >= 365 days = LT.
The 38.8% (ST) / 23.8% (LT) hurdle rates are a config concern (config/strategy.yaml
is the home for the ΔH hurdles); this module only classifies and records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from personal_strat_pai.state.nosql import (
    ConditionalCheckFailed,
    NoSqlStore,
)

__all__ = [
    "HoldingClass",
    "LedgerError",
    "LedgerRepo",
    "LotClosed",
    "LotId",
    "LotStatus",
    "RealizedLot",
    "SellRequest",
    "SellResult",
    "TaxLot",
    "close_lot",
    "hifo_select",
    "holding_class",
    "holding_days",
    "realize_lot",
    "split_proceeds",
    "st_lt_split",
]

LotId = str

# Brief §1: holding period < 365 days = short-term; >= 365 days = long-term.
ST_THRESHOLD_DAYS: int = 365


class HoldingClass(StrEnum):
    """Realized holding-period classification (brief §1).

    ST (< 365 days) taxed at the 38.8% hurdle (35% ordinary + 20% LTCG base +
    3.8% NIIT); LT (>= 365 days) at the 23.8% hurdle (20% + 3.8% NIIT). The
    rates themselves are a config concern; this enum is the per-lot tag the
    realized-P&L record carries so the backtest's tax-drag attribution can
    split ST vs LT gains.
    """

    SHORT_TERM = "ST"
    LONG_TERM = "LT"


class LotStatus(StrEnum):
    """The lifecycle state of a tax lot (plan §8 ``tax_lots.status``).

    ``open`` — the lot holds an unclosed position; eligible for HIFO selection.
    ``closed`` — fully realized; the lot is no longer selectable.
    ``washed`` — closed at a loss AND the triplet slot is within its 30-day
    wash-sale restricted window (brief §1, plan §8 ``triplet_state``). A washed
    lot's loss is still realized for accounting purposes; the wash-sale rule is
    enforced downstream by the triplet machine (the slot is RESTRICTED so the
    loss is not "repurchased" inside 30 days).
    """

    OPEN = "open"
    CLOSED = "closed"
    WASHED = "washed"


@dataclass(frozen=True, slots=True)
class TaxLot:
    """An open or partially-closed tax lot (plan §8 ``tax_lots`` table).

    HIFO selection sorts by ``cost_basis_per_share`` descending; ties break by
    ``acquired_at`` descending (newer first — under HIFO, a same-cost newer
    lot is preferred because it is more likely to be ST, which harvests a
    loss against the higher ST hurdle). Frozen + slots so the pure functions
    can be shared across threads without copying.
    """

    lot_id: LotId
    account: str
    bucket_id: int
    ticker: str
    triplet_slot: str  # "A" | "B" | "C" — plan §8
    qty: float  # original quantity
    closed_qty: float  # qty already closed (0 for a fresh open lot)
    cost_basis_per_share: float
    acquired_at: datetime
    status: LotStatus = LotStatus.OPEN
    wash_immunity_until: datetime | None = None  # set when the slot is wash-restricted

    @property
    def open_qty(self) -> float:
        """Qty still selectable for HIFO closure."""
        return self.qty - self.closed_qty

    @property
    def is_open(self) -> bool:
        return self.status == LotStatus.OPEN and self.open_qty > 0


@dataclass(frozen=True, slots=True)
class RealizedLot:
    """A single lot's realization record — one per closed lot slice (plan §8 ``realized_pnl``).

    The ST/LT class is computed from the lot's holding period at realization
    time, *not* from an aggregate portfolio holding period. ``holding_days``
    is ``(as_of - acquired_at).days`` clamped to >= 0; the boundary is
    ``ST_THRESHOLD_DAYS`` (365).
    """

    lot_id: LotId
    closed_qty: float
    fill_price: float
    cost_basis_per_share: float
    proceeds: float
    cost: float
    gain: float  # proceeds - cost; negative = harvested loss
    holding_days: int
    holding_class: HoldingClass
    closed_at: datetime
    was_loss: bool
    triplet_slot: str


@dataclass(frozen=True, slots=True)
class SellRequest:
    """A sell order's parameters — what the caller wants to realize."""

    account: str
    bucket_id: int
    ticker: str
    qty: float  # requested sell qty
    fill_price: float  # the fill price (per share) — same for all lots in this sell
    as_of: datetime
    is_wash_liquidation: bool = False  # set when the sell is a triplet stop-out


@dataclass(slots=True)
class SellResult:
    """The outcome of a HIFO sell — selected lots, realized P&L, and any unfilled qty.

    ``unfilled_qty`` is non-zero only when the open lots' total qty is less than
    the requested sell qty (an under-filled sell). The caller treats this as a
    partial fill; the NoSQL write path records exactly the closed slices.
    """

    realized: list[RealizedLot] = field(default_factory=list)
    unfilled_qty: float = 0.0
    total_proceeds: float = 0.0
    total_cost: float = 0.0
    total_gain: float = 0.0
    st_gain: float = 0.0  # sum of ST gains (negative = harvested ST loss)
    lt_gain: float = 0.0  # sum of LT gains (negative = harvested LT loss)
    harvested_loss: float = 0.0  # sum of |gain| for losing lots only (>= 0)

    @property
    def filled_qty(self) -> float:
        return sum(r.closed_qty for r in self.realized)


@dataclass(frozen=True, slots=True)
class LotClosed:
    """The new lot state to persist after a slice is closed (LedgerRepo write path)."""

    lot_id: LotId
    new_closed_qty: float
    new_status: LotStatus
    realized: RealizedLot


class LedgerError(Exception):
    """Raised when the HIFO selection or realization is internally inconsistent."""


# --- Pure helpers --- #


def holding_days(lot: TaxLot, as_of: datetime) -> int:
    """The lot's holding period at ``as_of``, in whole days (>= 0).

    Brief §1: holding period < 365 days = ST; >= 365 days = LT. The boundary
    is computed from the lot's ``acquired_at``; a same-day close (a buy and a
    sell on the same calendar day) counts as 0 days = ST.
    """
    delta = as_of - lot.acquired_at
    return max(delta.days, 0)


def holding_class(lot: TaxLot, as_of: datetime) -> HoldingClass:
    """ST if holding period < 365 days, else LT (brief §1)."""
    return (
        HoldingClass.SHORT_TERM
        if holding_days(lot, as_of) < ST_THRESHOLD_DAYS
        else HoldingClass.LONG_TERM
    )


def st_lt_split(realized: list[RealizedLot]) -> tuple[float, float]:
    """Aggregate ST and LT gains across a list of realized lots.

    Returns ``(st_gain, lt_gain)`` where each value is the sum of `gain` for
    the realized lots in that class (negative for harvested losses).
    """
    st = sum(r.gain for r in realized if r.holding_class == HoldingClass.SHORT_TERM)
    lt = sum(r.gain for r in realized if r.holding_class == HoldingClass.LONG_TERM)
    return st, lt


def split_proceeds(realized: list[RealizedLot]) -> tuple[float, float, float]:
    """Returns ``(total_proceeds, total_cost, total_gain)`` across realized lots.

    The invariant ``total_proceeds == fill_price * sum(closed_qty)`` is checked
    by the hypothesis property tests (proceeds = Σ per-lot proceeds).
    """
    proceeds = sum(r.proceeds for r in realized)
    cost = sum(r.cost for r in realized)
    gain = sum(r.gain for r in realized)
    return proceeds, cost, gain


def hifo_select(open_lots: list[TaxLot], sell_qty: float) -> list[tuple[TaxLot, float]]:
    """Pure HIFO selection — highest cost-basis first, returns ``(lot, close_qty)`` pairs.

    Sort order (brief §1 "Highest-In, First-Out"):
      1. ``cost_basis_per_share`` descending — the highest-cost lot is selected
         first. This maximizes the harvested loss on a down-move (a higher
         basis realizes a larger loss at the same fill price) and minimizes the
         realized gain on an up-move (a higher basis realizes a smaller gain).
         Both effects are tax-favorable.
      2. ``acquired_at`` descending (tie-break) — under HIFO, a same-cost newer
         lot is preferred because it is more likely ST, which harvests a loss
         against the higher 38.8% ST hurdle (or, on a gain, defers into the
         lower 23.8% LT classification only once it ages past 365 days).
      3. ``lot_id`` ascending (final tie-break) — deterministic for testability.

    Selection rules:
      * Only lots with ``status == OPEN`` and ``open_qty > 0`` are eligible.
      * Each lot is selected at most once (no double-counting).
      * The sum of ``close_qty`` is ``min(sell_qty, total_open_qty)`` — the
        property test asserts qty conservation.
      * If ``sell_qty`` exceeds total open qty, the surplus is left for the
        caller to surface as ``SellResult.unfilled_qty``.
    """
    if sell_qty <= 0:
        return []

    eligible = [lot for lot in open_lots if lot.is_open]
    # Highest cost-basis first; tie-break newer-acquired first; final tie-break lot_id asc.
    eligible.sort(
        key=lambda lot: (-lot.cost_basis_per_share, -lot.acquired_at.timestamp(), lot.lot_id)
    )

    selected: list[tuple[TaxLot, float]] = []
    remaining = float(sell_qty)
    for lot in eligible:
        if remaining <= 0:
            break
        take = min(lot.open_qty, remaining)
        if take > 0:
            selected.append((lot, take))
            remaining -= take
    return selected


def realize_lot(lot: TaxLot, close_qty: float, fill_price: float, as_of: datetime) -> RealizedLot:
    """Realize a single lot slice — computes proceeds, cost, gain, ST/LT class.

    Pure: returns a ``RealizedLot`` without mutating the input. The caller
    (``LedgerRepo``) persists the new lot state via the NoSQL conditional-write
    fence; this function is the math.

    Raises ``LedgerError`` if ``close_qty`` exceeds the lot's open qty (caller
    bug — the HIFO selector should never ask for more than is open).
    """
    if close_qty <= 0:
        raise LedgerError(f"close_qty must be positive; got {close_qty} for lot {lot.lot_id}")
    if close_qty > lot.open_qty + 1e-9:
        raise LedgerError(
            f"close_qty {close_qty} exceeds open qty {lot.open_qty} for lot {lot.lot_id}"
        )

    proceeds = close_qty * fill_price
    cost = close_qty * lot.cost_basis_per_share
    gain = proceeds - cost
    days = holding_days(lot, as_of)
    cls = HoldingClass.SHORT_TERM if days < ST_THRESHOLD_DAYS else HoldingClass.LONG_TERM
    return RealizedLot(
        lot_id=lot.lot_id,
        closed_qty=close_qty,
        fill_price=fill_price,
        cost_basis_per_share=lot.cost_basis_per_share,
        proceeds=proceeds,
        cost=cost,
        gain=gain,
        holding_days=days,
        holding_class=cls,
        closed_at=as_of,
        was_loss=gain < 0,
        triplet_slot=lot.triplet_slot,
    )


def close_lot(
    lot: TaxLot, close_qty: float, fill_price: float, as_of: datetime
) -> tuple[RealizedLot, LotClosed]:
    """Realize a slice and compute the new lot state to persist.

    Returns ``(realized_record, new_lot_state)``. The new lot state's
    ``new_status`` flips to ``CLOSED`` when the lot is fully closed, else
    stays ``OPEN`` with an incremented ``closed_qty``. The caller persists
    this via ``LedgerRepo._persist_close`` under the conditional-write fence.
    """
    realized = realize_lot(lot, close_qty, fill_price, as_of)
    new_closed_qty = lot.closed_qty + close_qty
    is_fully_closed = abs(new_closed_qty - lot.qty) < 1e-9 or new_closed_qty >= lot.qty
    new_status = LotStatus.CLOSED if is_fully_closed else LotStatus.OPEN
    return realized, LotClosed(
        lot_id=lot.lot_id,
        new_closed_qty=new_closed_qty,
        new_status=new_status,
        realized=realized,
    )


# --- Ledger repo — composes pure HIFO with the NoSQL conditional-write fence --- #


class LedgerRepo:
    """HIFO ledger backed by NoSQL conditional writes (plan §8, §16 acceptance).

    The sell path is atomic per-lot: each lot closure is a NoSQL
    ``update_if_condition`` that only succeeds if the lot's version matches
    AND its ``status == 'open'`` AND its ``closed_qty == expected``. Two
    concurrent sells against the same lot cannot both close it — the second
    sees the version bump from the first and raises ``ConditionalCheckFailed``.
    The caller retries the sell from the re-read.

    The pure HIFO selection (``hifo_select``) is computed against a snapshot of
    open lots; the persistence loop then closes each selected slice under the
    fence. If a concurrent writer closes a lot between the snapshot and the
    write, the conditional write fails and the caller re-runs the sell with a
    fresh snapshot. This is the "NoSQL conditional-write transition atomicity"
    acceptance criterion (plan §16).
    """

    TABLE = "tax_lots"
    REALIZED_TABLE = "realized_pnl"

    def __init__(self, store: NoSqlStore) -> None:
        self.store = store

    def open_lots(
        self,
        *,
        account: str,
        bucket_id: int,
        ticker: str,
    ) -> list[TaxLot]:
        """Read all open lots for a (account, bucket_id, ticker) position.

        Uses ``query`` (secondary lookup) on the ``tax_lots`` table. The OCI
        backend would use a NoSQL index on ``(account, bucket_id, ticker)``;
        the in-memory backend scans with equality predicates (sufficient for
        the small v1 universe).
        """
        rows = self.store.query(
            self.TABLE,
            where={"account": account, "bucket_id": bucket_id, "ticker": ticker},
        )
        lots: list[TaxLot] = []
        for row in rows:
            lot = _row_to_lot(row.payload)
            if lot.is_open:
                lots.append(lot)
        # Stable sort: highest cost-basis first, then newest acquired, then lot_id.
        lots.sort(
            key=lambda lot: (-lot.cost_basis_per_share, -lot.acquired_at.timestamp(), lot.lot_id)
        )
        return lots

    def get_lot(self, lot_id: LotId) -> TaxLot | None:
        """Point lookup by lot_id (the tax_lots primary key)."""
        row = self.store.get(self.TABLE, {"lot_id": lot_id})
        if row is None:
            return None
        return _row_to_lot(row.payload)

    def open_lot(self, lot: TaxLot) -> TaxLot:
        """Insert a new (open) lot. Raises ``ConditionalCheckFailed`` if the lot_id exists."""
        row = self.store.put_if_absent(self.TABLE, {"lot_id": lot.lot_id}, _lot_to_row(lot))
        return _row_to_lot(row.payload)

    def sell(
        self,
        request: SellRequest,
        *,
        max_retries: int = 8,
    ) -> SellResult:
        """Execute a HIFO sell against the (account, bucket_id, ticker) position.

        Atomicity: each lot closure is a conditional write. On
        ``ConditionalCheckFailed`` (a concurrent writer beat us to the lot),
        the sell re-reads the open-lots snapshot and re-runs the HIFO selection
        with the remaining qty. Up to ``max_retries`` rounds; raises
        ``LedgerError`` if the retries are exhausted (indicates a hot-contention
        bug or a stuck writer).

        Records each realized slice in ``realized_pnl`` as a separate row keyed
        by ``lot_id`` (the realized table's PK per plan §8). A lot that is
        partially closed multiple times will have multiple ``realized_pnl``
        rows with the same ``lot_id`` — the table is append-only by
        ``closed_at``; the PK is ``lot_id`` but NoSQL allows multiple rows per
        PK only with a designated array value (the v1 provisioning uses a
        composite PK of (lot_id, closed_at) for that reason — see
        infra/oci/nosql_tables.py).
        """
        if request.qty <= 0:
            return SellResult()
        if max_retries < 1:
            raise LedgerError("max_retries must be >= 1")

        result = SellResult()
        remaining = float(request.qty)
        retries = 0

        while remaining > 1e-9 and retries <= max_retries:
            round_closed, round_failed, exhausted = self._close_round(request, remaining, result)
            remaining -= round_closed
            if exhausted:
                # No more open lots to close — surface the remaining qty as unfilled.
                result.unfilled_qty += remaining
                break
            if round_closed < 1e-9 and round_failed > 0:
                # All closures in this round failed the fence — retry from a fresh snapshot.
                retries += 1
                continue
            if round_closed < 1e-9:
                # Nothing closed and nothing failed — open lots exist but HIFO selected none.
                result.unfilled_qty += remaining
                break

        if remaining > 1e-9 and retries > max_retries:
            raise LedgerError(
                f"HIFO sell retries exhausted: {retries} rounds, {remaining} qty unfilled "
                f"for {request.ticker}"
            )
        return result

    def _close_round(
        self,
        request: SellRequest,
        remaining: float,
        result: SellResult,
    ) -> tuple[float, int, bool]:
        """One HIFO selection + close-attempt pass.

        Returns ``(round_closed_qty, round_failed_count, exhausted)``. When
        ``exhausted`` is True, no open lots remain; the caller surfaces the
        remaining qty as unfilled and stops. On ``round_failed > 0`` with no
        closes, the caller retries from a fresh snapshot.
        """
        open_lots = self.open_lots(
            account=request.account,
            bucket_id=request.bucket_id,
            ticker=request.ticker,
        )
        if not open_lots:
            return 0.0, 0, True

        selected = hifo_select(open_lots, remaining)
        if not selected:
            return 0.0, 0, True

        round_closed = 0.0
        round_failed = 0
        for lot, close_qty in selected:
            try:
                realized, _ = self._close_under_fence(
                    lot, close_qty, request.fill_price, request.as_of
                )
            except ConditionalCheckFailed:
                round_failed += 1
                continue
            self._accumulate(result, realized, close_qty)
            round_closed += close_qty
        return round_closed, round_failed, False

    @staticmethod
    def _accumulate(result: SellResult, realized: RealizedLot, close_qty: float) -> None:
        """Fold one realized slice into the running SellResult totals."""
        result.realized.append(realized)
        result.total_proceeds += realized.proceeds
        result.total_cost += realized.cost
        result.total_gain += realized.gain
        if realized.holding_class == HoldingClass.SHORT_TERM:
            result.st_gain += realized.gain
        else:
            result.lt_gain += realized.gain
        if realized.was_loss:
            result.harvested_loss += abs(realized.gain)
        # `close_qty` is consumed by the caller's `round_closed` accumulator.

    def _close_under_fence(
        self,
        lot: TaxLot,
        close_qty: float,
        fill_price: float,
        as_of: datetime,
    ) -> tuple[RealizedLot, LotClosed]:
        """Close one lot slice under the NoSQL conditional-write fence.

        The fence: ``status == 'open'`` AND ``closed_qty == lot.closed_qty``.
        A concurrent sell that closes the same lot between our read and our
        write will have bumped ``closed_qty`` and the row's version; our write
        fails with ``ConditionalCheckFailed`` and the caller retries.
        """
        realized, new_state = close_lot(lot, close_qty, fill_price, as_of)
        # Re-read the current row to get the version (the open_lots snapshot may be stale).
        current = self.store.get(self.TABLE, {"lot_id": lot.lot_id})
        if current is None:
            raise ConditionalCheckFailed(
                self.TABLE, {"lot_id": lot.lot_id}, "lot vanished between snapshot and close"
            )
        expected_closed_qty = lot.closed_qty
        # Build the new payload: merge the existing row's fields with the closure updates.
        new_payload: dict[str, Any] = dict(current.payload)
        new_payload["closed_qty"] = new_state.new_closed_qty
        new_payload["status"] = new_state.new_status.value
        if new_state.new_status == LotStatus.CLOSED and lot.wash_immunity_until is not None:
            new_payload["wash_immunity_until"] = lot.wash_immunity_until.isoformat()
        # The fence: status must still be 'open' and closed_qty must match what we read.
        self.store.update_if_condition(
            self.TABLE,
            {"lot_id": lot.lot_id},
            new_payload,
            expected_version=current.version,
            condition=lambda payload: (
                payload.get("status") == LotStatus.OPEN.value
                and abs(float(payload.get("closed_qty", 0.0)) - expected_closed_qty) < 1e-9
            ),
            condition_desc=f"status==open and closed_qty=={expected_closed_qty}",
        )
        # Record the realized slice in realized_pnl (append by (lot_id, closed_at)).
        realized_payload: dict[str, Any] = {
            "lot_id": realized.lot_id,
            "closed_at": realized.closed_at.isoformat(),
            "closed_qty": realized.closed_qty,
            "fill_price": realized.fill_price,
            "cost_basis_per_share": realized.cost_basis_per_share,
            "proceeds": realized.proceeds,
            "cost": realized.cost,
            "gain": realized.gain,
            "holding_days": realized.holding_days,
            "st_lt": realized.holding_class.value,
            "was_loss": realized.was_loss,
            "triplet_slot": realized.triplet_slot,
            "account": lot.account,
            "bucket_id": lot.bucket_id,
            "ticker": lot.ticker,
        }
        self.store.put_if_absent(
            self.REALIZED_TABLE,
            {"lot_id": realized.lot_id, "closed_at": realized.closed_at.isoformat()},
            realized_payload,
        )
        return realized, new_state


# --- Row <-> TaxLot marshaling --- #


def _lot_to_row(lot: TaxLot) -> dict[str, Any]:
    return {
        "lot_id": lot.lot_id,
        "account": lot.account,
        "bucket_id": lot.bucket_id,
        "ticker": lot.ticker,
        "triplet_slot": lot.triplet_slot,
        "qty": lot.qty,
        "closed_qty": lot.closed_qty,
        "cost_basis_per_share": lot.cost_basis_per_share,
        "acquired_at": lot.acquired_at.isoformat(),
        "status": lot.status.value,
        "wash_immunity_until": lot.wash_immunity_until.isoformat()
        if lot.wash_immunity_until
        else None,
    }


def _row_to_lot(payload: dict[str, Any]) -> TaxLot:
    return TaxLot(
        lot_id=str(payload["lot_id"]),
        account=str(payload["account"]),
        bucket_id=int(payload["bucket_id"]),
        ticker=str(payload["ticker"]),
        triplet_slot=str(payload["triplet_slot"]),
        qty=float(payload["qty"]),
        closed_qty=float(payload.get("closed_qty", 0.0)),
        cost_basis_per_share=float(payload["cost_basis_per_share"]),
        acquired_at=datetime.fromisoformat(str(payload["acquired_at"])),
        status=LotStatus(str(payload.get("status", LotStatus.OPEN.value))),
        wash_immunity_until=(
            datetime.fromisoformat(str(payload["wash_immunity_until"]))
            if payload.get("wash_immunity_until")
            else None
        ),
    )
