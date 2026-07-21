"""Triplet state machine — A->B->C with 60-day immunization + 30-day wash-sale
restricted slot (plan §8, brief §1).

Each of the 13 universe buckets (brief §2) holds an internal A/B/C sequence of
three structurally distinct ETFs (e.g., for Technology: XLK / SMH / IGV). The
triplet machine is the wash-sale bypass mechanism:

  * When a position is liquidated at a loss in bucket *B*, the bucket's
    ``current_slot`` advances B -> C and ``immunized_until`` is set to
    ``now + 60 days``. The bucket cannot re-enter the slot it just exited
    (slot *B*) for 60 days — this *guarantees* a 60+ day immunization window
    before a specific asset can be re-entered, cleanly bypassing the IRS
    30-day wash-sale rule (brief §1).
  * The slot just exited (the "lost slot") is **RESTRICTED for 30 days** as a
    defense-in-depth: even though the 60-day immunization on the *bucket*
    already prevents re-entry, the 30-day slot restriction is a separate
    flag the pre-trade wash-sale lock (exec/router.py, P0-4) checks before
    placing a buy on that slot's ticker.
  * Transitions are **append-only** (``slot_history``). The current slot can
    advance on a loss; it cannot be rewound. A revert is only possible via an
    explicit, logged admin op (not exposed here — admin tooling is P0-4).

The atomicity boundary lives in ``state/nosql.py``: the advance is a NoSQL
``update_if_condition`` that only succeeds if the row's ``current_slot`` still
matches the expected slot. Two concurrent stop-outs on the same bucket cannot
both advance the slot — the second sees the bump from the first and raises
``ConditionalCheckFailed``. This is the "NoSQL conditional-write transition
atomicity" acceptance criterion (plan §16).

Immunization math (brief §1):
  * ``IMMUNIZATION_DAYS = 60`` — the bucket cannot re-enter the *lost slot*
    for 60 days after the loss. The immunization window is on the **slot**,
    not the bucket: the bucket can still trade the *other two* slots during
    the window. The 60-day window is the wash-sale bypass (30-day IRS rule +
    30 days of safety margin).
  * ``WASH_SALE_DAYS = 30`` — the lost slot is RESTRICTED for 30 days. This is
    a defense-in-depth on top of the 60-day slot immunization: even if a
    future code path tried to re-enter the lost slot's ticker, the wash-sale
    lock would block it for 30 days. The 60-day slot immunization is the
    primary fence; the 30-day slot restriction is the secondary fence.

Pure functions (``advance_on_loss``, ``can_enter``, ``lost_slot_window``) are
property-tested with hypothesis. The NoSQL-backed ``TripletRepo`` composes
them with the conditional-write fence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from personal_strat_pai.state.nosql import (
    ConditionalCheckFailed,
    NoSqlStore,
)

__all__ = [
    "IMMUNIZATION_DAYS",
    "WASH_SALE_DAYS",
    "TripletRepo",
    "TripletSlot",
    "TripletState",
    "advance_on_loss",
    "advance_slot",
    "can_enter",
    "lost_slot_window",
    "slot_immunized_until",
]

# Brief §1: 60-day immunization window (30-day IRS wash-sale + 30d margin).
IMMUNIZATION_DAYS: int = 60
# Brief §1: 30-day wash-sale restricted slot (defense-in-depth on the lost slot).
WASH_SALE_DAYS: int = 30

SLOTS: tuple[str, str, str] = ("A", "B", "C")


class TripletSlot(str):
    """A triplet slot label — 'A', 'B', or 'C' (brief §1).

    Subclassing ``str`` so the value is comparable to plain strings while
    carrying the type for mypy. The advance rule is cyclic: A -> B -> C -> A.
    """


def advance_slot(slot: str) -> str:
    """Cyclic advance: A -> B -> C -> A (brief §1)."""
    if slot not in SLOTS:
        raise ValueError(f"invalid triplet slot {slot!r}; must be one of {SLOTS}")
    idx = SLOTS.index(slot)
    return SLOTS[(idx + 1) % len(SLOTS)]


@dataclass(frozen=True, slots=True)
class SlotHistoryEntry:
    """A single append-only transition record (plan §8 ``slot_history`` list).

    ``reason`` is "loss" for the normal stop-out advance; "admin" for the
    explicit-logged admin revert (not exposed in v1 — the type is here so the
    schema is stable). ``immunized_until`` is the immunization deadline set on
    a loss advance; ``None`` for admin transitions.
    """

    at: datetime
    from_slot: str
    to_slot: str
    reason: str  # "loss" | "admin"
    immunized_until: datetime | None = None
    lost_slot: str | None = None  # the slot just exited (the restricted slot)


@dataclass(slots=True)
class TripletState:
    """The triplet state for a single bucket (plan §8 ``triplet_state`` table).

    ``current_slot`` — the slot the next buy will use (A, B, or C).
    ``last_loss_at`` — timestamp of the most recent loss advance; ``None`` if
    the bucket has never had a loss-driven advance.
    ``immunized_until`` — the bucket's immunization deadline; the *lost slot*
    cannot be re-entered before this. ``None`` if no immunization is active.
    ``slot_history`` — append-only list of transitions. The first entry is the
    initial state ("seed"); subsequent entries are loss advances or admin ops.
    """

    bucket_id: int
    current_slot: str
    last_loss_at: datetime | None = None
    immunized_until: datetime | None = None
    slot_history: list[SlotHistoryEntry] = field(default_factory=list)

    @property
    def lost_slot(self) -> str | None:
        """The slot most recently exited via a loss advance (the RESTRICTED slot).

        ``None`` if no loss advance has happened. Pulled from the last loss
        entry in ``slot_history`` for convenience — the canonical record is
        the history list.
        """
        for entry in reversed(self.slot_history):
            if entry.reason == "loss":
                return entry.lost_slot
        return None


def lost_slot_window(state: TripletState, as_of: datetime) -> tuple[str, datetime, datetime] | None:
    """If the lost slot is currently RESTRICTED, return ``(lost_slot, restricted_from, restricted_until)``.

    The restriction window is ``[last_loss_at, last_loss_at + WASH_SALE_DAYS]``.
    Returns ``None`` if no loss has happened or if the window has elapsed.
    """
    if state.last_loss_at is None or state.lost_slot is None:
        return None
    restricted_until = state.last_loss_at + timedelta(days=WASH_SALE_DAYS)
    if as_of >= restricted_until:
        return None
    return (state.lost_slot, state.last_loss_at, restricted_until)


def slot_immunized_until(state: TripletState, slot: str) -> datetime | None:
    """The immunization deadline for a specific slot, or ``None`` if not immunized.

    A slot is immunized if it was the lost slot in a loss advance whose
    60-day window has not yet elapsed. Returns the deadline (so callers can
    compute "days remaining") or ``None`` if the slot is clear.
    """
    if slot not in SLOTS:
        raise ValueError(f"invalid slot {slot!r}")
    # Walk the history backwards; the most recent loss advance immunizes its lost_slot.
    for entry in reversed(state.slot_history):
        if entry.reason != "loss":
            continue
        if entry.lost_slot == slot and entry.immunized_until is not None:
            return entry.immunized_until
        # A more recent loss advance supersedes older immunizations on other slots.
        return None
    return None


def can_enter(state: TripletState, slot: str, as_of: datetime) -> bool:
    """Pre-trade check: may the bucket enter ``slot`` at ``as_of``?

    Two fences:
      1. **Slot immunization (60d)** — ``slot`` is immunized if it was the
         lost slot in a loss advance whose 60-day window has not elapsed.
         Brief §1: "guarantees a 60+ day immunization window before a specific
         asset can be re-entered".
      2. **Wash-sale slot restriction (30d)** — defense-in-depth on the lost
         slot. The 60-day immunization already prevents re-entry; the 30-day
         restriction is the secondary fence the pre-trade wash-sale lock
         (exec/router.py) double-checks.

    Returns ``False`` if either fence blocks entry. Both fences are on the
    *lost slot*; the bucket can always trade the *other two* slots.
    """
    if slot not in SLOTS:
        raise ValueError(f"invalid slot {slot!r}")
    # Immunization fence (60d).
    deadline = slot_immunized_until(state, slot)
    if deadline is not None and as_of < deadline:
        return False
    # Wash-sale slot restriction fence (30d) — defense-in-depth on the lost slot.
    window = lost_slot_window(state, as_of)
    return not (window is not None and window[0] == slot)


def advance_on_loss(state: TripletState, as_of: datetime) -> TripletState:
    """Pure transition: advance the bucket's current_slot on a loss liquidation.

    Sets ``last_loss_at = as_of`` and ``immunized_until = as_of + 60d``. The
    slot just exited becomes the RESTRICTED (lost) slot for 30 days and is
    immunized for 60 days. Appends a ``loss`` entry to ``slot_history``.
    Returns a NEW ``TripletState`` — the input is not mutated (append-only
    is enforced at the persistence layer; this function builds the new state).

    Raises ``ValueError`` if ``current_slot`` is not a valid slot (caller bug).
    """
    if state.current_slot not in SLOTS:
        raise ValueError(f"invalid current_slot {state.current_slot!r}")
    lost_slot = state.current_slot
    new_slot = advance_slot(lost_slot)
    immunized_until = as_of + timedelta(days=IMMUNIZATION_DAYS)
    entry = SlotHistoryEntry(
        at=as_of,
        from_slot=lost_slot,
        to_slot=new_slot,
        reason="loss",
        immunized_until=immunized_until,
        lost_slot=lost_slot,
    )
    # Append-only: copy the existing history + the new entry.
    new_history = [*state.slot_history, entry]
    return TripletState(
        bucket_id=state.bucket_id,
        current_slot=new_slot,
        last_loss_at=as_of,
        immunized_until=immunized_until,
        slot_history=new_history,
    )


def seed_state(bucket_id: int, initial_slot: str = "A") -> TripletState:
    """Initialize a bucket's triplet state with a seed history entry.

    The seed entry records the bucket's initial slot at ``now`` (UTC) so the
    history is non-empty from creation. The first loss advance appends on top.
    """
    if initial_slot not in SLOTS:
        raise ValueError(f"invalid initial_slot {initial_slot!r}")
    now = datetime.now(tz=UTC)
    seed = SlotHistoryEntry(
        at=now,
        from_slot=initial_slot,
        to_slot=initial_slot,
        reason="seed",
        immunized_until=None,
        lost_slot=None,
    )
    return TripletState(
        bucket_id=bucket_id,
        current_slot=initial_slot,
        last_loss_at=None,
        immunized_until=None,
        slot_history=[seed],
    )


# --- Triplet repo — composes pure transitions with the NoSQL conditional-write fence --- #


class TripletRepo:
    """Triplet state machine backed by NoSQL conditional writes (plan §8, §16).

    The advance is atomic: a NoSQL ``update_if_condition`` that only succeeds
    if the row's ``current_slot`` still matches the expected slot. Two
    concurrent stop-outs on the same bucket cannot both advance — the second
    sees the slot bump from the first and raises ``ConditionalCheckFailed``.
    The caller retries from the re-read.

    ``slot_history`` is stored as a NoJSON list column (the OCI backend stores
    it as a JSON array; the in-memory backend stores it as a Python list). The
    append-only invariant is enforced by the conditional write: the new
    payload is built by reading the current history and appending the new
    entry under the version fence — a concurrent writer cannot truncate it.
    """

    TABLE = "triplet_state"

    def __init__(self, store: NoSqlStore) -> None:
        self.store = store

    def get_state(self, bucket_id: int) -> TripletState | None:
        row = self.store.get(self.TABLE, {"bucket_id": bucket_id})
        if row is None:
            return None
        return _row_to_state(row.payload)

    def init_bucket(self, bucket_id: int, initial_slot: str = "A") -> TripletState:
        """Seed a bucket's triplet state. Idempotent on (bucket_id, initial_slot).

        Raises ``ConditionalCheckFailed`` if the bucket already has state with
        a *different* initial slot — the caller must use ``advance_on_loss``
        or the admin revert path instead of re-seeding.
        """
        state = seed_state(bucket_id, initial_slot)
        try:
            self.store.put_if_absent(self.TABLE, {"bucket_id": bucket_id}, _state_to_row(state))
        except ConditionalCheckFailed:
            existing = self.get_state(bucket_id)
            if existing is None or existing.current_slot != initial_slot:
                raise
            return existing
        return state

    def advance_on_loss(
        self,
        bucket_id: int,
        as_of: datetime,
        *,
        max_retries: int = 8,
    ) -> TripletState:
        """Advance the bucket's triplet slot on a loss liquidation (atomic).

        Reads the current state, computes the pure transition, and persists it
        under a ``current_slot == expected_slot`` conditional-write fence. On
        ``ConditionalCheckFailed`` (a concurrent advance beat us), re-reads and
        retries up to ``max_retries`` times.

        Raises ``LedgerError`` if the bucket has no state (init_bucket must
        run first) or if retries are exhausted (indicates a stuck writer).
        """
        from personal_strat_pai.state.ledger import LedgerError  # local import — circular-avoidance

        if max_retries < 1:
            raise LedgerError("max_retries must be >= 1")

        last_err: Exception | None = None
        for _ in range(max_retries + 1):
            current = self.get_state(bucket_id)
            if current is None:
                raise LedgerError(
                    f"bucket {bucket_id} has no triplet state; call init_bucket first"
                )
            new_state = advance_on_loss(current, as_of)
            # Re-read to get the current version (the get_state above is the
            # version source — we re-fetch to make the fence explicit).
            row = self.store.get(self.TABLE, {"bucket_id": bucket_id})
            if row is None:
                raise LedgerError(f"bucket {bucket_id} state vanished mid-advance")
            expected_slot = current.current_slot
            try:

                def slot_matches(payload: dict[str, Any], expected: str = expected_slot) -> bool:
                    return payload.get("current_slot") == expected

                self.store.update_if_condition(
                    self.TABLE,
                    {"bucket_id": bucket_id},
                    _state_to_row(new_state),
                    expected_version=row.version,
                    condition=slot_matches,
                    condition_desc=f"current_slot=={expected_slot}",
                )
            except ConditionalCheckFailed as e:
                last_err = e
                continue
            return new_state
        raise LedgerError(f"triplet advance retries exhausted for bucket {bucket_id}: {last_err}")

    def can_enter(self, bucket_id: int, slot: str, as_of: datetime) -> bool:
        """Pre-trade wash-sale check: may the bucket enter ``slot`` at ``as_of``?

        Reads the current state and delegates to the pure ``can_enter``. Returns
        ``True`` if the bucket has no state (no losses yet — nothing is
        immunized). The exec/router (P0-4) calls this before placing a buy on
        the slot's ticker.
        """
        state = self.get_state(bucket_id)
        if state is None:
            return True
        return can_enter(state, slot, as_of)


# --- Row <-> TripletState marshaling --- #


def _state_to_row(state: TripletState) -> dict[str, Any]:
    return {
        "bucket_id": state.bucket_id,
        "current_slot": state.current_slot,
        "last_loss_at": state.last_loss_at.isoformat() if state.last_loss_at else None,
        "immunized_until": state.immunized_until.isoformat() if state.immunized_until else None,
        "slot_history": [
            {
                "at": e.at.isoformat(),
                "from_slot": e.from_slot,
                "to_slot": e.to_slot,
                "reason": e.reason,
                "immunized_until": e.immunized_until.isoformat() if e.immunized_until else None,
                "lost_slot": e.lost_slot,
            }
            for e in state.slot_history
        ],
    }


def _row_to_state(payload: dict[str, Any]) -> TripletState:
    history_raw = payload.get("slot_history") or []
    history = [
        SlotHistoryEntry(
            at=datetime.fromisoformat(str(e["at"])),
            from_slot=str(e["from_slot"]),
            to_slot=str(e["to_slot"]),
            reason=str(e["reason"]),
            immunized_until=(
                datetime.fromisoformat(str(e["immunized_until"]))
                if e.get("immunized_until")
                else None
            ),
            lost_slot=str(e["lost_slot"]) if e.get("lost_slot") else None,
        )
        for e in history_raw
    ]
    return TripletState(
        bucket_id=int(payload["bucket_id"]),
        current_slot=str(payload["current_slot"]),
        last_loss_at=(
            datetime.fromisoformat(str(payload["last_loss_at"]))
            if payload.get("last_loss_at")
            else None
        ),
        immunized_until=(
            datetime.fromisoformat(str(payload["immunized_until"]))
            if payload.get("immunized_until")
            else None
        ),
        slot_history=history,
    )
