"""Tests for state/triplet.py — A->B->C state machine (plan §8, §16; brief §1).

Hypothesis property tests (the plan §16 acceptance criteria):
  * **never re-enters a slot inside the 60-day immunization window** —
    ``can_enter(state, lost_slot, t)`` is False for every ``t`` in
    ``[last_loss, last_loss + 60d)``.
  * **30-day wash-sale restricted slot** — ``can_enter(state, lost_slot, t)``
    is False for ``t`` in ``[last_loss, last_loss + 30d)`` (defense-in-depth
    on top of the 60d immunization; the 30d window is a subset).
  * **transitions are append-only** — ``slot_history`` grows monotonically;
    no entry is removed or mutated.
  * **NoSQL conditional-write transition atomicity** — two concurrent
    ``advance_on_loss`` calls on the same bucket cannot both succeed; one
    wins, the other retries from a fresh snapshot.

Plus targeted tests for:
  * Cyclic advance A -> B -> C -> A.
  * Immunization deadline math (60d from the loss timestamp).
  * The other-two slots remain enterable during the immunization window.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from personal_strat_pai.state.nosql import ConditionalCheckFailed, InMemoryNoSqlStore
from personal_strat_pai.state.triplet import (
    IMMUNIZATION_DAYS,
    WASH_SALE_DAYS,
    TripletRepo,
    advance_on_loss,
    advance_slot,
    can_enter,
    lost_slot_window,
    seed_state,
    slot_immunized_until,
)

# --- Pure transition tests --- #


def test_advance_slot_cyclic():
    """A -> B -> C -> A (brief §1)."""
    assert advance_slot("A") == "B"
    assert advance_slot("B") == "C"
    assert advance_slot("C") == "A"


def test_advance_slot_rejects_invalid():
    with pytest.raises(ValueError):
        advance_slot("D")
    with pytest.raises(ValueError):
        advance_slot("")


def test_seed_state_initial():
    state = seed_state(bucket_id=1, initial_slot="A")
    assert state.bucket_id == 1
    assert state.current_slot == "A"
    assert state.last_loss_at is None
    assert state.immunized_until is None
    assert len(state.slot_history) == 1
    assert state.slot_history[0].reason == "seed"
    assert state.lost_slot is None


def test_advance_on_loss_advances_and_immunizes():
    state = seed_state(bucket_id=1, initial_slot="A")
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    new_state = advance_on_loss(state, as_of)
    assert new_state.current_slot == "B"  # A -> B
    assert new_state.last_loss_at == as_of
    assert new_state.immunized_until == as_of + timedelta(days=IMMUNIZATION_DAYS)
    assert new_state.lost_slot == "A"
    # History is append-only: original seed + 1 loss entry.
    assert len(new_state.slot_history) == 2
    assert new_state.slot_history[-1].reason == "loss"
    assert new_state.slot_history[-1].from_slot == "A"
    assert new_state.slot_history[-1].to_slot == "B"
    assert new_state.slot_history[-1].lost_slot == "A"
    # The original state is NOT mutated (append-only at the persistence layer;
    # the pure function returns a new state).
    assert state.current_slot == "A"
    assert len(state.slot_history) == 1


def test_advance_on_loss_chains_abc_then_a():
    """Three loss advances cycle A -> B -> C -> A (brief §1)."""
    state = seed_state(bucket_id=1, initial_slot="A")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(days=70)  # past the 60d window
    t2 = t1 + timedelta(days=70)
    s1 = advance_on_loss(state, t0)
    s2 = advance_on_loss(s1, t1)
    s3 = advance_on_loss(s2, t2)
    assert s1.current_slot == "B"
    assert s2.current_slot == "C"
    assert s3.current_slot == "A"
    assert len(s3.slot_history) == 4  # seed + 3 losses


# --- Immunization + wash-sale fence tests --- #


def test_can_enter_blocks_lost_slot_for_60_days():
    """Brief §1: 60+ day immunization window before a specific asset can be re-entered."""
    state = seed_state(bucket_id=1, initial_slot="A")
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    state = advance_on_loss(state, t0)  # A -> B, lost slot = A, immunized 60d
    # Inside the 60d window: cannot re-enter A.
    for days in (0, 1, 10, 30, 45, 59):
        t = t0 + timedelta(days=days)
        assert not can_enter(state, "A", t), f"slot A should be immunized at +{days}d"
    # At exactly 60d: allowed.
    assert can_enter(state, "A", t0 + timedelta(days=IMMUNIZATION_DAYS))


def test_can_enter_allows_other_two_slots_during_window():
    """The immunization is on the *lost slot*; the other two remain enterable."""
    state = seed_state(bucket_id=1, initial_slot="A")
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    state = advance_on_loss(state, t0)  # A -> B, immunized A
    # B (current) and C are enterable.
    assert can_enter(state, "B", t0 + timedelta(days=1))
    assert can_enter(state, "C", t0 + timedelta(days=1))


def test_lost_slot_window_returns_30d_window():
    state = seed_state(bucket_id=1, initial_slot="A")
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    state = advance_on_loss(state, t0)
    window = lost_slot_window(state, t0 + timedelta(days=1))
    assert window is not None
    lost_slot, restricted_from, restricted_until = window
    assert lost_slot == "A"
    assert restricted_from == t0
    assert restricted_until == t0 + timedelta(days=WASH_SALE_DAYS)


def test_lost_slot_window_returns_none_after_30d():
    state = seed_state(bucket_id=1, initial_slot="A")
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    state = advance_on_loss(state, t0)
    # After 30d, the wash-sale restriction has elapsed (the 60d immunization
    # still applies — can_enter checks both fences).
    assert lost_slot_window(state, t0 + timedelta(days=WASH_SALE_DAYS + 1)) is None


def test_slot_immunized_until_returns_deadline_for_lost_slot():
    state = seed_state(bucket_id=1, initial_slot="A")
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    state = advance_on_loss(state, t0)
    assert slot_immunized_until(state, "A") == t0 + timedelta(days=IMMUNIZATION_DAYS)
    # Other slots are not immunized.
    assert slot_immunized_until(state, "B") is None
    assert slot_immunized_until(state, "C") is None


def test_can_enter_rejects_invalid_slot():
    state = seed_state(bucket_id=1, initial_slot="A")
    with pytest.raises(ValueError):
        can_enter(state, "D", datetime(2026, 7, 1, tzinfo=UTC))


# --- Hypothesis property tests (plan §16 acceptance) --- #


@given(
    initial_slot=st.sampled_from(["A", "B", "C"]),
    loss_count=st.integers(min_value=1, max_value=6),
    spacing_days=st.integers(min_value=0, max_value=120),
)
@settings(max_examples=100, deadline=None)
def test_property_never_reenters_inside_60d_window(
    initial_slot: str, loss_count: int, spacing_days: int
):
    """After each loss advance, the lost slot is un-enterable for 60 days."""
    state = seed_state(bucket_id=1, initial_slot=initial_slot)
    t = datetime(2026, 1, 1, tzinfo=UTC)
    for _ in range(loss_count):
        t = t + timedelta(days=spacing_days)
        state = advance_on_loss(state, t)
        lost = state.lost_slot
        assert lost is not None
        # Inside the 60d window: cannot re-enter.
        for d in (0, 1, 30, 59):
            probe = t + timedelta(days=d)
            if probe < state.immunized_until:
                assert not can_enter(state, lost, probe), (
                    f"lost slot {lost} should be immunized at {probe}; "
                    f"immunized_until={state.immunized_until}"
                )
        # At exactly 60d: allowed.
        assert can_enter(state, lost, state.immunized_until)


@given(
    initial_slot=st.sampled_from(["A", "B", "C"]),
    loss_count=st.integers(min_value=1, max_value=4),
    spacing_days=st.integers(min_value=70, max_value=200),  # always past 60d window
)
@settings(max_examples=100, deadline=None)
def test_property_transitions_append_only(initial_slot: str, loss_count: int, spacing_days: int):
    """slot_history grows monotonically; no entry is removed or mutated."""
    state = seed_state(bucket_id=1, initial_slot=initial_slot)
    history_snapshots: list[list] = [list(state.slot_history)]
    t = datetime(2026, 1, 1, tzinfo=UTC)
    for _ in range(loss_count):
        t = t + timedelta(days=spacing_days)
        state = advance_on_loss(state, t)
        # The previous snapshot is a prefix of the new history (append-only).
        prev = history_snapshots[-1]
        assert state.slot_history[: len(prev)] == prev, "history was mutated, not appended"
        history_snapshots.append(list(state.slot_history))
    # Final length = seed + loss_count.
    assert len(state.slot_history) == 1 + loss_count


@given(
    initial_slot=st.sampled_from(["A", "B", "C"]),
    loss_count=st.integers(min_value=0, max_value=10),
    spacing_days=st.integers(min_value=70, max_value=200),
)
@settings(max_examples=100, deadline=None)
def test_property_advance_always_cycles(initial_slot: str, loss_count: int, spacing_days: int):
    """advance_on_loss always moves A -> B -> C -> A cyclically."""
    state = seed_state(bucket_id=1, initial_slot=initial_slot)
    t = datetime(2026, 1, 1, tzinfo=UTC)
    expected = initial_slot
    for _ in range(loss_count):
        t = t + timedelta(days=spacing_days)
        state = advance_on_loss(state, t)
        expected = advance_slot(expected)
        assert state.current_slot == expected


@given(
    initial_slot=st.sampled_from(["A", "B", "C"]),
    probe_days=st.integers(min_value=0, max_value=200),
)
@settings(max_examples=100, deadline=None)
def test_property_other_two_slots_always_enterable_after_loss(initial_slot: str, probe_days: int):
    """A loss immunizes only the lost slot; the other two are always enterable."""
    state = seed_state(bucket_id=1, initial_slot=initial_slot)
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    state = advance_on_loss(state, t0)
    lost = state.lost_slot
    others = [s for s in ("A", "B", "C") if s != lost]
    probe = t0 + timedelta(days=probe_days)
    for s in others:
        assert can_enter(state, s, probe), f"slot {s} should be enterable at {probe}"


# --- TripletRepo tests (NoSQL-backed; atomicity) --- #


@pytest.fixture
def triplet_repo():
    s = InMemoryNoSqlStore()
    s.create_table("triplet_state", key_schema={"bucket_id": "int"})
    return TripletRepo(s)


def test_repo_init_bucket_seeds_state(triplet_repo):
    state = triplet_repo.init_bucket(bucket_id=1, initial_slot="A")
    assert state.current_slot == "A"
    # Re-initializing with the same slot is idempotent.
    state2 = triplet_repo.init_bucket(bucket_id=1, initial_slot="A")
    assert state2.current_slot == "A"


def test_repo_init_bucket_rejects_conflicting_initial_slot(triplet_repo):
    triplet_repo.init_bucket(bucket_id=1, initial_slot="A")
    with pytest.raises(ConditionalCheckFailed):
        triplet_repo.init_bucket(bucket_id=1, initial_slot="B")


def test_repo_advance_on_loss_persists_new_slot(triplet_repo):
    triplet_repo.init_bucket(bucket_id=1, initial_slot="A")
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    new_state = triplet_repo.advance_on_loss(bucket_id=1, as_of=as_of)
    assert new_state.current_slot == "B"
    # Re-read from the store: the persisted state matches.
    persisted = triplet_repo.get_state(bucket_id=1)
    assert persisted.current_slot == "B"
    assert persisted.last_loss_at == as_of
    assert persisted.immunized_until == as_of + timedelta(days=IMMUNIZATION_DAYS)
    assert persisted.lost_slot == "A"


def test_repo_can_enter_blocks_lost_slot(triplet_repo):
    triplet_repo.init_bucket(bucket_id=1, initial_slot="A")
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    triplet_repo.advance_on_loss(bucket_id=1, as_of=t0)
    # Inside 60d: cannot re-enter A.
    assert not triplet_repo.can_enter(bucket_id=1, slot="A", as_of=t0 + timedelta(days=30))
    # B (current) is enterable.
    assert triplet_repo.can_enter(bucket_id=1, slot="B", as_of=t0 + timedelta(days=30))


def test_repo_can_enter_true_when_no_state(triplet_repo):
    """No state = no losses = nothing immunized."""
    assert triplet_repo.can_enter(bucket_id=99, slot="A", as_of=datetime(2026, 7, 1, tzinfo=UTC))


def test_repo_concurrent_advances_one_wins(triplet_repo):
    """The atomicity fence: two concurrent advance_on_loss on the same bucket
    cannot both advance the slot. One wins (slot A -> B); the other retries
    from the fresh snapshot and advances B -> C. End state: C, not B-with-
    two-advances-applied-as-one.
    """
    triplet_repo.init_bucket(bucket_id=1, initial_slot="A")
    t0 = datetime(2026, 7, 1, tzinfo=UTC)

    barrier = threading.Barrier(2)
    results: list = [None, None]

    def advancer(idx: int) -> None:
        barrier.wait(timeout=2.0)
        try:
            results[idx] = ("ok", triplet_repo.advance_on_loss(bucket_id=1, as_of=t0))
        except Exception as e:
            results[idx] = ("err", e)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(advancer, 0)
        f2 = ex.submit(advancer, 1)
        f1.result()
        f2.result()

    # Both should succeed (the retry loop handles the fence), but the final
    # state must reflect TWO advances (A -> B -> C), not one.
    final = triplet_repo.get_state(bucket_id=1)
    assert final.current_slot == "C"
    # Two loss entries in the history (plus the seed).
    loss_entries = [e for e in final.slot_history if e.reason == "loss"]
    assert len(loss_entries) == 2
    # The two advances must be sequential (B happened after A, C after B).
    assert loss_entries[0].from_slot == "A"
    assert loss_entries[0].to_slot == "B"
    assert loss_entries[1].from_slot == "B"
    assert loss_entries[1].to_slot == "C"


def test_repo_advance_no_state_raises_ledger_error(triplet_repo):
    from personal_strat_pai.state.ledger import LedgerError

    with pytest.raises(LedgerError, match="no triplet state"):
        triplet_repo.advance_on_loss(bucket_id=99, as_of=datetime(2026, 7, 1, tzinfo=UTC))
