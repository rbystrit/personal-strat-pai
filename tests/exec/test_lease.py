"""Tests for exec/lease.py — the v1 NO-OP stub (plan §3.3, §8; D12).

The v1 lease stub always succeeds (single writer, no fencing). The contract:
  * held_by_me() always returns True.
  * acquire() / renew() / release() are no-ops that do NOT touch NoSQL.
  * The execution_lease table is NOT created in v1.

These tests verify the v1 stub contract. The v2 fencing property tests (two
concurrent acquirers never both hold; backup never trades while primary TTL
is live) are P0-5 deliverables — not in this suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from personal_strat_pai.exec import ExecutionLease, LeaseError, NoOpLease
from personal_strat_pai.state.nosql import InMemoryNoSqlStore


def test_noop_lease_satisfies_execution_lease_protocol():
    """NoOpLease is a structural ExecutionLease (the Protocol is the seam for v2)."""
    lease = NoOpLease()
    assert isinstance(lease, ExecutionLease)


def test_noop_lease_held_by_me_always_true():
    lease = NoOpLease()
    assert lease.held_by_me() is True
    assert lease.held_by_me(now=datetime(2026, 7, 1, tzinfo=UTC)) is True


def test_noop_lease_acquire_returns_placeholder():
    lease = NoOpLease()
    info = lease.acquire()
    assert info.holder == "primary"
    assert info.expires_at > datetime.now(tz=UTC)
    assert info.generation == 0


def test_noop_lease_renew_returns_placeholder():
    lease = NoOpLease()
    info = lease.renew()
    assert info.holder == "primary"


def test_noop_lease_release_is_noop():
    lease = NoOpLease()
    # Returns None; does not raise.
    assert lease.release() is None


def test_noop_lease_does_not_create_execution_lease_table():
    """The v1 stub MUST NOT create the execution_lease table (D12).

    The table is created in P0-5 (v2 backup) by the real OciExecutionLease;
    the stub never touches NoSQL.
    """
    store = InMemoryNoSqlStore()
    NoOpLease(store=store)
    # Confirm the stub did not create the execution_lease table.
    from personal_strat_pai.state.nosql import TableNotProvisioned

    with pytest.raises(TableNotProvisioned):
        store.get("execution_lease", {"account": "paper"})


def test_noop_lease_store_arg_is_accepted_but_unused():
    """The store arg is accepted for API parity with the v2 implementation.

    The composition root in P0-4 passes the same store to ledger, triplet, and
    lease; the v2 swap then uses it without changing the call site.
    """
    store = InMemoryNoSqlStore()
    lease = NoOpLease(store=store, account="paper")
    assert lease.held_by_me() is True


def test_lease_error_is_defined_for_v2_swap():
    """LeaseError is in the module signature so the exec layer can catch it
    from day one; the v1 stub never raises it."""
    assert issubclass(LeaseError, Exception)
