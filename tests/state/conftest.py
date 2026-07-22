"""Shared fixtures for the state-plane test suite (plan §16).

The v1 state plane is exercised with the in-memory NoSQL backend so CI runs
without the ``oci`` SDK. The hypothesis property tests (HIFO, triplet, NoSQL
conditional-write atomicity) construct their stores from these fixtures.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import pytest

# Default: never hit the real OCI tenancy. The OciNoSqlStore tests are gated
# behind an explicit opt-in marker so CI never constructs the OCI client.
os.environ.setdefault("OCI_NOSQL_TABLE_PREFIX", "personal_strat_pai_test_")
os.environ.setdefault("OCI_CLI_PROFILE", "DEFAULT")


@pytest.fixture
def now() -> datetime:
    """A fixed 'now' (UTC) for deterministic state-machine tests."""
    return datetime(2026, 7, 21, 16, 0, tzinfo=UTC)


@pytest.fixture
def store():
    """A fresh in-memory NoSQL store with the 8 v1 tables provisioned."""
    from personal_strat_pai.infra.oci.nosql_tables import provision_all
    from personal_strat_pai.state.nosql import InMemoryNoSqlStore

    s = InMemoryNoSqlStore()
    provision_all(s)  # creates the 8 v1 tables; raises if execution_lease is requested
    return s


@pytest.fixture
def ledger_repo(store):
    """A LedgerRepo backed by the in-memory store (tax_lots + realized_pnl provisioned)."""
    from personal_strat_pai.state.ledger import LedgerRepo

    return LedgerRepo(store)


@pytest.fixture
def triplet_repo(store):
    """A TripletRepo backed by the in-memory store (triplet_state provisioned)."""
    from personal_strat_pai.state.triplet import TripletRepo

    return TripletRepo(store)


def make_lot(
    *,
    lot_id: str,
    bucket_id: int = 1,
    ticker: str = "XLB",
    triplet_slot: str = "A",
    qty: float = 100.0,
    cost_basis_per_share: float = 100.0,
    acquired_at: datetime | None = None,
    account: str = "paper",
    closed_qty: float = 0.0,
) -> dict[str, Any]:
    """A raw tax_lots row payload (the form ``store.put_if_absent`` expects)."""
    return {
        "lot_id": lot_id,
        "account": account,
        "bucket_id": bucket_id,
        "ticker": ticker,
        "triplet_slot": triplet_slot,
        "qty": qty,
        "closed_qty": closed_qty,
        "cost_basis_per_share": cost_basis_per_share,
        "acquired_at": (acquired_at or datetime(2026, 1, 1, tzinfo=UTC)).isoformat(),
        "status": "open",
        "wash_immunity_until": None,
    }


@pytest.fixture
def make_lot_factory():
    """Factory for raw tax_lots row payloads."""
    return make_lot
