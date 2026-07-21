"""Tests for infra/oci/nosql_tables.py — v1 STORAGE-plane provisioning (plan §4, §8; D12).

Verifies:
  * The 8 v1 tables are defined with the correct key schemas.
  * ``execution_lease`` is NOT in V1_TABLE_NAMES (D12 fence).
  * ``provision_all`` rejects ``execution_lease`` if a caller tries to slip it in.
  * ``provision_all`` is idempotent (re-running is safe).
  * The dry-run path does not touch the store.
  * The CLI ``--dry-run`` prints the 8 tables without constructing the OCI client.
"""

from __future__ import annotations

import pytest

from personal_strat_pai.infra.oci.nosql_tables import (
    TABLE_SCHEMAS,
    V1_TABLE_NAMES,
    V2_TABLE_NAMES,
    provision_all,
)
from personal_strat_pai.state.nosql import InMemoryNoSqlStore


def test_v1_table_names_excludes_execution_lease():
    """D12 fence: execution_lease is NOT in v1 (single writer, no fencing)."""
    assert "execution_lease" not in V1_TABLE_NAMES
    assert V2_TABLE_NAMES == ("execution_lease",)


def test_v1_table_count_is_8():
    """Plan §8: 8 v1 tables (tax_lots, positions, triplet_state, realized_pnl,
    order_intent, risk_state, ibkr_session, sec_compliance)."""
    assert len(V1_TABLE_NAMES) == 8
    expected = {
        "tax_lots",
        "positions",
        "triplet_state",
        "realized_pnl",
        "order_intent",
        "risk_state",
        "ibkr_session",
        "sec_compliance",
    }
    assert set(V1_TABLE_NAMES) == expected


def test_table_schemas_match_plan_section_8():
    """Spot-check the key schemas against plan §8."""
    assert TABLE_SCHEMAS["tax_lots"].key_schema == {"lot_id": "str"}
    assert TABLE_SCHEMAS["positions"].key_schema == {
        "account": "str",
        "bucket_id": "int",
        "ticker": "str",
    }
    assert TABLE_SCHEMAS["triplet_state"].key_schema == {"bucket_id": "int"}
    assert TABLE_SCHEMAS["realized_pnl"].key_schema == {"lot_id": "str", "closed_at": "str"}
    assert TABLE_SCHEMAS["order_intent"].key_schema == {"client_order_id": "str"}
    assert TABLE_SCHEMAS["risk_state"].key_schema == {"account": "str"}
    assert TABLE_SCHEMAS["ibkr_session"].key_schema == {"account": "str"}
    assert TABLE_SCHEMAS["sec_compliance"].key_schema == {"ticker": "str", "month": "str"}


def test_provision_all_creates_8_tables_in_memory():
    s = InMemoryNoSqlStore()
    created = provision_all(s)
    assert len(created) == 8
    assert set(created) == set(V1_TABLE_NAMES)
    # Each table is now usable (no TableNotProvisioned).
    from personal_strat_pai.state.nosql import TableNotProvisioned

    for name in V1_TABLE_NAMES:
        try:
            s.get(name, {k: "x" for k in TABLE_SCHEMAS[name].key_schema})
        except TableNotProvisioned as e:
            pytest.fail(f"table {name} not provisioned: {e}")
        except Exception:
            pass  # key-mismatch errors are fine; we only care the table exists


def test_provision_all_idempotent():
    """Re-running provision_all is safe — no error on the second pass."""
    s = InMemoryNoSqlStore()
    provision_all(s)
    provision_all(s)  # no raise
    # Sanity: a write/read still works.
    s.put_if_absent("tax_lots", {"lot_id": "L1"}, {"qty": 10.0})
    assert s.get("tax_lots", {"lot_id": "L1"}).payload["qty"] == 10.0


def test_provision_all_rejects_execution_lease():
    """D12 fence: provision_all refuses to create execution_lease in v1."""
    s = InMemoryNoSqlStore()
    with pytest.raises(ValueError, match="execution_lease is NOT a v1 table"):
        provision_all(s, table_names=("tax_lots", "execution_lease"))


def test_provision_all_rejects_unknown_table():
    s = InMemoryNoSqlStore()
    with pytest.raises(ValueError, match="unknown table"):
        provision_all(s, table_names=("tax_lots", "made_up_table"))


def test_provision_all_with_prefix():
    s = InMemoryNoSqlStore()
    created = provision_all(s, table_prefix="personal_strat_pai_")
    assert all(name.startswith("personal_strat_pai_") for name in created)
    assert "personal_strat_pai_tax_lots" in created


def test_provision_all_dry_run_does_not_touch_store():
    """Dry-run returns the table names but does not create them."""
    s = InMemoryNoSqlStore()
    created = provision_all(s, dry_run=True)
    assert len(created) == 8
    # The store should not have any tables created.
    from personal_strat_pai.state.nosql import TableNotProvisioned

    with pytest.raises(TableNotProvisioned):
        s.get("tax_lots", {"lot_id": "x"})


def test_provision_all_on_table_callback():
    """The on_table callback is invoked per table (used by the CLI for progress)."""
    s = InMemoryNoSqlStore()
    seen: list[str] = []
    provision_all(s, on_table=lambda name, schema: seen.append(name))
    assert len(seen) == 8
    assert seen == list(V1_TABLE_NAMES)


def test_cli_dry_run_prints_8_tables(capsys):
    """The CLI --dry-run prints the 8 v1 tables without touching the tenancy."""
    from personal_strat_pai.infra.oci.provision import main

    rc = main(["--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "v1 STORAGE-plane NoSQL provisioning" in captured.out
    assert "NO execution_lease" in captured.out
    for name in V1_TABLE_NAMES:
        assert name in captured.out
    # The dry-run must not require compartment-id.
    assert "would create 8 tables" in captured.out


def test_cli_real_run_requires_compartment_id(capsys, monkeypatch):
    """The real run is CEO-gated: it requires --compartment-id (or the env var)."""
    from personal_strat_pai.infra.oci.provision import main

    monkeypatch.delenv("OCI_NOSQL_COMPARTMENT_ID", raising=False)
    # No --dry-run flag => real-run path; missing --compartment-id should return 2.
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "compartment-id" in captured.err or "compartment-id" in captured.out
