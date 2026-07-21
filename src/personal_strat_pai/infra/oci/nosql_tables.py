"""Provision the v1 STORAGE-plane NoSQL tables (plan §4, §8; D5, D12).

The 8 v1 tables (NO ``execution_lease`` — D12 defers it to v2 / P0-5):

  tax_lots        — HIFO tax lots (one row per lot_id; the atomicity boundary
                    for HIFO selection + lot closure).
  positions       — aggregate positions keyed by (account, bucket_id, ticker).
                    Derived from tax_lots; updated on each fill.
  triplet_state   — A->B->C triplet machine state per bucket_id. The atomicity
                    boundary for triplet slot advance.
  realized_pnl    — append-only realized P&L per (lot_id, closed_at). One row
                    per closed lot slice; the backtest's tax-drag attribution
                    reads this for ST/LT gain splits.
  order_intent    — idempotent order intents keyed by client_order_id. The
                    fence against duplicate orders across restarts (and, in
                    v2, primary/backup).
  risk_state      — per-account risk state (kill switch, NAV cap, beta ceiling,
                    account DD stop). Read by the pre-trade checks.
  ibkr_session    — IBKR Gateway session material ref (Vault secret ref) per
                    account; used by the session lifecycle (P0-4).
  sec_compliance  — per-(ticker, month) SEC compliance verdict (PIMCO blocklist
                    + >=15 issuer check). v1-v3: static whitelist source.

**Not created in v1 (D12):** ``execution_lease`` — single writer, no fencing
needed. Re-introduced in P0-5 with the OCI Functions backup.

Idempotent: ``provision_all`` is safe to re-run. ``OciNoSqlStore.create_table``
is the per-table idempotent primitive; ``provision_all`` calls it for each of
the 8 tables in dependency order (tax_lots before positions, etc. — though
NoSQL has no FK constraints, the order is logical for audit logs).

Capacity: v1 defaults to small on-demand tables (1 read unit / 1 write unit
per table). The v1 workload is tiny (~45 tickers x ~13 buckets x a few hundred
lots, sub-hourly reads on the Risk Clock). The CEO can bump capacity via the
``--read-units`` / ``--write-units`` flags if Phase 0 shows throttling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from personal_strat_pai.state.nosql import NoSqlStore

__all__ = [
    "TABLE_SCHEMAS",
    "V1_TABLE_NAMES",
    "TableSchema",
    "provision_all",
]


@dataclass(frozen=True, slots=True)
class TableSchema:
    """A single NoSQL table's DDL inputs (plan §8).

    ``key_schema`` maps key-column-name -> Python short type ('str', 'int',
    'long', 'float', 'bytes'). The provisioner maps these to NoSQL DDL types
    (STRING / INTEGER / LONG / DOUBLE / BINARY).
    """

    name: str
    key_schema: dict[str, str]
    capacity: dict[str, int] = field(default_factory=lambda: {"read_units": 1, "write_units": 1})


# --- v1 STORAGE-plane table schemas --- #
# The 8 v1 tables (plan §8). NO `execution_lease` here — D12 defers it to v2.
TABLE_SCHEMAS: dict[str, TableSchema] = {
    "tax_lots": TableSchema(
        name="tax_lots",
        # lot_id is a string: "{account}:{bucket_id}:{ticker}:{slot}:{acquired_at_epoch_ms}:{nonce}"
        key_schema={"lot_id": "str"},
    ),
    "positions": TableSchema(
        name="positions",
        # Composite PK: (account, bucket_id, ticker). NoSQL supports composite keys.
        key_schema={"account": "str", "bucket_id": "int", "ticker": "str"},
    ),
    "triplet_state": TableSchema(
        name="triplet_state",
        key_schema={"bucket_id": "int"},
    ),
    "realized_pnl": TableSchema(
        name="realized_pnl",
        # Composite PK: (lot_id, closed_at). Allows multiple realized slices per lot.
        # closed_at stored as a string (ISO-8601) for stable lexicographic ordering.
        key_schema={"lot_id": "str", "closed_at": "str"},
    ),
    "order_intent": TableSchema(
        name="order_intent",
        # client_order_id is the idempotency key (plan §11).
        key_schema={"client_order_id": "str"},
    ),
    "risk_state": TableSchema(
        name="risk_state",
        key_schema={"account": "str"},
    ),
    "ibkr_session": TableSchema(
        name="ibkr_session",
        key_schema={"account": "str"},
    ),
    "sec_compliance": TableSchema(
        name="sec_compliance",
        # Composite PK: (ticker, month). `month` is "YYYY-MM".
        key_schema={"ticker": "str", "month": "str"},
    ),
}

# The 8 v1 tables in provisioning order. NO `execution_lease` (D12).
V1_TABLE_NAMES: tuple[str, ...] = (
    "tax_lots",
    "positions",
    "triplet_state",
    "realized_pnl",
    "order_intent",
    "risk_state",
    "ibkr_session",
    "sec_compliance",
)

# Tables that the v2 backup (P0-5) will add — listed for auditability only.
# The v1 provisioner MUST NOT create these.
V2_TABLE_NAMES: tuple[str, ...] = ("execution_lease",)


class ProvisionerProtocol(Protocol):
    """The minimal store surface ``provision_all`` needs."""

    def create_table(
        self,
        table: str,
        *,
        key_schema: dict[str, str],
        capacity: dict[str, int] | None = None,
    ) -> None: ...


def provision_all(
    store: NoSqlStore | ProvisionerProtocol,
    *,
    table_prefix: str = "",
    table_names: tuple[str, ...] = V1_TABLE_NAMES,
    dry_run: bool = False,
    on_table: Any | None = None,
) -> list[str]:
    """Provision the v1 NoSQL tables idempotently.

    Args:
        store: a ``NoSqlStore`` (in-memory or OCI). For the real tenancy use
            ``OciNoSqlStore``; for CI / dry-run use ``InMemoryNoSqlStore``.
        table_prefix: optional prefix (e.g., "personal_strat_pai_") to
            namespace tables in a shared tenancy. Empty by default.
        table_names: the tables to create. Defaults to ``V1_TABLE_NAMES``
            (the 8 v1 tables). v1 NEVER includes ``execution_lease`` (D12).
        dry_run: if True, only logs the DDL it would run and returns the
            prefixed names — does NOT call ``store.create_table``. Useful for
            the CEO-gated first provision to review before any cloud resource
            is created.
        on_table: optional callback ``(name, schema) -> None`` invoked per
            table after a successful create (or per table in dry_run). Used by
            the CLI to print progress.

    Returns the list of fully-prefixed table names that were created (or would
    be, in dry_run).

    Raises ``ValueError`` if ``table_names`` includes ``execution_lease``
    (v1 fence — D12).
    """
    if "execution_lease" in table_names:
        raise ValueError(
            "execution_lease is NOT a v1 table (D12 defers it to v2 / P0-5). "
            "Remove it from the table_names argument; the v1 provisioner must "
            "not create it."
        )

    created: list[str] = []
    for name in table_names:
        if name not in TABLE_SCHEMAS:
            raise ValueError(f"unknown table {name!r}; known v1 tables: {list(TABLE_SCHEMAS)}")
        schema = TABLE_SCHEMAS[name]
        full_name = f"{table_prefix}{name}" if table_prefix else name
        if dry_run:
            created.append(full_name)
            if on_table is not None:
                on_table(full_name, schema)
            continue
        store.create_table(
            full_name,
            key_schema=schema.key_schema,
            capacity=schema.capacity,
        )
        created.append(full_name)
        if on_table is not None:
            on_table(full_name, schema)
    return created
