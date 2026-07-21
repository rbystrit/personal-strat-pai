"""State plane — Oracle NoSQL state, HIFO ledger, triplet machine (plan §8, D5).

Modules:
    nosql   — Oracle NoSQL access (Protocol + InMemory + OCI backend); the
              conditional-write helpers (put_if_absent / update_if_version /
              update_if_condition / delete_if_version) are the atomicity
              boundary for HIFO lot closure and triplet slot advance.
    ledger  — HIFO tax-lot selection (highest-cost-first) + ST(<365d)/LT(>=365d)
              split at the lot level (brief §1). Pure functions, property-tested.
    triplet — A->B->C state machine with 60-day immunization + 30-day
              wash-sale restricted slot + append-only slot_history (brief §1,
              plan §8). Pure transitions composed with the NoSQL fence.

v1 STORAGE-plane NoSQL tables (provisioned by infra/oci/nosql_tables.py):
    tax_lots, positions, triplet_state, realized_pnl, order_intent,
    risk_state, ibkr_session, sec_compliance. The ``execution_lease`` table is
    NOT created in v1 (single writer — D12); exec/lease.py is a no-op stub
    wired in v2 / P0-5.
"""

from __future__ import annotations

from personal_strat_pai.state.ledger import (
    HoldingClass,
    LedgerError,
    LedgerRepo,
    LotClosed,
    LotId,
    LotStatus,
    RealizedLot,
    SellRequest,
    SellResult,
    TaxLot,
    close_lot,
    hifo_select,
    holding_class,
    holding_days,
    realize_lot,
    split_proceeds,
    st_lt_split,
)
from personal_strat_pai.state.nosql import (
    ConditionalCheckFailed,
    InMemoryNoSqlStore,
    NoSqlRow,
    NoSqlStore,
    OciNoSqlStore,
    Row,
    TableNotProvisioned,
    row_version,
)
from personal_strat_pai.state.triplet import (
    IMMUNIZATION_DAYS,
    WASH_SALE_DAYS,
    TripletRepo,
    TripletSlot,
    TripletState,
    advance_on_loss,
    advance_slot,
    can_enter,
    lost_slot_window,
    seed_state,
    slot_immunized_until,
)

__all__ = [
    # triplet
    "IMMUNIZATION_DAYS",
    "ST_THRESHOLD_DAYS",
    "WASH_SALE_DAYS",
    # nosql
    "ConditionalCheckFailed",
    # ledger
    "HoldingClass",
    "InMemoryNoSqlStore",
    "LedgerError",
    "LedgerRepo",
    "LotClosed",
    "LotId",
    "LotStatus",
    "NoSqlRow",
    "NoSqlStore",
    "OciNoSqlStore",
    "RealizedLot",
    "Row",
    "SellRequest",
    "SellResult",
    "TableNotProvisioned",
    "TaxLot",
    "TripletRepo",
    "TripletSlot",
    "TripletState",
    "advance_on_loss",
    "advance_slot",
    "can_enter",
    "close_lot",
    "hifo_select",
    "holding_class",
    "holding_days",
    "lost_slot_window",
    "realize_lot",
    "row_version",
    "seed_state",
    "slot_immunized_until",
    "split_proceeds",
    "st_lt_split",
]

# Re-export the ST threshold from ledger so callers can import from state.
from personal_strat_pai.state.ledger import ST_THRESHOLD_DAYS
