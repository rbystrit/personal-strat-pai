"""OCI infra-as-code — v1 STORAGE PLANE ONLY (plan §4, §8; D11/D12).

v1 scope (P0-2): NoSQL Database Cloud Service tables for the state plane.
**No OCI Functions infra in v1** (D12 — backup deferred to v2 / P0-5). The
Functions packaging, the OCI Resource Scheduler wiring, and the
``execution_lease`` table are P0-5 deliverables; the provisioner here MUST NOT
create them.

Modules:
    nosql_tables — the 8 v1 table schemas (DDL, key schema, capacity) and the
                   idempotent ``provision_all`` driver.
    provision    — CLI entrypoint: ``uv run python -m personal_strat_pai.infra.oci.provision``.
                   Reads OCI config from the env / ~/.oci/config (profile from
                   ``OCI_CLI_PROFILE``); creates the 8 tables idempotently;
                   supports ``--dry-run`` to print the DDL without touching the
                   tenancy. **The actual cloud run is CEO-gated**: creating
                   NoSQL tables incurs a monthly cost (small for the v1
                   workload, but non-zero); the CEO approves the first
                   provision before any cloud resources are created.

The provisioner is Python (not Terraform) to keep the v1 toolchain single-language
(uv + Python everywhere — D13). It uses the same ``oci`` SDK the
``OciNoSqlStore`` backend uses; the table schemas are the single source of
truth shared by ``provision_all`` (creates the tables) and
``OciNoSqlStore.create_table`` (idempotent re-creation if a table is missing).
"""

from __future__ import annotations

from personal_strat_pai.infra.oci.nosql_tables import (
    TABLE_SCHEMAS,
    V1_TABLE_NAMES,
    provision_all,
)

__all__ = [
    "TABLE_SCHEMAS",
    "V1_TABLE_NAMES",
    "provision_all",
]
