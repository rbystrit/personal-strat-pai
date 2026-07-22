"""Infra-as-code for the v1 STORAGE plane (plan §4, §8; D11/D12).

v1: Oracle NoSQL Database Cloud Service tables only. No OCI Functions, no
``execution_lease`` table, no Resource Scheduler (all P0-5 / v2 — D12).
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
