"""CLI: provision the v1 STORAGE-plane NoSQL tables (plan §4, §8; D5, D12).

Usage:
    uv run python -m personal_strat_pai.infra.oci.provision [--dry-run] [--prefix PREFIX]

Reads OCI config from the env / ``~/.oci/config`` (profile from
``OCI_CLI_PROFILE``; defaults to ``DEFAULT``). The NoSQL compartment is read
from ``OCI_NOSQL_COMPARTMENT_ID`` (or falls back to the tenancy's root
compartment if unset — the CLI warns in that case).

**The actual cloud run is CEO-gated.** Creating NoSQL tables incurs a small
monthly cost (small for the v1 workload, but non-zero). The CEO approves the
first provision before any cloud resources are created. ``--dry-run`` prints
the DDL it would run and the table list without touching the tenancy — the
default for the first review pass.

This module is the only place that constructs ``OciNoSqlStore`` with the real
OCI config; everywhere else (CI, tests, local dev) uses
``InMemoryNoSqlStore``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from personal_strat_pai.infra.oci.nosql_tables import (
    V1_TABLE_NAMES,
    provision_all,
)

__all__ = ["main"]


def _print_table(name: str, schema: Any) -> None:
    keys = ", ".join(f"{k}:{v}" for k, v in schema.key_schema.items())
    cap = (
        f"read={schema.capacity.get('read_units', 1)}u, "
        f"write={schema.capacity.get('write_units', 1)}u"
    )
    print(f"  - {name:<20} PK({keys})  capacity={cap}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m personal_strat_pai.infra.oci.provision",
        description=(
            "Provision the v1 STORAGE-plane NoSQL tables. CEO-gated: the first "
            "real run (no --dry-run) requires CEO sign-off — creating NoSQL "
            "tables incurs a small monthly cost. v1 creates 8 tables; the "
            "execution_lease table is NOT created (D12 — v2 / P0-5)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the DDL and table list without touching the tenancy. Default for first review.",
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("OCI_NOSQL_TABLE_PREFIX", ""),
        help="Table name prefix (e.g., 'personal_strat_pai_'). Defaults to $OCI_NOSQL_TABLE_PREFIX.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("OCI_CLI_PROFILE", "DEFAULT"),
        help="OCI config profile. Defaults to $OCI_CLI_PROFILE or 'DEFAULT'.",
    )
    parser.add_argument(
        "--compartment-id",
        default=os.environ.get("OCI_NOSQL_COMPARTMENT_ID", ""),
        help="OCI compartment OCID for the NoSQL tables. Defaults to $OCI_NOSQL_COMPARTMENT_ID.",
    )
    parser.add_argument(
        "--config-file",
        default=os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"),
        help="OCI config file path. Defaults to $OCI_CONFIG_FILE or ~/.oci/config.",
    )
    args = parser.parse_args(argv)

    print("# v1 STORAGE-plane NoSQL provisioning")
    print(f"#   profile: {args.profile}")
    print(f"#   config:  {args.config_file}")
    print(f"#   prefix:  {args.prefix!r}")
    print(f"#   tables:  {len(V1_TABLE_NAMES)} (NO execution_lease — D12)")
    print(f"#   dry_run: {args.dry_run}")
    print()

    if args.dry_run:
        # Dry run: use an in-memory store; provision_all logs the DDL via on_table.
        from personal_strat_pai.state.nosql import InMemoryNoSqlStore

        store: Any = InMemoryNoSqlStore()
        created = provision_all(
            store,
            table_prefix=args.prefix,
            dry_run=True,
            on_table=_print_table,
        )
        print()
        print(f"# dry-run OK: would create {len(created)} tables (no cloud resources touched).")
        print("# Review the DDL above; re-run without --dry-run after CEO sign-off.")
        return 0

    # Real run — CEO-gated. Construct the OCI-backed store.
    if not args.compartment_id:
        print(
            "ERROR: --compartment-id (or $OCI_NOSQL_COMPARTMENT_ID) is required for a real provision.",
            file=sys.stderr,
        )
        print(
            "The NoSQL compartment OCID is the Oracle Cloud compartment where the "
            "tables will be created. Find it in the OCI Console under Identity > "
            "Compartments. This is a CEO-gated step — the first provision requires "
            "CEO sign-off (plan §3.2, §15).",
            file=sys.stderr,
        )
        return 2

    from personal_strat_pai.state.nosql import OciNoSqlStore

    store = OciNoSqlStore(
        compartment_id=args.compartment_id,
        config_file=args.config_file,
        profile=args.profile,
    )
    try:
        created = provision_all(
            store,
            table_prefix=args.prefix,
            dry_run=False,
            on_table=_print_table,
        )
    except Exception as e:
        print(f"ERROR: provisioning failed: {e}", file=sys.stderr)
        return 1
    print()
    print(f"# OK: provisioned {len(created)} tables.")
    print("# Tables are idempotent — re-running is safe.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
