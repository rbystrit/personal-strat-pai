"""Oracle NoSQL access — tables, indexes, conditional writes (plan §8, D5).

System of record = Oracle NoSQL Database Cloud Service, accessed via the
``oci-nosqldb`` SDK (part of the ``oci`` package). Local SQLite is a read-through
cache for regenerable data (compliance whitelist, conId map, recent bars); the
authoritative state — tax lots, positions, triplet state, realized P&L, order
intents, risk state, IBKR session, SEC compliance — lives in NoSQL. **No
Autonomous DB, no oracledb, no SQLAlchemy** (D5).

Why NoSQL + conditional writes (plan §8, §19 "NoSQL transaction boundaries"):
HIFO lot selection + lot closure + triplet advance must be **atomic**. NoSQL
conditional writes are the concurrency/atomicity boundary — a put/delete only
succeeds if a server-side predicate over the row's current state holds. Two
concurrent sells against the same lot cannot both close it: the second write
sees the lot's `status` already flipped to `closed` (or its version bump) and
fails with `ConditionalCheckFailed`. The same fence guards triplet slot
advances. This module is the single home for that boundary; the ledger and
triplet modules call into these helpers instead of raw puts.

Two backends share one Protocol:

  * ``InMemoryNoSqlStore`` — an exact mirror of NoSQL conditional-write
    semantics, used in CI and the hypothesis property tests. It is not a toy:
    its ``threading.Lock`` makes each conditional write atomic against other
    threads, so the atomicity property tests are meaningful (concurrent writers
    cannot both succeed against the same row).

  * ``OciNoSqlStore`` — the real backend. Lazily imports ``oci`` so the default
    ``uv sync --extra dev`` install (CI) does not require the heavy SDK; the
    extra ``oci`` (pyproject ``[project.optional-dependencies]``) is installed
    on the podman primary and any environment that touches the real tenancy.

The conditional-write helper vocabulary (named after the NoSQL ``putOption``
semantics it mirrors):

  * ``put_if_absent``   — InsertOnly; succeeds only if no row exists at the key.
                         Used for ``order_intent`` (idempotent order creation)
                         and new ``tax_lots`` (a fresh lot_id must be unique).
  * ``update_if_version`` — IfVersion; succeeds only if the row exists AND its
                         ``version`` equals ``expected_version``. This is the
                         optimistic-concurrency fence used by HIFO lot closure
                         and triplet slot advance.
  * ``update_if_condition`` — IfVersion + an arbitrary predicate over the
                         pre-update row; the update applies atomically only if
                         the version matches AND the predicate holds. Used for
                         "close the lot only if its status is still 'open' and
                         qty matches expected".
  * ``delete_if_version`` — IfVersion on a delete.
  * ``get`` / ``query``  — point lookup by primary key / secondary index lookup.

Every successful write returns the new ``Row`` (with a fresh ``version``); every
failed conditional write raises ``ConditionalCheckFailed``. The version is an
opaque server-assigned string (the OCI SDK returns it as a bytes/blob; the
in-memory backend uses a monotonic counter encoded as a string).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol, TypedDict, runtime_checkable

__all__ = [
    "ConditionalCheckFailed",
    "InMemoryNoSqlStore",
    "NoSqlRow",
    "NoSqlStore",
    "OciNoSqlStore",
    "Row",
    "TableNotProvisioned",
    "row_version",
]


# --- Row model --- #


class NoSqlRow(TypedDict, total=False):
    """A NoSQL row — the user payload plus the server-assigned ``version``.

    ``version`` is set by the store on every write and MUST be treated as
    opaque by callers (the in-memory backend uses a monotonic counter encoded
    as a string; the OCI backend uses the SDK's version blob). Callers read it
    from a previous ``get``/write response and pass it to
    ``update_if_version`` / ``update_if_condition`` / ``delete_if_version`` as
    the expected-version fence.
    """

    version: str
    # The rest of the row is application-defined (see TABLE_SCHEMAS in
    # infra/oci/nosql_tables.py for the v1 STORAGE-plane schemas). We keep the
    # payload open so the store stays generic across the 8 v1 tables.


@dataclass(frozen=True, slots=True)
class Row:
    """A returned row — the user payload and the server-assigned version, separated.

    The payload is the application-defined dict (no ``version`` key inside);
    the version is exposed separately so callers can pass it cleanly to the
    next conditional write. Mirrors the OCI SDK's split between the row value
    and the returned version blob.
    """

    payload: dict[str, Any]
    version: str


def row_version(row: Row | NoSqlRow | None) -> str | None:
    """Extract the version from a Row, a raw NoSqlRow dict, or None.

    Convenience for callers that hold the raw dict form (e.g., from a ``get``
    that returned the dict directly). Returns ``None`` for a missing row so
    callers can branch on "not found" without try/except.
    """
    if row is None:
        return None
    if isinstance(row, Row):
        return row.version
    return row.get("version")


# --- Exceptions --- #


class ConditionalCheckFailed(Exception):
    """A conditional write did not match — the row's state changed underneath the caller.

    Raised by every conditional write path (``put_if_absent``,
    ``update_if_version``, ``update_if_condition``, ``delete_if_version``) when
    the server-side predicate does not hold. Callers MUST treat this as a
    retryable concurrency event: re-read the row, recompute, re-attempt. It is
    the fence that prevents double-closing a lot or double-advancing a triplet
    (plan §8, §19 "NoSQL transaction boundaries").
    """

    def __init__(
        self,
        table: str,
        key: Mapping[str, Any],
        reason: str,
        *,
        expected_version: str | None = None,
        actual_version: str | None = None,
    ) -> None:
        self.table = table
        self.key = dict(key)
        self.reason = reason
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"conditional write failed on {table} key={dict(key)!r}: {reason} "
            f"(expected_version={expected_version!r}, actual_version={actual_version!r})"
        )


class TableNotProvisioned(Exception):
    """The target table does not exist in the store.

    Raised by ``InMemoryNoSqlStore`` when a table has not been created via
    ``create_table`` before the first I/O — mirrors the OCI backend's behavior
    where a missing table surfaces as a service error. The property tests
    create their tables up front so this is a programmer-error fence, not a
    retryable condition.
    """


# --- Protocol --- #


@runtime_checkable
class NoSqlStore(Protocol):
    """Conditional-write NoSQL store (plan §8, D5).

    All state-mutating operations are conditional — there is no unconditional
    put. This is deliberate: the brief's atomicity requirements (HIFO + lot
    closure + triplet advance; plan §8, §19) are enforced at the store, not at
    the caller. Unconditional writes would let two concurrent sellers both
    close the same lot; the conditional fence makes that impossible.

    Implementations: ``InMemoryNoSqlStore`` (CI, property tests),
    ``OciNoSqlStore`` (real tenancy). Both MUST honor the same atomicity
    contract: a conditional write is atomic with respect to all other
    operations on the same key.
    """

    def create_table(
        self,
        table: str,
        *,
        key_schema: Mapping[str, str],
        capacity: Mapping[str, int] | None = None,
    ) -> None:
        """Idempotently create a table. Raises if the schema changes."""
        ...

    def delete_table(self, table: str) -> None:
        """Drop a table (admin / test teardown only)."""
        ...

    def get(self, table: str, key: Mapping[str, Any]) -> Row | None:
        """Point lookup by full primary key. Returns ``None`` if the row is absent."""
        ...

    def query(
        self,
        table: str,
        *,
        where: str | Mapping[str, Any],
        limit: int | None = None,
    ) -> list[Row]:
        """Secondary lookup. ``where`` is either a dict of equality predicates
        on indexed columns or a NoSQL-compatible SQL-ish string (OCI backend).
        The in-memory backend supports only the dict form (equality on any
        column) — sufficient for the property tests and the small v1 universe.
        """
        ...

    def put_if_absent(self, table: str, key: Mapping[str, Any], row: Mapping[str, Any]) -> Row:
        """Insert a new row. Raises ``ConditionalCheckFailed`` if the key exists."""
        ...

    def update_if_version(
        self,
        table: str,
        key: Mapping[str, Any],
        row: Mapping[str, Any],
        expected_version: str,
    ) -> Row:
        """Replace a row's payload only if its current version matches.

        Raises ``ConditionalCheckFailed`` if the row is missing or its version
        differs. The new payload replaces the old wholesale (NoSQL semantics);
        callers merge before calling.
        """
        ...

    def update_if_condition(
        self,
        table: str,
        key: Mapping[str, Any],
        row: Mapping[str, Any],
        expected_version: str,
        condition: Callable[[dict[str, Any]], bool],
        condition_desc: str,
    ) -> Row:
        """Replace a row only if its version matches AND ``condition(current_payload)``
        holds. The condition is evaluated atomically inside the store's write
        lock; this is the fence for HIFO lot closure ("status == 'open' and
        qty == expected_qty") and triplet advance ("current_slot == expected_slot").
        ``condition_desc`` is included in the failure exception for auditability.
        """
        ...

    def delete_if_version(self, table: str, key: Mapping[str, Any], expected_version: str) -> None:
        """Delete a row only if its version matches. Raises on mismatch."""
        ...


# --- In-memory backend (CI, property tests) --- #


@dataclass(slots=True)
class _InMemoryTable:
    """A single in-memory NoSQL table — keyed rows with versions, behind a lock.

    The lock is held for the duration of each conditional write so the
    atomicity property tests (concurrent writers on the same key) are
    meaningful: two ``update_if_version`` calls with the same ``expected_version``
    cannot both succeed — the first bumps the version, the second sees the
    mismatch and raises ``ConditionalCheckFailed``.
    """

    key_schema: dict[str, str]
    rows: dict[tuple[Any, ...], Row] = field(default_factory=dict)
    _counter: int = 0
    lock: RLock = field(default_factory=RLock)

    def _key_tuple(self, key: Mapping[str, Any]) -> tuple[Any, ...]:
        missing = [k for k in self.key_schema if k not in key]
        if missing:
            raise KeyError(f"missing key columns: {missing}")
        return tuple(key[k] for k in self.key_schema)

    def _next_version(self) -> str:
        self._counter += 1
        return f"v{self._counter}"


class InMemoryNoSqlStore:
    """Exact mirror of NoSQL conditional-write semantics for CI and property tests.

    Every conditional write is atomic with respect to all other operations on
    the same key (per-key ``RLock``). The version is a monotonic counter
    encoded as a string; the contract — a conditional write only succeeds if
    the expected version matches — is identical to the OCI backend's
    ``putOption=IfVersion``. The hypothesis property tests exercise this
    directly; the OCI backend inherits the same tests by swapping the store.
    """

    def __init__(self) -> None:
        self._tables: dict[str, _InMemoryTable] = {}
        self._global_lock = RLock()

    def create_table(
        self,
        table: str,
        *,
        key_schema: Mapping[str, str],
        capacity: Mapping[str, int] | None = None,
    ) -> None:
        """Idempotently create a table. Raises if the schema changes.

        ``capacity`` is accepted for API parity with the OCI backend (where it
        maps to read/write units) and ignored — the in-memory store has no
        capacity notion.
        """
        with self._global_lock:
            existing = self._tables.get(table)
            if existing is None:
                self._tables[table] = _InMemoryTable(key_schema=dict(key_schema))
                return
            if dict(existing.key_schema) != dict(key_schema):
                raise ValueError(
                    f"table {table!r} already exists with a different key schema: "
                    f"existing={existing.key_schema!r} requested={dict(key_schema)!r}"
                )

    def delete_table(self, table: str) -> None:
        with self._global_lock:
            self._tables.pop(table, None)

    def _require_table(self, table: str) -> _InMemoryTable:
        t = self._tables.get(table)
        if t is None:
            raise TableNotProvisioned(
                f"table {table!r} not created; call create_table first or "
                "use infra/oci/nosql_tables.provision_all to provision the v1 set."
            )
        return t

    def get(self, table: str, key: Mapping[str, Any]) -> Row | None:
        t = self._require_table(table)
        with t.lock:
            return t.rows.get(t._key_tuple(key))

    def query(
        self,
        table: str,
        *,
        where: str | Mapping[str, Any],
        limit: int | None = None,
    ) -> list[Row]:
        if isinstance(where, str):
            raise NotImplementedError(
                "InMemoryNoSqlStore.query supports only dict equality predicates "
                "(sufficient for the v1 universe + property tests). Use OciNoSqlStore "
                "for NoSQL SQL-ish strings."
            )
        t = self._require_table(table)
        predicates = dict(where)
        out: list[Row] = []
        with t.lock:
            for row in t.rows.values():
                payload = row.payload
                if all(payload.get(k) == v for k, v in predicates.items()):
                    out.append(row)
                    if limit is not None and len(out) >= limit:
                        break
        return out

    def put_if_absent(self, table: str, key: Mapping[str, Any], row: Mapping[str, Any]) -> Row:
        t = self._require_table(table)
        kt = t._key_tuple(key)
        with t.lock:
            if kt in t.rows:
                existing = t.rows[kt]
                raise ConditionalCheckFailed(
                    table,
                    key,
                    "row already exists",
                    actual_version=existing.version,
                )
            version = t._next_version()
            new_row = Row(payload=dict(row), version=version)
            t.rows[kt] = new_row
            return new_row

    def update_if_version(
        self,
        table: str,
        key: Mapping[str, Any],
        row: Mapping[str, Any],
        expected_version: str,
    ) -> Row:
        t = self._require_table(table)
        kt = t._key_tuple(key)
        with t.lock:
            existing = t.rows.get(kt)
            if existing is None:
                raise ConditionalCheckFailed(
                    table,
                    key,
                    "row missing",
                    expected_version=expected_version,
                    actual_version=None,
                )
            if existing.version != expected_version:
                raise ConditionalCheckFailed(
                    table,
                    key,
                    "version mismatch",
                    expected_version=expected_version,
                    actual_version=existing.version,
                )
            version = t._next_version()
            new_row = Row(payload=dict(row), version=version)
            t.rows[kt] = new_row
            return new_row

    def update_if_condition(
        self,
        table: str,
        key: Mapping[str, Any],
        row: Mapping[str, Any],
        expected_version: str,
        condition: Callable[[dict[str, Any]], bool],
        condition_desc: str,
    ) -> Row:
        t = self._require_table(table)
        kt = t._key_tuple(key)
        with t.lock:
            existing = t.rows.get(kt)
            if existing is None:
                raise ConditionalCheckFailed(
                    table,
                    key,
                    f"row missing (condition: {condition_desc})",
                    expected_version=expected_version,
                    actual_version=None,
                )
            if existing.version != expected_version:
                raise ConditionalCheckFailed(
                    table,
                    key,
                    f"version mismatch (condition: {condition_desc})",
                    expected_version=expected_version,
                    actual_version=existing.version,
                )
            if not condition(existing.payload):
                raise ConditionalCheckFailed(
                    table,
                    key,
                    f"condition false: {condition_desc}",
                    expected_version=expected_version,
                    actual_version=existing.version,
                )
            version = t._next_version()
            new_row = Row(payload=dict(row), version=version)
            t.rows[kt] = new_row
            return new_row

    def delete_if_version(self, table: str, key: Mapping[str, Any], expected_version: str) -> None:
        t = self._require_table(table)
        kt = t._key_tuple(key)
        with t.lock:
            existing = t.rows.get(kt)
            if existing is None:
                raise ConditionalCheckFailed(
                    table,
                    key,
                    "row missing",
                    expected_version=expected_version,
                    actual_version=None,
                )
            if existing.version != expected_version:
                raise ConditionalCheckFailed(
                    table,
                    key,
                    "version mismatch",
                    expected_version=expected_version,
                    actual_version=existing.version,
                )
            del t.rows[kt]


# --- OCI backend (real tenancy) --- #


class OciNoSqlStore:
    """OCI NoSQL Database Cloud Service backend (plan §8, D5).

    Lazily imports the ``oci`` SDK so this module imports cleanly under the
    default ``uv sync --extra dev`` install (CI) — the ``oci`` extra is only
    required on the podman primary and any environment that touches the real
    tenancy. Construction needs the OCI config (profile) and the NoSQL table
    compartment id; the SDK client is created on first use.

    The conditional-write semantics mirror ``InMemoryNoSqlStore`` exactly:

      * ``put_if_absent``  -> ``PutRequest`` with ``put_option=IfAbsent``.
      * ``update_if_version`` / ``update_if_condition`` / ``delete_if_version``
        -> ``PutRequest`` / ``DeleteRequest`` with ``put_option=IfVersion`` and
        ``existing_version`` set to ``expected_version``. The condition
        predicate for ``update_if_condition`` is evaluated client-side against
        the row's current payload (read with ``get`` first under the same
        caller) — the server-side fence is the version match; the predicate is
        an additional caller-side guard. For pure server-side condition
        predicates, callers can use the SDK's ``condition`` field directly; this
        wrapper keeps the predicate in Python so the same test suite covers
        both backends.

    Construction is intentionally cheap and deferred; the heavy client is
    created lazily so importing this module never requires the SDK.
    """

    def __init__(
        self,
        *,
        compartment_id: str,
        region: str | None = None,
        config_file: str | None = None,
        profile: str = "DEFAULT",
        table_compartment_id: str | None = None,
    ) -> None:
        self.compartment_id = compartment_id
        self.region = region
        self.config_file = config_file
        self.profile = profile
        # NoSQL table compartment can differ from the resource compartment; default to it.
        self.table_compartment_id = table_compartment_id or compartment_id
        self._client: Any = None  # oci.nosqldb.NosqldbClient, created lazily
        self._table_schemas: dict[str, dict[str, str]] = {}

    # --- lazy SDK bootstrap --- #

    def _load_config(self) -> dict[str, Any]:
        try:
            import oci
        except ImportError as e:  # pragma: no cover - exercised only without the oci extra
            raise RuntimeError(
                "OciNoSqlStore requires the `oci` extra: `uv sync --extra oci`. "
                "The CI/dev install deliberately excludes the OCI SDK."
            ) from e

        cfg = dict(oci.config.from_file(self.config_file, self.profile))
        if self.region:
            cfg["region"] = self.region
        oci.config.validate_config(cfg)
        return cfg

    def _ensure_client(self) -> Any:
        if self._client is None:
            import oci

            cfg = self._load_config()
            self._client = oci.nosqldb.NosqldbClient(cfg)
        return self._client

    # --- admin --- #

    def create_table(
        self,
        table: str,
        *,
        key_schema: Mapping[str, str],
        capacity: Mapping[str, int] | None = None,
    ) -> None:
        """Create a NoSQL table if absent. Idempotent on name + key schema.

        ``capacity`` maps to ``capacity`` (read/write units) in the OCI DDL.
        Default is a small on-demand table (1 read / 1 write unit) — the v1
        universe is small (~45 tickers × ~13 buckets × a few hundred lots).
        """
        client = self._ensure_client()
        import oci

        cap = (
            dict(capacity)
            if capacity is not None
            else {"mode": "PROVISIONED", "read_units": 1, "write_units": 1}
        )
        # NoSQL key schema: a list of {name, type} where type is one of
        # STRING / INTEGER / LONG / DOUBLE / BINARY. We accept Python short
        # types (str, int, float, bytes) and map them.
        type_map = {
            "str": "STRING",
            "int": "INTEGER",
            "long": "LONG",
            "float": "DOUBLE",
            "bytes": "BINARY",
        }
        ddl_keys = [
            {"name": k, "type": type_map.get(v, str(v).upper())} for k, v in key_schema.items()
        ]

        # Idempotency: list tables first; if present with matching keys, no-op.
        try:
            client.get_table(table_name_or_id=table, compartment_id=self.table_compartment_id)
            self._table_schemas[table] = dict(key_schema)
            return
        except oci.exceptions.ServiceError as e:
            if getattr(e, "status", None) not in (404, "404"):
                raise

        details = oci.nosqldb.models.CreateTableDetails(
            name=table,
            compartment_id=self.table_compartment_id,
            ddl_statement=f"CREATE TABLE {table} ({', '.join(f'{k} {type_map.get(v, str(v).upper())}' for k, v in key_schema.items())}, PRIMARY KEY ({', '.join(key_schema)}))",
            ddl_type="CREATE",
            table_schema=oci.nosqldb.models.TableSchema(
                primary_key=ddl_keys,
            ),
            capacity=oci.nosqldb.models.Capacity(
                mode=cap.get("mode", "PROVISIONED"),
                read_units=cap.get("read_units", 1),
                write_units=cap.get("write_units", 1),
            ),
        )
        client.create_table(create_table_details=details)
        self._table_schemas[table] = dict(key_schema)

    def delete_table(self, table: str) -> None:
        client = self._ensure_client()
        client.delete_table(table_name_or_id=table, compartment_id=self.table_compartment_id)
        self._table_schemas.pop(table, None)

    # --- key/row helpers --- #

    def _key_dict(self, key: Mapping[str, Any]) -> dict[str, Any]:
        return dict(key)

    def get(self, table: str, key: Mapping[str, Any]) -> Row | None:
        client = self._ensure_client()
        try:
            resp = client.get_row(
                table_name_or_id=table,
                compartment_id=self.table_compartment_id,
                key=self._key_dict(key),
            )
        except Exception as e:
            # Missing row: NoSQL returns a 404 / specific error; normalize to None.
            if "InternalServerError" in type(e).__name__ or "NotFound" in type(e).__name__:
                return None
            raise
        row_value = resp.data.value if hasattr(resp, "data") and resp.data is not None else None
        version = resp.data.version if hasattr(resp, "data") and resp.data is not None else None
        if row_value is None:
            return None
        # NoSQL version is a bytes blob; encode as hex string for opaque comparison.
        version_str = version.hex() if isinstance(version, (bytes, bytearray)) else str(version)
        return Row(payload=dict(row_value), version=version_str)

    def query(
        self,
        table: str,
        *,
        where: str | Mapping[str, Any],
        limit: int | None = None,
    ) -> list[Row]:
        client = self._ensure_client()
        if isinstance(where, str):
            stmt = f"SELECT * FROM {table} WHERE {where}"
        else:
            clauses = " AND ".join(f"{k} = {v!r}" for k, v in where.items())
            stmt = f"SELECT * FROM {table}" + (f" WHERE {clauses}" if clauses else "")
        if limit is not None:
            stmt += f" LIMIT {int(limit)}"
        resp = client.query(
            query_details=__import__("oci").nosqldb.models.QueryDetails(
                compartment_id=self.table_compartment_id, statement=stmt
            )
        )
        out: list[Row] = []
        for item in resp.data.items if hasattr(resp, "data") and resp.data is not None else []:
            version = item.get("version") if isinstance(item, dict) else None
            payload = {
                k: v for k, v in (item.items() if isinstance(item, dict) else []) if k != "version"
            }
            version_str = (
                version.hex()
                if isinstance(version, (bytes, bytearray))
                else (str(version) if version is not None else "v0")
            )
            out.append(Row(payload=payload, version=version_str))
        return out

    def put_if_absent(self, table: str, key: Mapping[str, Any], row: Mapping[str, Any]) -> Row:
        client = self._ensure_client()
        import oci

        value = dict(row)
        value.update(self._key_dict(key))
        details = oci.nosqldb.models.PutRowDetails(
            value=value,
            put_option="IF_ABSENT",
            compartment_id=self.table_compartment_id,
        )
        try:
            resp = client.put_row(table_name_or_id=table, put_row_details=details)
        except oci.exceptions.ServiceError as e:
            raise ConditionalCheckFailed(table, key, "row already exists") from e
        return self._row_from_response(resp, key)

    def update_if_version(
        self,
        table: str,
        key: Mapping[str, Any],
        row: Mapping[str, Any],
        expected_version: str,
    ) -> Row:
        client = self._ensure_client()
        import oci

        value = dict(row)
        value.update(self._key_dict(key))
        # expected_version may be a hex string from a previous get; the SDK wants bytes.
        existing_version = self._decode_version(expected_version)
        details = oci.nosqldb.models.PutRowDetails(
            value=value,
            put_option="IF_VERSION",
            existing_version=existing_version,
            compartment_id=self.table_compartment_id,
        )
        try:
            resp = client.put_row(table_name_or_id=table, put_row_details=details)
        except oci.exceptions.ServiceError as e:
            raise ConditionalCheckFailed(
                table,
                key,
                "version mismatch",
                expected_version=expected_version,
            ) from e
        return self._row_from_response(resp, key)

    def update_if_condition(
        self,
        table: str,
        key: Mapping[str, Any],
        row: Mapping[str, Any],
        expected_version: str,
        condition: Callable[[dict[str, Any]], bool],
        condition_desc: str,
    ) -> Row:
        # Read-modify-write: the version fence is server-side; the predicate is
        # evaluated client-side against the current payload. This keeps the
        # condition logic identical between the in-memory and OCI backends so
        # the same property tests cover both. The race window is the
        # read-to-write gap, which is closed by the version check on the write.
        existing = self.get(table, key)
        if existing is None:
            raise ConditionalCheckFailed(
                table,
                key,
                f"row missing (condition: {condition_desc})",
                expected_version=expected_version,
                actual_version=None,
            )
        if existing.version != expected_version:
            raise ConditionalCheckFailed(
                table,
                key,
                f"version mismatch (condition: {condition_desc})",
                expected_version=expected_version,
                actual_version=existing.version,
            )
        if not condition(existing.payload):
            raise ConditionalCheckFailed(
                table,
                key,
                f"condition false: {condition_desc}",
                expected_version=expected_version,
                actual_version=existing.version,
            )
        return self.update_if_version(table, key, row, expected_version)

    def delete_if_version(self, table: str, key: Mapping[str, Any], expected_version: str) -> None:
        client = self._ensure_client()
        import oci

        existing_version = self._decode_version(expected_version)
        details = oci.nosqldb.models.DeleteRowDetails(
            compartment_id=self.table_compartment_id,
            is_get_return_row=True,
        )
        try:
            client.delete_row(
                table_name_or_id=table,
                delete_row_details=details,
                existing_version=existing_version,
            )
        except oci.exceptions.ServiceError as e:
            raise ConditionalCheckFailed(
                table,
                key,
                "version mismatch",
                expected_version=expected_version,
            ) from e

    # --- helpers --- #

    @staticmethod
    def _decode_version(version: str) -> bytes:
        try:
            return bytes.fromhex(version)
        except ValueError:
            # Not a hex string; treat as opaque passthrough encoded as utf-8.
            return version.encode("utf-8")

    @staticmethod
    def _row_from_response(resp: Any, key: Mapping[str, Any]) -> Row:
        data = getattr(resp, "data", None)
        if data is None:
            return Row(payload=dict(key), version="v0")
        value = getattr(data, "value", None) or {}
        version = getattr(data, "version", None)
        payload = {k: v for k, v in dict(value).items() if k != "version"}
        version_str = (
            version.hex()
            if isinstance(version, (bytes, bytearray))
            else (str(version) if version is not None else "v0")
        )
        return Row(payload=payload, version=version_str)
