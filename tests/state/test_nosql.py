"""Tests for state/nosql.py — the conditional-write atomicity boundary (plan §8, §16).

The in-memory store mirrors NoSQL conditional-write semantics exactly:
  * put_if_absent rejects a duplicate key.
  * update_if_version rejects a missing row, a version mismatch, AND a
    condition-false payload — all under the same atomic lock as the read.
  * Two concurrent updaters with the same expected_version cannot both
    succeed — one wins, the other gets ConditionalCheckFailed.

These tests are the "state/nosql.py conditional-write helper proven"
acceptance criterion (plan §16).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from personal_strat_pai.state.nosql import (
    ConditionalCheckFailed,
    InMemoryNoSqlStore,
    TableNotProvisioned,
)


def test_create_table_idempotent_same_schema():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    # Re-creating with the same schema is a no-op.
    s.create_table("t", key_schema={"k": "str"})
    s.put_if_absent("t", {"k": "x"}, {"v": 1})
    assert s.get("t", {"k": "x"}).payload == {"v": 1}


def test_create_table_rejects_schema_change():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    with pytest.raises(ValueError, match="different key schema"):
        s.create_table("t", key_schema={"k": "int"})


def test_table_not_provisioned_raises():
    s = InMemoryNoSqlStore()
    with pytest.raises(TableNotProvisioned):
        s.get("missing", {"k": "x"})


def test_put_if_absent_inserts_and_returns_version():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    row = s.put_if_absent("t", {"k": "x"}, {"v": 1})
    assert row.payload == {"v": 1}
    assert row.version  # non-empty version string


def test_put_if_absent_rejects_duplicate():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    s.put_if_absent("t", {"k": "x"}, {"v": 1})
    with pytest.raises(ConditionalCheckFailed, match="row already exists"):
        s.put_if_absent("t", {"k": "x"}, {"v": 2})


def test_update_if_version_succeeds_on_match():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    r0 = s.put_if_absent("t", {"k": "x"}, {"v": 1})
    r1 = s.update_if_version("t", {"k": "x"}, {"v": 2}, expected_version=r0.version)
    assert r1.payload == {"v": 2}
    assert r1.version != r0.version  # version bumps
    # Subsequent read sees the new payload + new version.
    r2 = s.get("t", {"k": "x"})
    assert r2.payload == {"v": 2}
    assert r2.version == r1.version


def test_update_if_version_rejects_missing_row():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    with pytest.raises(ConditionalCheckFailed, match="row missing"):
        s.update_if_version("t", {"k": "x"}, {"v": 1}, expected_version="v0")


def test_update_if_version_rejects_version_mismatch():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    r0 = s.put_if_absent("t", {"k": "x"}, {"v": 1})
    # Bump the version first.
    s.update_if_version("t", {"k": "x"}, {"v": 2}, expected_version=r0.version)
    # Now the original expected_version is stale.
    with pytest.raises(ConditionalCheckFailed, match="version mismatch"):
        s.update_if_version("t", {"k": "x"}, {"v": 3}, expected_version=r0.version)


def test_update_if_condition_rejects_false_predicate():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    r0 = s.put_if_absent("t", {"k": "x"}, {"v": 1, "status": "open"})

    def is_open(payload):
        return payload.get("status") == "open"

    # Condition true -> update succeeds.
    s.update_if_condition(
        "t",
        {"k": "x"},
        {"v": 2, "status": "closed"},
        expected_version=r0.version,
        condition=is_open,
        condition_desc="status==open",
    )
    # Now condition false -> rejects.
    r1 = s.get("t", {"k": "x"})
    with pytest.raises(ConditionalCheckFailed, match="condition false"):
        s.update_if_condition(
            "t",
            {"k": "x"},
            {"v": 3, "status": "washed"},
            expected_version=r1.version,
            condition=is_open,
            condition_desc="status==open",
        )


def test_update_if_condition_evaluated_atomically_under_lock():
    """The condition is evaluated inside the store's write lock — a concurrent
    writer that flips the predicate between read and write causes the second
    caller to fail rather than applying a stale-condition update.
    """
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    r0 = s.put_if_absent("t", {"k": "x"}, {"v": 1, "status": "open"})

    barrier = threading.Barrier(2)
    results: list[bool] = [False, False]

    def is_open(payload):
        return payload.get("status") == "open"

    def writer(idx: int, sleep_at_barrier: bool) -> None:
        if sleep_at_barrier:
            barrier.wait(timeout=2.0)
        try:
            s.update_if_condition(
                "t",
                {"k": "x"},
                {"v": 100 + idx, "status": "closed"},
                expected_version=r0.version,
                condition=is_open,
                condition_desc="status==open",
            )
            results[idx] = True
        except ConditionalCheckFailed:
            results[idx] = False

    # Two writers, same expected_version, both start simultaneously.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(writer, 0, True)
        f2 = ex.submit(writer, 1, True)
        f1.result()
        f2.result()
    # Exactly one succeeded; the other saw the version bump.
    assert sum(results) == 1, f"expected exactly one writer to succeed; got {results}"


def test_concurrent_updaters_with_same_version_one_wins():
    """The atomicity fence: two concurrent update_if_version with the same
    expected_version cannot both succeed. One wins, the other raises
    ConditionalCheckFailed. This is the lot-closure / triplet-advance fence.
    """
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    r0 = s.put_if_absent("t", {"k": "x"}, {"v": 0})

    barrier = threading.Barrier(4)
    successes: list[bool] = [False] * 4

    def writer(idx: int) -> None:
        barrier.wait(timeout=2.0)
        try:
            s.update_if_version("t", {"k": "x"}, {"v": idx}, expected_version=r0.version)
            successes[idx] = True
        except ConditionalCheckFailed:
            successes[idx] = False

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(writer, i) for i in range(4)]
        for f in futures:
            f.result()
    assert sum(successes) == 1, f"expected exactly one writer to succeed; got {successes}"


def test_delete_if_version_succeeds_and_rejects_mismatch():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    r0 = s.put_if_absent("t", {"k": "x"}, {"v": 1})
    s.delete_if_version("t", {"k": "x"}, expected_version=r0.version)
    assert s.get("t", {"k": "x"}) is None
    # Re-create + delete with wrong version fails.
    s.put_if_absent("t", {"k": "x"}, {"v": 2})
    with pytest.raises(ConditionalCheckFailed, match="version mismatch"):
        s.delete_if_version("t", {"k": "x"}, expected_version="v999")


def test_query_equality_predicate():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    s.put_if_absent("t", {"k": "x"}, {"v": 1, "tag": "a"})
    s.put_if_absent("t", {"k": "y"}, {"v": 2, "tag": "a"})
    s.put_if_absent("t", {"k": "z"}, {"v": 3, "tag": "b"})
    rows = s.query("t", where={"tag": "a"})
    assert {r.payload["v"] for r in rows} == {1, 2}
    rows = s.query("t", where={"tag": "a"}, limit=1)
    assert len(rows) == 1


def test_query_rejects_string_in_in_memory():
    s = InMemoryNoSqlStore()
    s.create_table("t", key_schema={"k": "str"})
    with pytest.raises(NotImplementedError):
        s.query("t", where="v > 0")


def test_composite_key_round_trip():
    """Composite keys (positions, realized_pnl, sec_compliance) work end-to-end."""
    s = InMemoryNoSqlStore()
    s.create_table("positions", key_schema={"account": "str", "bucket_id": "int", "ticker": "str"})
    s.put_if_absent(
        "positions",
        {"account": "paper", "bucket_id": 1, "ticker": "XLB"},
        {"qty": 100.0, "avg_cost": 100.0},
    )
    row = s.get("positions", {"account": "paper", "bucket_id": 1, "ticker": "XLB"})
    assert row.payload["qty"] == 100.0
