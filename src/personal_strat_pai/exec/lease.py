"""Execution lease — v1 NO-OP STUB (plan §3.3, §8, §11; D12).

The execution lease is the fencing boundary that prevents split-brain between
the podman primary and the OCI Functions backup once the backup exists: only
the lease holder may place orders, and acquisition by the backup only succeeds
if the primary's lease has expired (``expires_at < now``). The full
acquire/renew/check/release cycle with NoSQL conditional writes on
``(holder, expires_at, generation)`` is **the first v2 deliverable (P0-5)** —
it is NOT wired in v1 because v1 has a single execution path (the podman
primary) and split-brain is impossible by construction (D12).

This module is a documented **no-op stub** so the exec layer's pre-trade check
surface (``exec/router.py`` in P0-4) can call ``lease.held_by_me()`` from day
one and get the v1-correct answer (always True on the primary). When the v2
backup lands, the stub is replaced by ``OciExecutionLease`` (the real
conditional-write impl) without touching the router — the
``ExecutionLease`` Protocol is the seam.

v1 invariants (enforced by the stub):
  * ``held_by_me()`` always returns True (the primary is the only writer).
  * ``acquire()`` / ``renew()`` / ``release()`` are no-ops that log a debug
    note and return success. They do NOT touch NoSQL; the ``execution_lease``
    table is NOT created in v1 (plan §8, D12).
  * The stub is NOT a fence — it provides no split-brain protection. The
    contract is "single writer, no fencing needed"; v2 replaces this with
    real fencing (P0-5).

v2 wiring (P0-5): ``OciExecutionLease`` will implement the same Protocol with
NoSQL ``update_if_condition`` on ``(holder, expires_at, generation)`` —
renewal is a conditional update that only succeeds if the caller is still the
holder; acquisition by the backup only succeeds if ``expires_at < now``. The
fencing property tests (plan §16: "two concurrent acquirers never both hold;
backup never trades while primary TTL is live") are a P0-5 must-pass gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from personal_strat_pai.state.nosql import NoSqlStore

__all__ = [
    "ExecutionLease",
    "LeaseError",
    "NoOpLease",
]

_log = logging.getLogger(__name__)


class LeaseError(Exception):
    """Raised by the real (v2) lease implementation on acquire/renew failure.

    The v1 ``NoOpLease`` never raises ``LeaseError`` — every operation
    succeeds (single writer). Kept in the module signature so the exec layer
    can catch it from day one; it becomes live in P0-5.
    """


@dataclass(frozen=True, slots=True)
class LeaseInfo:
    """A snapshot of the lease state (holder, expires_at, generation).

    v1 ``NoOpLease`` returns a placeholder with ``holder="primary"``,
    ``expires_at`` far in the future, and ``generation=0``. The real v2
    implementation reads this from the ``execution_lease`` NoSQL row.
    """

    holder: str
    expires_at: datetime
    generation: int


@runtime_checkable
class ExecutionLease(Protocol):
    """The lease contract the exec layer's pre-trade check calls (plan §3.3, §11).

    v1: ``NoOpLease`` — every method succeeds; ``held_by_me`` is always True.
    v2: ``OciExecutionLease`` — NoSQL-conditional-write fencing (P0-5).
    """

    def held_by_me(self, *, now: datetime | None = None) -> bool:
        """True iff this path holds a live lease at ``now`` (v1: always True)."""
        ...

    def acquire(self, *, now: datetime | None = None) -> LeaseInfo:
        """Acquire the lease (v1: no-op; v2: conditional write if expired)."""
        ...

    def renew(self, *, now: datetime | None = None) -> LeaseInfo:
        """Renew the lease (v1: no-op; v2: conditional update as the holder)."""
        ...

    def release(self, *, now: datetime | None = None) -> None:
        """Release the lease (v1: no-op; v2: conditional delete as the holder)."""
        ...


class NoOpLease:
    """v1 no-op lease stub (plan §3.3, §8, D12).

    Single writer — no fencing needed. Every method succeeds. The
    ``execution_lease`` NoSQL table is NOT created in v1; this stub never
    touches NoSQL. Replaced by ``OciExecutionLease`` in P0-5 (v2 backup).

    The ``store`` argument is accepted (and ignored) so the composition root
    in P0-4 can pass the same ``NoSqlStore`` it passes to the ledger and
    triplet repos — the v2 swap then uses it without changing the call site.
    """

    HOLDER = "primary"

    def __init__(self, store: NoSqlStore | None = None, *, account: str = "paper") -> None:
        # `store` is accepted for API parity with the v2 implementation and
        # deliberately unused (v1: no lease table, no NoSQL writes).
        self._store = store
        self._account = account
        _log.debug(
            "NoOpLease initialized (v1 stub — no fencing; single writer; "
            "execution_lease table NOT created per D12)"
        )

    def held_by_me(self, *, now: datetime | None = None) -> bool:
        """v1: always True — the primary is the only writer."""
        return True

    def acquire(self, *, now: datetime | None = None) -> LeaseInfo:
        """v1: no-op. Returns a placeholder lease that never expires.

        v2 will: read the ``execution_lease`` row; if absent or
        ``expires_at < now``, conditionally write ``(holder=primary, ...)``;
        raise ``LeaseError`` if the backup holds a live lease.
        """
        _log.debug("NoOpLease.acquire (v1 no-op — single writer)")
        now_dt = now or datetime.now(tz=UTC)
        return LeaseInfo(holder=self.HOLDER, expires_at=now_dt + timedelta(days=365), generation=0)

    def renew(self, *, now: datetime | None = None) -> LeaseInfo:
        """v1: no-op. Returns the same placeholder lease."""
        _log.debug("NoOpLease.renew (v1 no-op — single writer)")
        return self.acquire(now=now)

    def release(self, *, now: datetime | None = None) -> None:
        """v1: no-op. Logs the release for audit; does NOT touch NoSQL."""
        _log.debug("NoOpLease.release (v1 no-op — single writer)")
