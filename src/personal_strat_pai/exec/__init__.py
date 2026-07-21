"""Execution layer — order routing, IBKR session, lease (plan §4, §7, §11).

v1 scope (P0-2): ``exec/lease.py`` only — the no-op stub. The IBKR session
(``ibkr_session.py``), order router (``router.py``), and target portfolio
diff (``target_portfolio.py``) are built in P0-3 / P0-4 (roadmap-30d Week 3).
"""

from __future__ import annotations

from personal_strat_pai.exec.lease import ExecutionLease, LeaseError, NoOpLease

__all__ = [
    "ExecutionLease",
    "LeaseError",
    "NoOpLease",
]
