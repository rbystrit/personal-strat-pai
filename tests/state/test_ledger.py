"""Tests for state/ledger.py — HIFO selection + ST/LT split (plan §8, §16).

Hypothesis property tests (the plan §16 acceptance criteria):
  * **qty conservation** — sum of closed_qty across selected lots == min(sell_qty, total_open_qty).
  * **proceeds = Σ per-lot proceeds** — total_proceeds == fill_price * filled_qty.
  * **no lot double-counted** — each lot appears at most once in a selection;
    close_qty <= open_qty for every selected lot.

Plus targeted tests for:
  * HIFO sort order (highest cost-basis first; tie-break newer-acquired first).
  * ST/LT split at the lot level (holding_days < 365 = ST, >= 365 = LT).
  * LedgerRepo.sell atomicity — a concurrent seller cannot double-close a lot.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from personal_strat_pai.state.ledger import (
    HoldingClass,
    LedgerError,
    LedgerRepo,
    LotStatus,
    SellRequest,
    TaxLot,
    close_lot,
    hifo_select,
    holding_class,
    holding_days,
    realize_lot,
    split_proceeds,
    st_lt_split,
)
from personal_strat_pai.state.nosql import ConditionalCheckFailed, InMemoryNoSqlStore

# --- Pure HIFO selection tests --- #


def _lot(
    lot_id: str = "L1",
    *,
    cost: float = 100.0,
    qty: float = 50.0,
    closed_qty: float = 0.0,
    acquired_at: datetime | None = None,
    bucket_id: int = 1,
    ticker: str = "XLB",
    slot: str = "A",
    status: LotStatus = LotStatus.OPEN,
) -> TaxLot:
    return TaxLot(
        lot_id=lot_id,
        account="paper",
        bucket_id=bucket_id,
        ticker=ticker,
        triplet_slot=slot,
        qty=qty,
        closed_qty=closed_qty,
        cost_basis_per_share=cost,
        acquired_at=acquired_at or datetime(2026, 1, 1, tzinfo=UTC),
        status=status,
    )


def test_hifo_select_highest_cost_first():
    """HIFO: the highest cost-basis lot is selected first."""
    lots = [
        _lot("L1", cost=100.0, qty=50.0),
        _lot("L2", cost=110.0, qty=50.0),
        _lot("L3", cost=90.0, qty=50.0),
    ]
    selected = hifo_select(lots, sell_qty=60.0)
    # Highest cost first: L2 (110) -> L1 (100) -> ...
    assert selected[0][0].lot_id == "L2"
    assert selected[0][1] == 50.0  # full lot taken
    assert selected[1][0].lot_id == "L1"
    assert selected[1][1] == 10.0  # remaining 10 from the 60 sell


def test_hifo_select_tie_break_newer_first():
    """Tie-break: same cost-basis, newer-acquired first (more likely ST)."""
    lots = [
        _lot("L1", cost=100.0, acquired_at=datetime(2026, 1, 1, tzinfo=UTC)),
        _lot("L2", cost=100.0, acquired_at=datetime(2026, 6, 1, tzinfo=UTC)),  # newer
    ]
    selected = hifo_select(lots, sell_qty=10.0)
    assert selected[0][0].lot_id == "L2"  # newer first


def test_hifo_select_skips_closed_and_zero_open():
    """Only OPEN lots with open_qty > 0 are eligible."""
    lots = [
        _lot("L1", cost=100.0, qty=50.0, closed_qty=50.0),  # fully closed
        _lot("L2", cost=110.0, qty=50.0, status=LotStatus.CLOSED),
        _lot("L3", cost=90.0, qty=50.0, closed_qty=10.0),  # partial
    ]
    selected = hifo_select(lots, sell_qty=100.0)
    # Only L3 is eligible; only 40 open.
    assert len(selected) == 1
    assert selected[0][0].lot_id == "L3"
    assert selected[0][1] == 40.0


def test_hifo_select_zero_or_negative_qty_returns_empty():
    lots = [_lot("L1", cost=100.0, qty=50.0)]
    assert hifo_select(lots, sell_qty=0.0) == []
    assert hifo_select(lots, sell_qty=-5.0) == []


def test_hifo_select_partial_fill_when_open_insufficient():
    """If open qty < sell qty, the selection returns all available and the
    caller surfaces the remainder as SellResult.unfilled_qty."""
    lots = [_lot("L1", cost=100.0, qty=30.0)]
    selected = hifo_select(lots, sell_qty=50.0)
    assert len(selected) == 1
    assert selected[0][1] == 30.0  # only 30 available


# --- Realization + ST/LT split tests --- #


def test_holding_class_st_under_365_days():
    lot = _lot(acquired_at=datetime(2026, 1, 1, tzinfo=UTC))
    as_of = datetime(2026, 7, 1, tzinfo=UTC)  # ~181 days
    assert holding_class(lot, as_of) == HoldingClass.SHORT_TERM
    assert holding_days(lot, as_of) == 181


def test_holding_class_lt_at_365_days():
    lot = _lot(acquired_at=datetime(2025, 7, 1, tzinfo=UTC))
    as_of = datetime(2026, 7, 1, tzinfo=UTC)  # exactly 365 days
    assert holding_class(lot, as_of) == HoldingClass.LONG_TERM
    assert holding_days(lot, as_of) == 365


def test_holding_class_zero_days_is_st():
    """Same-day buy+sell = 0 holding days = ST (brief §1)."""
    lot = _lot(acquired_at=datetime(2026, 7, 1, tzinfo=UTC))
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    assert holding_class(lot, as_of) == HoldingClass.SHORT_TERM
    assert holding_days(lot, as_of) == 0


def test_realize_lot_computes_gain_proceeds_cost():
    lot = _lot(cost=100.0, qty=50.0, acquired_at=datetime(2025, 1, 1, tzinfo=UTC))
    as_of = datetime(2026, 7, 1, tzinfo=UTC)  # LT
    realized = realize_lot(lot, close_qty=20.0, fill_price=110.0, as_of=as_of)
    assert realized.closed_qty == 20.0
    assert realized.proceeds == pytest.approx(20.0 * 110.0)
    assert realized.cost == pytest.approx(20.0 * 100.0)
    assert realized.gain == pytest.approx(20.0 * 10.0)  # +200 gain
    assert realized.holding_class == HoldingClass.LONG_TERM
    assert not realized.was_loss


def test_realize_lot_harvests_loss_on_down_move():
    """HIFO on a down-move: high cost-basis => larger harvested loss."""
    lot = _lot(cost=120.0, qty=50.0, acquired_at=datetime(2025, 1, 1, tzinfo=UTC))
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    realized = realize_lot(lot, close_qty=50.0, fill_price=100.0, as_of=as_of)
    assert realized.gain == pytest.approx(50.0 * (100.0 - 120.0))  # -1000 loss
    assert realized.was_loss
    assert realized.holding_class == HoldingClass.LONG_TERM


def test_realize_lot_rejects_overclose():
    lot = _lot(cost=100.0, qty=10.0)
    with pytest.raises(LedgerError, match="exceeds open qty"):
        realize_lot(lot, close_qty=20.0, fill_price=100.0, as_of=datetime(2026, 7, 1, tzinfo=UTC))


def test_close_lot_partial_keeps_open_status():
    lot = _lot(cost=100.0, qty=50.0, closed_qty=10.0)
    realized, new_state = close_lot(
        lot, close_qty=20.0, fill_price=110.0, as_of=datetime(2026, 7, 1, tzinfo=UTC)
    )
    assert new_state.new_closed_qty == 30.0
    assert new_state.new_status == LotStatus.OPEN  # still 20 open
    assert realized.closed_qty == 20.0


def test_close_lot_full_flips_to_closed():
    lot = _lot(cost=100.0, qty=50.0, closed_qty=0.0)
    _realized, new_state = close_lot(
        lot, close_qty=50.0, fill_price=110.0, as_of=datetime(2026, 7, 1, tzinfo=UTC)
    )
    assert new_state.new_closed_qty == 50.0
    assert new_state.new_status == LotStatus.CLOSED


def test_st_lt_split_aggregates_by_class():
    lots = [
        _lot("L1", cost=100.0, qty=10.0, acquired_at=datetime(2025, 1, 1, tzinfo=UTC)),  # LT
        _lot("L2", cost=100.0, qty=10.0, acquired_at=datetime(2026, 6, 1, tzinfo=UTC)),  # ST
    ]
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    realized = [
        realize_lot(lots[0], 10.0, 110.0, as_of),  # +100 LT gain
        realize_lot(lots[1], 10.0, 90.0, as_of),  # -100 ST loss
    ]
    st, lt = st_lt_split(realized)
    assert st == pytest.approx(-100.0)
    assert lt == pytest.approx(100.0)


def test_split_proceeds_sums_per_lot():
    lots = [
        _lot("L1", cost=100.0, qty=30.0, acquired_at=datetime(2025, 1, 1, tzinfo=UTC)),
        _lot("L2", cost=110.0, qty=20.0, acquired_at=datetime(2026, 1, 1, tzinfo=UTC)),
    ]
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    realized = [
        realize_lot(lots[0], 30.0, 110.0, as_of),
        realize_lot(lots[1], 20.0, 110.0, as_of),
    ]
    proceeds, cost, gain = split_proceeds(realized)
    assert proceeds == pytest.approx((30.0 + 20.0) * 110.0)
    assert cost == pytest.approx(30.0 * 100.0 + 20.0 * 110.0)
    assert gain == pytest.approx(proceeds - cost)


# --- Hypothesis property tests (plan §16 acceptance) --- #


@st.composite
def _lot_strategy(draw):
    """A strategy for a single TaxLot with a realistic-but-random state."""
    lot_id = draw(
        st.text(min_size=1, max_size=8, alphabet=st.characters(blacklist_categories=("Cs",)))
    )
    cost = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    qty = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    closed_qty = draw(
        st.floats(min_value=0.0, max_value=qty, allow_nan=False, allow_infinity=False)
    )
    # Acquired between 2020-01-01 and 2026-06-01.
    acquired_at = datetime.fromtimestamp(
        draw(
            st.integers(
                min_value=int(datetime(2020, 1, 1, tzinfo=UTC).timestamp()),
                max_value=int(datetime(2026, 6, 1, tzinfo=UTC).timestamp()),
            )
        ),
        tz=UTC,
    )
    return _lot(lot_id, cost=cost, qty=qty, closed_qty=closed_qty, acquired_at=acquired_at)


@st.composite
def _lots_strategy(draw, max_lots: int = 6):
    """A strategy for a list of TaxLots with distinct lot_ids."""
    n = draw(st.integers(min_value=0, max_value=max_lots))
    ids = [f"L{i}" for i in range(n)]
    lots: list[TaxLot] = []
    for lot_id in ids:
        lots.append(
            draw(
                _lot_strategy().map(
                    lambda lot, lid=lot_id: TaxLot(
                        lot_id=lid,
                        account=lot.account,
                        bucket_id=lot.bucket_id,
                        ticker=lot.ticker,
                        triplet_slot=lot.triplet_slot,
                        qty=lot.qty,
                        closed_qty=lot.closed_qty,
                        cost_basis_per_share=lot.cost_basis_per_share,
                        acquired_at=lot.acquired_at,
                        status=lot.status,
                    )
                )
            )
        )
    return lots


@given(
    lots=_lots_strategy(max_lots=6),
    sell_qty=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_hifo_property_qty_conservation(lots: list[TaxLot], sell_qty: float):
    """Σ selected close_qty == min(sell_qty, total open qty). No lot double-counted."""
    selected = hifo_select(lots, sell_qty=sell_qty)
    total_open = sum(lot.open_qty for lot in lots if lot.is_open)
    expected = min(sell_qty, total_open)
    actual = sum(q for _, q in selected)
    # Float tolerance for sum.
    assert abs(actual - expected) < 1e-6, f"qty not conserved: {actual} != {expected}"
    # No lot selected twice.
    lot_ids = [lot.lot_id for lot, _ in selected]
    assert len(lot_ids) == len(set(lot_ids)), f"lot double-counted: {lot_ids}"
    # close_qty <= open_qty for every selected lot.
    for lot, q in selected:
        assert q <= lot.open_qty + 1e-9
        assert q > 0


@given(
    lots=_lots_strategy(max_lots=6),
    sell_qty=st.floats(min_value=0.01, max_value=500.0, allow_nan=False, allow_infinity=False),
    fill_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_hifo_property_proceeds_equal_sum_of_per_lot_proceeds(lots, sell_qty, fill_price):
    """total_proceeds == fill_price * filled_qty (Σ per-lot proceeds)."""
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    selected = hifo_select(lots, sell_qty=sell_qty)
    realized = [realize_lot(lot, q, fill_price, as_of) for lot, q in selected]
    proceeds, cost, gain = split_proceeds(realized)
    filled_qty = sum(r.closed_qty for r in realized)
    assert abs(proceeds - fill_price * filled_qty) < 1e-6
    assert abs(gain - (proceeds - cost)) < 1e-6


@given(lots=_lots_strategy(max_lots=8))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_hifo_property_no_lot_double_counted(lots):
    """A lot can never appear twice in a single HIFO selection."""
    for sell_qty in (1.0, 10.0, 100.0, 1000.0):
        selected = hifo_select(lots, sell_qty=sell_qty)
        ids = [lot.lot_id for lot, _ in selected]
        assert len(ids) == len(set(ids)), f"double-counted at sell_qty={sell_qty}: {ids}"


@given(lots=_lots_strategy(max_lots=8))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_hifo_property_highest_cost_first(lots):
    """The first selected lot is the highest-cost open lot (HIFO)."""
    open_lots = [lot for lot in lots if lot.is_open]
    if not open_lots:
        return
    expected_first = max(
        open_lots, key=lambda lot: (lot.cost_basis_per_share, lot.acquired_at.timestamp())
    )
    selected = hifo_select(lots, sell_qty=1e6)  # ask for everything
    if selected:
        assert (
            selected[0][0].lot_id == expected_first.lot_id
            or selected[0][0].cost_basis_per_share == expected_first.cost_basis_per_share
        )


# --- LedgerRepo sell tests (atomicity + integration) --- #


@pytest.fixture
def ledger_with_lots():
    """A LedgerRepo with 3 open lots for (paper, bucket=1, XLB): costs 110, 100, 90."""
    s = InMemoryNoSqlStore()
    s.create_table("tax_lots", key_schema={"lot_id": "str"})
    s.create_table("realized_pnl", key_schema={"lot_id": "str", "closed_at": "str"})
    repo = LedgerRepo(s)
    base = datetime(2025, 1, 1, tzinfo=UTC)
    for lid, cost in (("L1", 110.0), ("L2", 100.0), ("L3", 90.0)):
        repo.open_lot(
            TaxLot(
                lot_id=lid,
                account="paper",
                bucket_id=1,
                ticker="XLB",
                triplet_slot="A",
                qty=50.0,
                closed_qty=0.0,
                cost_basis_per_share=cost,
                acquired_at=base,
            )
        )
    return repo, s


def test_ledger_repo_open_lots_returns_hifo_order(ledger_with_lots):
    repo, _ = ledger_with_lots
    lots = repo.open_lots(account="paper", bucket_id=1, ticker="XLB")
    assert [lot.lot_id for lot in lots] == ["L1", "L2", "L3"]  # highest cost first
    assert all(
        lot.cost_basis_per_share == cost
        for lot, cost in zip(lots, (110.0, 100.0, 90.0), strict=True)
    )


def test_ledger_repo_sell_closes_hifo_first(ledger_with_lots):
    repo, _ = ledger_with_lots
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    result = repo.sell(
        SellRequest(
            account="paper",
            bucket_id=1,
            ticker="XLB",
            qty=60.0,
            fill_price=95.0,
            as_of=as_of,
        )
    )
    # HIFO: L1 (110) closed first (full 50), then L2 (100) for the remaining 10.
    assert len(result.realized) == 2
    assert result.realized[0].lot_id == "L1"
    assert result.realized[0].closed_qty == 50.0
    assert result.realized[1].lot_id == "L2"
    assert result.realized[1].closed_qty == 10.0
    # Proceeds = 60 * 95.
    assert result.total_proceeds == pytest.approx(60.0 * 95.0)
    # L1 was a loss (cost 110, sold 95) -> harvested.
    assert result.realized[0].was_loss
    assert result.harvested_loss > 0


def test_ledger_repo_sell_records_realized_pnl(ledger_with_lots):
    repo, s = ledger_with_lots
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    repo.sell(
        SellRequest(
            account="paper",
            bucket_id=1,
            ticker="XLB",
            qty=50.0,
            fill_price=95.0,
            as_of=as_of,
        )
    )
    rows = s.query("realized_pnl", where={"lot_id": "L1"})
    assert len(rows) == 1
    assert rows[0].payload["gain"] == pytest.approx(50.0 * (95.0 - 110.0))
    assert rows[0].payload["st_lt"] == "LT"  # 2025-01-01 to 2026-07-01 = 546 days


def test_ledger_repo_sell_unfilled_when_open_insufficient(ledger_with_lots):
    repo, _ = ledger_with_lots
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    # Total open = 150; ask for 200.
    result = repo.sell(
        SellRequest(
            account="paper",
            bucket_id=1,
            ticker="XLB",
            qty=200.0,
            fill_price=95.0,
            as_of=as_of,
        )
    )
    assert result.filled_qty == pytest.approx(150.0)
    assert result.unfilled_qty == pytest.approx(50.0)


def test_ledger_repo_sell_zero_qty_returns_empty(ledger_with_lots):
    repo, _ = ledger_with_lots
    result = repo.sell(
        SellRequest(
            account="paper",
            bucket_id=1,
            ticker="XLB",
            qty=0.0,
            fill_price=95.0,
            as_of=datetime(2026, 7, 1, tzinfo=UTC),
        )
    )
    assert result.realized == []
    assert result.unfilled_qty == 0.0


def test_ledger_repo_concurrent_sellers_cannot_double_close(ledger_with_lots):
    """The atomicity fence: two concurrent sells against the SAME lot cannot
    both close it. The HIFO selection reads the same snapshot, but the
    conditional-write fence lets only one close succeed; the other retries
    with a fresh snapshot and sees the lot is no longer open.
    """
    repo, _ = ledger_with_lots
    as_of = datetime(2026, 7, 1, tzinfo=UTC)

    # Two sellers each try to close 50 qty of L1 (the highest-cost lot, qty=50).
    # Without the fence, both would close L1 -> 100 closed_qty on a 50-qty lot.
    # With the fence, only one succeeds; the other sees L1 is closed and moves
    # to L2 (the next-highest cost).
    barrier = threading.Barrier(2)
    results: list = [None, None]

    def seller(idx: int) -> None:
        barrier.wait(timeout=2.0)
        results[idx] = repo.sell(
            SellRequest(
                account="paper",
                bucket_id=1,
                ticker="XLB",
                qty=50.0,
                fill_price=95.0,
                as_of=as_of,
            )
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(seller, 0)
        f2 = ex.submit(seller, 1)
        f1.result()
        f2.result()

    r0, r1 = results
    # Total filled qty across both sellers must be <= total open (150) and
    # each lot's closed_qty must be <= its qty (no double-close).
    total_filled = r0.filled_qty + r1.filled_qty
    assert total_filled <= 150.0 + 1e-6
    # No lot should be closed twice. Aggregate per-lot closed_qty across both sellers.
    closed_by_lot: dict[str, float] = {}
    for r in (r0, r1):
        for realized in r.realized:
            closed_by_lot[realized.lot_id] = (
                closed_by_lot.get(realized.lot_id, 0.0) + realized.closed_qty
            )
    # L1 has qty=50; total closed across both sellers must be <= 50.
    assert closed_by_lot.get("L1", 0.0) <= 50.0 + 1e-6
    assert closed_by_lot.get("L2", 0.0) <= 50.0 + 1e-6
    assert closed_by_lot.get("L3", 0.0) <= 50.0 + 1e-6


def test_ledger_repo_sell_partial_close_keeps_lot_open(ledger_with_lots):
    repo, _s = ledger_with_lots
    as_of = datetime(2026, 7, 1, tzinfo=UTC)
    # Close 20 of L1 (qty 50).
    repo.sell(
        SellRequest(
            account="paper",
            bucket_id=1,
            ticker="XLB",
            qty=20.0,
            fill_price=95.0,
            as_of=as_of,
        )
    )
    lot = repo.get_lot("L1")
    assert lot.status == LotStatus.OPEN
    assert lot.closed_qty == 20.0
    assert lot.open_qty == 30.0


def test_ledger_repo_open_lot_rejects_duplicate(ledger_with_lots):
    repo, _ = ledger_with_lots
    with pytest.raises(ConditionalCheckFailed):
        repo.open_lot(
            TaxLot(
                lot_id="L1",
                account="paper",
                bucket_id=1,
                ticker="XLB",
                triplet_slot="A",
                qty=50.0,
                closed_qty=0.0,
                cost_basis_per_share=110.0,
                acquired_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
        )
