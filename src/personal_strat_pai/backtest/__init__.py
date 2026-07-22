"""Backtester engine — one code path with live (plan §10, RBY-6).

The backtester reuses the strategy, risk, and cost modules that live trading
uses. Only the data source (parquet vs IBKR) and the execution venue
(simulated fills vs IBKR orders) differ. No backtest-only strategy logic.

Quick start:
    from personal_strat_pai.backtest import (
        Backtester, BacktestConfig, ParquetBarDriver,
        MomentumStrategy, RiskGuard, RiskLimits,
    )
    from personal_strat_pai.config import load_strategy

    driver = ParquetBarDriver("data/local/bars/daily")
    strategy = MomentumStrategy(load_strategy())
    guard = RiskGuard(RiskLimits())
    result = Backtester(BacktestConfig(initial_capital=100_000)).run(driver, strategy, guard)
    print(result.summary())

Validation gate: the backtest reproduces a hand-computed 3-bucket, 2-year
scenario to the penny against an independently computed P&L
(``tests/backtest/test_reconciliation.py``).
"""

from personal_strat_pai.backtest.costs import (
    CommissionModel,
    CostModel,
    FeeModel,
    Fill,
    IbkrTieredCostModel,
    Side,
    SlippageKind,
    SlippageModel,
)
from personal_strat_pai.backtest.data import (
    BarDriver,
    ParquetBarDriver,
    TradingCalendar,
)
from personal_strat_pai.backtest.engine import (
    BacktestConfig,
    Backtester,
    BacktestResult,
)
from personal_strat_pai.backtest.manifest import build_manifest
from personal_strat_pai.backtest.metrics import (
    compute_cost_attribution,
    compute_stats,
    compute_tax_drag,
)
from personal_strat_pai.backtest.portfolio import (
    Lot,
    Portfolio,
    Position,
    TradeRecord,
)
from personal_strat_pai.backtest.risk import (
    BetaProvider,
    DrawdownStopHit,
    KillSwitchAction,
    KillSwitchTripped,
    RiskEvent,
    RiskGuard,
    RiskLimits,
    RiskResult,
)
from personal_strat_pai.backtest.signals import (
    MomentumStrategy,
    Strategy,
    compute_blended_roc,
    compute_regime_votes,
    compute_roc,
)

__all__ = [
    # Engine
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    # Data
    "BarDriver",
    "ParquetBarDriver",
    "TradingCalendar",
    # Costs
    "CommissionModel",
    "CostModel",
    "FeeModel",
    "Fill",
    "IbkrTieredCostModel",
    "Side",
    "SlippageKind",
    "SlippageModel",
    # Portfolio
    "Lot",
    "Portfolio",
    "Position",
    "TradeRecord",
    # Risk
    "BetaProvider",
    "DrawdownStopHit",
    "KillSwitchAction",
    "KillSwitchTripped",
    "RiskEvent",
    "RiskGuard",
    "RiskLimits",
    "RiskResult",
    # Signals
    "MomentumStrategy",
    "Strategy",
    "compute_blended_roc",
    "compute_regime_votes",
    "compute_roc",
    # Metrics
    "compute_cost_attribution",
    "compute_stats",
    "compute_tax_drag",
    # Manifest
    "build_manifest",
]
