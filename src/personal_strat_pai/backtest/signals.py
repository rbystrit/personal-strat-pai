"""Strategy protocol + momentum signal — shared backtest/live code path (plan §10).

The ``Strategy`` protocol is the seam between the strategy and the execution
engine. Both backtest and live code against it — the backtester calls
``target_weights`` on historical bars; the live scheduler calls the same
method on the current bar snapshot. No backtest-only strategy logic (plan §10
acceptance criterion).

For P0-3, a simplified ``MomentumStrategy`` is provided — enough to run the
reconciliation scenario and prove the full loop. The full 4-layer sieve
(options pre-filters → absolute regime voting → relative-strength ROC →
tax-hurdle) is Phase 1 (roadmap Week 4). The strategy config (``StrategyConfig``)
already carries the D7 conservative defaults for the full sieve; the
momentum signal here uses the ROC weights and regime windows from it.

Brief §3 Layer 3: Blended ROC = 0.5·ROC₃M + 0.5·ROC₆M.
Brief §3 Layer 2: Absolute regime voting — ≥3 of 4 windows positive.
Brief §5: 50/30/20 waterfall allocation into the top 3.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl

from personal_strat_pai.backtest.portfolio import Portfolio
from personal_strat_pai.config import StrategyConfig

__all__ = [
    "MomentumStrategy",
    "Strategy",
    "compute_blended_roc",
    "compute_regime_votes",
    "compute_roc",
]


class Strategy(Protocol):
    """Strategy protocol — the seam between strategy and execution (plan §10).

    ``target_weights`` returns `{ticker: weight}` where weight is the fraction
    of NAV to allocate. The engine computes dollar allocations and generates
    orders to move from current to target. Weights sum to ≤ 1.0 (long-only,
    unlevered — the risk guard enforces this).
    """

    def target_weights(
        self,
        bars: pl.DataFrame,
        as_of: datetime,
        portfolio: Portfolio,
    ) -> dict[str, float]: ...


def compute_roc(closes: pl.Series, window: int) -> float | None:
    """Rate of change over ``window`` periods: (close[-1] / close[-window-1]) - 1.

    Returns ``None`` if not enough history. ``closes`` must be sorted ascending
    by date (caller responsibility).
    """
    if closes.len() < window + 1:
        return None
    end = closes[-1]
    start = closes[-(window + 1)]
    if start <= 0:
        return None
    return float(end / start - 1.0)


def compute_blended_roc(
    closes: pl.Series,
    *,
    roc_weight_3m: float = 0.5,
    roc_weight_6m: float = 0.5,
    months_3m: int = 63,  # ~3 months of trading days
    months_6m: int = 126,  # ~6 months of trading days
) -> float | None:
    """Blended ROC = w₃·ROC₃M + w₆·ROC₆M (brief §3 Layer 3).

    Returns ``None`` if not enough history for any window with non-zero weight.
    Windows with zero weight are skipped (don't require data).
    """
    roc_3m = compute_roc(closes, months_3m)
    roc_6m = compute_roc(closes, months_6m)

    # Only require windows with non-zero weight.
    if roc_weight_3m > 0 and roc_3m is None:
        return None
    if roc_weight_6m > 0 and roc_6m is None:
        return None

    total = 0.0
    if roc_weight_3m > 0 and roc_3m is not None:
        total += roc_weight_3m * roc_3m
    if roc_weight_6m > 0 and roc_6m is not None:
        total += roc_weight_6m * roc_6m
    return total


def compute_regime_votes(
    closes: pl.Series,
    windows: list[int],
    *,
    min_votes: int,
) -> int:
    """Absolute regime voting (brief §3 Layer 2).

    Each window's return > 0 = 1 vote. Returns the vote count. A ticker
    passes if votes >= ``min_votes``. Returns 0 if not enough history.
    """
    votes = 0
    for w in windows:
        roc = compute_roc(closes, w)
        if roc is not None and roc > 0:
            votes += 1
    return votes


class MomentumStrategy:
    """Simplified momentum strategy for P0-3 (plan §10, brief §3).

    Month-end rebalance:
      1. Compute blended ROC (0.5·ROC₃M + 0.5·ROC₆M) for each ticker.
      2. Absolute regime filter: ≥3 of 4 windows (3/6/9/12 months) positive.
         Crypto: ≥2 of 3 (1/2/3 months).
      3. Rank passing tickers by blended ROC descending.
      4. Top N get the waterfall allocation (default 50/30/20 for top 3).

    This is a simplified version of the full 4-layer sieve — no options
    pre-filters (smoke detector, FIRF) and no tax-hurdle (L4). Those are
    Phase 1. The strategy is deterministic (no randomness) so the
    reconciliation gate is reproducible.

    Config is injected (D7 conservative defaults from ``config/strategy.yaml``).
    The universe config (bucket definitions) is optional — when provided,
    the strategy picks one ticker per bucket (the highest-ROC slot's ETF).
    """

    def __init__(
        self,
        config: StrategyConfig,
        *,
        n_positions: int = 3,
        trading_days_per_month: int = 21,
        crypto_buckets: frozenset[int] = frozenset(),
        crypto_tickers: frozenset[str] = frozenset(),
    ) -> None:
        self.config = config
        self.n_positions = n_positions
        self.trading_days_per_month = trading_days_per_month
        self.crypto_buckets = crypto_buckets
        self.crypto_tickers = crypto_tickers

    def target_weights(
        self,
        bars: pl.DataFrame,
        as_of: datetime,
        portfolio: Portfolio,
    ) -> dict[str, float]:
        """Compute target weights at the month-end decision boundary.

        ``bars`` is the eager frame of all bars up to and including ``as_of``
        (produced by ``BarDriver.bars_through``). The method filters to
        ``as_of``, computes trailing returns, applies the regime filter,
        ranks by blended ROC, and returns the waterfall allocation.
        """
        as_of_date = as_of.date() if hasattr(as_of, "date") else as_of

        # Filter to bars on or before the decision date.
        sub = bars.filter(pl.col("ts").dt.date() <= as_of_date).sort(["symbol", "ts"])
        if sub.is_empty():
            return {}

        # Compute per-ticker blended ROC and regime votes.
        candidates: list[tuple[str, float, float]] = []  # (ticker, roc, beta_placeholder)
        symbols = sub["symbol"].unique().sort().to_list()

        for sym in symbols:
            sym_bars = sub.filter(pl.col("symbol") == sym)
            closes = sym_bars["close"].cast(pl.Float64)
            if closes.len() < 10:
                continue

            is_crypto = sym in self.crypto_tickers
            regime_windows = (
                self.config.regime_windows_crypto if is_crypto
                else self.config.regime_windows_sectors
            )
            min_votes = (
                self.config.regime_votes_crypto if is_crypto
                else self.config.regime_votes_sectors
            )

            # Convert month-based windows to trading days.
            regime_windows_days = [w * self.trading_days_per_month for w in regime_windows]
            votes = compute_regime_votes(closes, regime_windows_days, min_votes=min_votes)
            if votes < min_votes:
                continue

            roc = compute_blended_roc(
                closes,
                roc_weight_3m=self.config.roc_weight_3m,
                roc_weight_6m=self.config.roc_weight_6m,
                months_3m=3 * self.trading_days_per_month,
                months_6m=6 * self.trading_days_per_month,
            )
            if roc is None:
                continue

            candidates.append((sym, roc, 1.0))  # beta placeholder

        if not candidates:
            return {}

        # Rank by blended ROC descending.
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_n = candidates[: self.n_positions]

        # Waterfall allocation (brief §5: 50/30/20).
        waterfall = self.config.waterfall[: len(top_n)]
        total_w = sum(waterfall)
        weights = {}
        for (sym, _roc, _beta), w in zip(top_n, waterfall, strict=False):
            weights[sym] = w / total_w if total_w > 0 else 0.0

        return weights
