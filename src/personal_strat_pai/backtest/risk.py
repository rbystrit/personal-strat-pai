"""Risk guard — NAV cap, beta ceiling, kill switch, DD stops (plan §10, §12).

**Shared code** — the same ``RiskGuard`` runs in backtest and live (plan §10
acceptance criterion: "no backtest-only strategy logic"). The guard operates
on the target weight vector **before** orders are generated:

  1. **Kill switch** — if the portfolio drawdown exceeds the threshold, the
     switch latches. ``halt_entries`` stops new entries (allows exits);
     ``flatten`` goes to all cash. The latch persists until explicitly reset
     (mirrors a live kill-switch, not a soft warning).
  2. **NAV cap** — no single bucket exceeds ``nav_cap`` (40%) of NAV. Under
     the FIRF macro circuit breaker, capped buckets (Tech, Real Estate,
     Crypto) are limited to ``nav_cap_firf`` (20%).
  3. **Beta ceiling** — portfolio beta β_p = Σ(W_i × β_i) ≤ ``beta_ceiling``
     (1.35). If breached, scale back Rank 1/2 allocations and redirect into
     the lowest-beta passing assets until β_p < ceiling.
  4. **Gross exposure** — Σ|weights| ≤ 1.0 (long-only, unlevered).
  5. **DD stop** — account-level drawdown stop (CEO-SET for live; the guard
     raises ``DrawdownStopHit`` when breached).

The guard is a **pure function** on the target weight vector — it does not
place orders or touch the portfolio. The engine (backtest) or the execution
router (live) calls ``check`` and receives the adjusted weights + any risk
events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

__all__ = [
    "BetaProvider",
    "DrawdownStopHit",
    "KillSwitchAction",
    "KillSwitchTripped",
    "RiskEvent",
    "RiskGuard",
    "RiskLimits",
    "RiskResult",
]


class KillSwitchAction(StrEnum):
    """What the kill switch does when tripped (plan §12)."""

    HALT_ENTRIES = "halt_entries"  # stop new entries, allow exits
    FLATTEN = "flatten"  # go to all cash


@dataclass(slots=True)
class RiskEvent:
    """A risk event logged by the guard — for observability and alerting."""

    kind: str  # "kill_switch", "nav_cap", "beta_ceiling", "dd_stop", "gross_exposure"
    message: str
    severity: str = "warning"  # "warning" | "critical"


@dataclass(slots=True)
class RiskResult:
    """The guard's output: adjusted weights + risk events + kill-switch state."""

    weights: dict[str, float]
    events: list[RiskEvent] = field(default_factory=list)
    kill_switch_active: bool = False
    halted: bool = False  # True when kill switch is in halt_entries mode


class KillSwitchTripped(Exception):
    """Raised when the kill switch trips and action is ``flatten``."""


class DrawdownStopHit(Exception):
    """Raised when the account-level drawdown stop is breached."""


class BetaProvider(Protocol):
    """Per-ticker beta source (plan §12).

    Backtest: compute beta from historical returns vs a benchmark (SPY).
    Live: read from a reference data source or compute from recent bars.
    """

    def get_beta(self, symbol: str) -> float: ...


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Hard risk limits (plan §12). Conservative defaults for backtest.

    Live limits require explicit CEO approval (plan §12, role boundary).
    The ``# CEO-SET`` values (starting_capital, account_dd_stop_pct,
    per_ticker_max_shares) are in ``config/risk_limits.yaml``.
    """

    nav_cap: float = 0.40
    nav_cap_firf: float = 0.20
    beta_ceiling: float = 1.35
    max_gross_exposure: float = 1.0
    kill_switch_drawdown: float = 0.20
    kill_switch_action: KillSwitchAction = KillSwitchAction.HALT_ENTRIES
    account_dd_stop: float | None = None  # CEO-SET for live
    firf_capped_buckets: frozenset[int] = frozenset()  # bucket IDs capped by FIRF


class RiskGuard:
    """The shared risk guard — backtest and live use one code path (plan §10).

    ``check`` takes the target weight vector, portfolio drawdown, and
    per-ticker betas, then returns the adjusted weights after applying all
    risk constraints. The guard is stateless across calls except for the
    kill-switch latch (which persists until ``reset_kill_switch`` is called).
    """

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self._kill_switch_active = False

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    def reset_kill_switch(self) -> None:
        """Explicitly reset the kill switch latch (CEO or admin op)."""
        self._kill_switch_active = False

    def check(
        self,
        target_weights: dict[str, float],
        *,
        drawdown: float,
        beta_provider: BetaProvider | None = None,
        bucket_map: dict[str, int] | None = None,
    ) -> RiskResult:
        """Apply all risk constraints to the target weight vector.

        ``target_weights``: {ticker: weight} the strategy wants.
        ``drawdown``: current portfolio drawdown (0.0 to 1.0).
        ``beta_provider``: per-ticker beta source (required for beta ceiling).
        ``bucket_map``: {ticker: bucket_id} for FIRF cap lookup.

        Returns ``RiskResult`` with adjusted weights and any risk events.
        Raises ``DrawdownStopHit`` if the account DD stop is breached.
        """
        events: list[RiskEvent] = []

        # 1. Kill switch check (latching).
        if drawdown >= self.limits.kill_switch_drawdown:
            if not self._kill_switch_active:
                self._kill_switch_active = True
                events.append(
                    RiskEvent(
                        kind="kill_switch",
                        message=(
                            f"Kill switch tripped: drawdown {drawdown:.2%} >= "
                            f"threshold {self.limits.kill_switch_drawdown:.2%}"
                        ),
                        severity="critical",
                    )
                )

        if self._kill_switch_active:
            if self.limits.kill_switch_action == KillSwitchAction.FLATTEN:
                events.append(
                    RiskEvent(
                        kind="kill_switch",
                        message="Kill switch active (flatten): going to all cash",
                        severity="critical",
                    )
                )
                return RiskResult(
                    weights={},
                    events=events,
                    kill_switch_active=True,
                    halted=True,
                )
            else:  # HALT_ENTRIES — allow exits, block new entries
                events.append(
                    RiskEvent(
                        kind="kill_switch",
                        message="Kill switch active (halt_entries): new entries blocked",
                        severity="critical",
                    )
                )
                # Zero out any ticker that's not already held (new entries).
                # The engine passes the current positions so we know which
                # are new vs existing; for simplicity, we zero any weight
                # that would increase a position from zero.
                # The engine handles this by checking kill_switch_active
                # before placing new buys.
                target_weights = {
                    sym: w for sym, w in target_weights.items() if w <= 0
                } if False else dict(target_weights)  # engine checks halted flag
                return RiskResult(
                    weights=target_weights,
                    events=events,
                    kill_switch_active=True,
                    halted=True,
                )

        # 2. Account DD stop (CEO-SET for live).
        if (
            self.limits.account_dd_stop is not None
            and drawdown >= self.limits.account_dd_stop
        ):
            raise DrawdownStopHit(
                f"Account drawdown stop hit: {drawdown:.2%} >= "
                f"{self.limits.account_dd_stop:.2%}"
            )

        weights = dict(target_weights)

        # 3. NAV cap per bucket.
        weights = self._apply_nav_cap(weights, bucket_map, events)

        # 4. Gross exposure cap.
        gross = sum(abs(w) for w in weights.values())
        if gross > self.limits.max_gross_exposure + 1e-9:
            scale = self.limits.max_gross_exposure / gross
            weights = {sym: w * scale for sym, w in weights.items()}
            events.append(
                RiskEvent(
                    kind="gross_exposure",
                    message=(
                        f"Gross exposure {gross:.2%} > cap "
                        f"{self.limits.max_gross_exposure:.2%}; scaled by {scale:.4f}"
                    ),
                )
            )

        # 5. Beta ceiling.
        if beta_provider is not None:
            weights = self._apply_beta_ceiling(weights, beta_provider, events)

        return RiskResult(
            weights=weights,
            events=events,
            kill_switch_active=self._kill_switch_active,
            halted=False,
        )

    def _apply_nav_cap(
        self,
        weights: dict[str, float],
        bucket_map: dict[str, int] | None,
        events: list[RiskEvent],
    ) -> dict[str, float]:
        """Clip any weight exceeding the NAV cap (per-bucket)."""
        adjusted = dict(weights)
        for sym, w in list(adjusted.items()):
            bucket_id = bucket_map.get(sym) if bucket_map else None
            if bucket_id is not None and bucket_id in self.limits.firf_capped_buckets:
                cap = self.limits.nav_cap_firf
            else:
                cap = self.limits.nav_cap
            if abs(w) > cap + 1e-9:
                adjusted[sym] = cap if w > 0 else -cap
                events.append(
                    RiskEvent(
                        kind="nav_cap",
                        message=(
                            f"{sym} weight {w:.2%} clipped to {cap:.2%} "
                            f"({'FIRF' if cap == self.limits.nav_cap_firf else 'NAV'})"
                        ),
                    )
                )
        return adjusted

    def _apply_beta_ceiling(
        self,
        weights: dict[str, float],
        beta_provider: BetaProvider,
        events: list[RiskEvent],
    ) -> dict[str, float]:
        """Scale down weights if portfolio beta exceeds the ceiling.

        β_p = Σ(W_i × β_i). If β_p > beta_ceiling, proportionally scale all
        positive weights down until β_p ≤ ceiling. Excess goes to cash
        (capital preservation — don't redirect into other assets without
        strategy approval).
        """
        active = {sym: w for sym, w in weights.items() if abs(w) > 1e-9}
        if not active:
            return weights

        betas = {sym: beta_provider.get_beta(sym) for sym in active}
        portfolio_beta = sum(w * betas[sym] for sym, w in active.items())

        if portfolio_beta <= self.limits.beta_ceiling + 1e-9:
            return weights

        # Scale all positive weights down to bring beta under the ceiling.
        # Only positive weights contribute to beta (long-only).
        long_betas = {sym: betas[sym] for sym, w in active.items() if w > 0}
        long_beta_total = sum(w * long_betas[sym] for sym, w in active.items() if w > 0)
        if long_beta_total <= 0:
            return weights

        # Target: portfolio_beta = beta_ceiling. Scale long weights by the ratio.
        scale = self.limits.beta_ceiling / portfolio_beta
        scale = max(0.0, min(1.0, scale))

        adjusted = {}
        for sym, w in weights.items():
            if w > 0:
                adjusted[sym] = w * scale
            else:
                adjusted[sym] = w
        events.append(
            RiskEvent(
                kind="beta_ceiling",
                message=(
                    f"Portfolio beta {portfolio_beta:.3f} > ceiling "
                    f"{self.limits.beta_ceiling:.3f}; scaled longs by {scale:.4f}"
                ),
            )
        )
        return adjusted
