"""Config models + YAML loaders (plan §4, §9.2, §12; D7, D8).

Three config files, all parametrized (D7 conservative defaults live as schema
defaults so the YAML can be sparse), with ``# CEO-SET`` markers on the D8 values
that remain CEO-set until Phase 1 (plan §18):

  config/universe.yaml      — 13-bucket × 3-ETF matrix + parking lot (brief §2).
  config/strategy.yaml      — D7 conservative defaults (ΔH, stops, ROC, NAV cap, beta).
  config/risk_limits.yaml   — hard limits + D8 CEO-SET placeholders (starting
                               capital, account DD stop %, per-ticker max shares).

Loaded via pydantic v2 ``model_validate`` so the YAML is validated at startup —
a bad value fails fast instead of mispricing a trade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CONFIG_DIR",
    "PIMCO_BLOCKLIST",
    "BucketConfig",
    "ParkingLotConfig",
    "RiskLimitsConfig",
    "StrategyConfig",
    "UniverseConfig",
    "load_risk_limits",
    "load_strategy",
    "load_universe",
]

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# D9 whitelist (plan §18): v1–v3 uses a static pre-verified universe. PIMCO is
# banned outright (brief §2 Anti-PIMCO Sovereign Rule). The blocklist is tiny
# and explicit; the full N-CEN/N-PORT/485BPOS engine is v4 (D9).
PIMCO_BLOCKLIST: frozenset[str] = frozenset({"BOND", "HYS", "MUNI", "LTPZ", "PHK", "PCM"})


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BucketConfig(_Base):
    """A single 13-bucket matrix row: ETF A (market-cap core) / B (equal-weight) / C (pure-play)."""

    id: int = Field(ge=1, le=13)
    name: str
    etf_a: str
    etf_b: str
    etf_c: str
    gics_sector: str | None = None
    is_crypto: bool = False
    # FIRF caps this bucket at nav_cap_firf (Tech, Real Estate, Crypto per plan §6.3).
    firf_capped: bool = False

    @property
    def etfs(self) -> tuple[str, str, str]:
        return (self.etf_a, self.etf_b, self.etf_c)


class ParkingLotConfig(_Base):
    """Parking-lot sweep (plan §9 risk/parking_lot.py, brief §4): SUB/VTES vs SGOV by TEY."""

    muni_tickers: list[str] = Field(default_factory=lambda: ["SUB", "VTES"])
    treasury_tickers: list[str] = Field(default_factory=lambda: ["SGOV"])
    max_exception_days: int = Field(default=7, ge=0, le=30)


class UniverseConfig(_Base):
    """The 13-bucket × 3-ETF universe (brief §2) + parking lot."""

    buckets: list[BucketConfig]
    parking_lot: ParkingLotConfig = Field(default_factory=ParkingLotConfig)

    @model_validator(mode="after")
    def _check_structure(self) -> UniverseConfig:
        if len(self.buckets) != 13:
            raise ValueError(f"universe must have exactly 13 buckets, got {len(self.buckets)}")
        ids = [b.id for b in self.buckets]
        if sorted(ids) != list(range(1, 14)):
            raise ValueError(f"bucket ids must be 1..13, got {ids}")
        # No ETF in two buckets; no PIMCO (brief §2 Anti-PIMCO rule, D9).
        seen: dict[str, int] = {}
        for b in self.buckets:
            for t in b.etfs:
                if t in PIMCO_BLOCKLIST:
                    raise ValueError(
                        f"ETF {t!r} in bucket {b.id} is on the PIMCO blocklist (brief §2)"
                    )
                if t in seen:
                    raise ValueError(
                        f"ETF {t!r} appears in bucket {seen[t]} and bucket {b.id} — must be unique"
                    )
                seen[t] = b.id
        return self

    def all_tickers(self) -> list[str]:
        """All universe tickers (39 ETFs) — excludes parking lot."""
        out: list[str] = []
        for b in self.buckets:
            out.extend(b.etfs)
        return out

    def all_tickers_with_parking(self) -> list[str]:
        return (
            self.all_tickers() + self.parking_lot.muni_tickers + self.parking_lot.treasury_tickers
        )


class StrategyConfig(_Base):
    """Strategy parameters — D7 conservative defaults (plan §9.2). All CEO-editable."""

    # L4 tax hurdle (ΔH) — highest hurdle => fewest replacement trades.
    delta_h_st: float = Field(default=0.08, ge=0.0, le=1.0)  # short-term 8%
    delta_h_lt: float = Field(default=0.04, ge=0.0, le=1.0)  # long-term 4%

    # IV-rank-scaled dynamic stops (plan §9 risk/stops.py) — tightest stops.
    stop_iv_rank_low: float = Field(default=0.05, ge=0.0, le=1.0)  # IV rank <50: 5%
    stop_iv_rank_high: float = Field(default=0.08, ge=0.0, le=1.0)  # IV rank >=50: 8%
    stop_crypto: float = Field(default=0.15, ge=0.0, le=1.0)  # crypto: 15%

    # Blended ROC (brief §3 Layer 3): 0.5·ROC3M + 0.5·ROC6M.
    roc_weight_3m: float = Field(default=0.5, ge=0.0, le=1.0)
    roc_weight_6m: float = Field(default=0.5, ge=0.0, le=1.0)

    # Absolute regime voting (brief §3 Layer 2).
    regime_windows_sectors: list[int] = Field(default_factory=lambda: [3, 6, 9, 12])
    regime_votes_sectors: int = Field(default=3)  # >=3 of 4 (75%)
    regime_windows_crypto: list[int] = Field(default_factory=lambda: [1, 2, 3])
    regime_votes_crypto: int = Field(default=2)  # >=2 of 3 (66%)

    # Allocation (brief §5).
    nav_cap: float = Field(default=0.40, ge=0.0, le=1.0)  # 40% absolute NAV cap
    nav_cap_firf: float = Field(default=0.20, ge=0.0, le=1.0)  # 20% under FIRF (plan §6.3)
    beta_ceiling: float = Field(default=1.35, ge=0.0)  # β_p ≤ 1.35
    waterfall: list[float] = Field(default_factory=lambda: [0.5, 0.3, 0.2])  # 50/30/20

    # IV proxy / smoke detector (plan §6.2).
    put_skew_extreme_percentile: float = Field(default=90.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _check_weights(self) -> StrategyConfig:
        if abs(self.roc_weight_3m + self.roc_weight_6m - 1.0) > 1e-9:
            raise ValueError(
                f"roc_weight_3m + roc_weight_6m must sum to 1.0, got {self.roc_weight_3m + self.roc_weight_6m}"
            )
        if abs(sum(self.waterfall) - 1.0) > 1e-9:
            raise ValueError(f"waterfall must sum to 1.0, got {sum(self.waterfall)}")
        return self


class RiskLimitsConfig(_Base):
    """Hard risk limits (plan §12). D8 CEO-SET values are None until Phase 1.

    The ``# CEO-SET`` markers in config/risk_limits.yaml flag the three values
    the CEO must set before Phase 1 live capital (plan §18): starting_capital,
    account_dd_stop_pct, per_ticker_max_shares. They are None here for paper
    trading; ``require_ceo_set_values()`` raises if any is still None when live
    mode is attempted.
    """

    # D8 literal (plan §1, §9.1): $10,000 monthly cash injection.
    monthly_cash_injection: float = Field(default=10_000.0, ge=0.0)

    # --- D8 CEO-SET (placeholders until Phase 1 — plan §18) --- #
    starting_capital: float | None = Field(default=None, ge=0.0)  # CEO-SET
    account_dd_stop_pct: float | None = Field(default=None, ge=0.0, le=1.0)  # CEO-SET
    per_ticker_max_shares: dict[str, int] | None = Field(default=None)  # CEO-SET

    # Hard limits (plan §12).
    max_bucket_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    max_bucket_weight_firf: float = Field(default=0.20, ge=0.0, le=1.0)
    max_gross_exposure_to_nav: float = Field(default=1.0, ge=0.0)  # long-only, unlevered
    beta_ceiling: float = Field(default=1.35, ge=0.0)
    kill_switch_default: bool = False
    # Daily reconciliation mismatch tolerance (plan §12).
    recon_tolerance: float = Field(default=0.01, ge=0.0)

    def require_ceo_set_values(self) -> None:
        """Raise if any D8 CEO-SET value is still None — call before Phase 1 live capital."""
        missing: list[str] = []
        if self.starting_capital is None:
            missing.append("starting_capital")
        if self.account_dd_stop_pct is None:
            missing.append("account_dd_stop_pct")
        if self.per_ticker_max_shares is None:
            missing.append("per_ticker_max_shares")
        if missing:
            raise ValueError(
                f"D8 CEO-SET values still None: {missing}. These must be set before "
                "Phase 1 live capital (plan §18). Paper trading may proceed with None."
            )

    @field_validator("per_ticker_max_shares")
    @classmethod
    def _all_shares_positive(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        if v is None:
            return None
        bad = {k: val for k, val in v.items() if val < 0}
        if bad:
            raise ValueError(f"per_ticker_max_shares must be >= 0, got {bad}")
        return v


# --- Loaders --- #
def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must be a YAML mapping, got {type(data)!r}")
    return data


def load_universe(path: Path | str | None = None) -> UniverseConfig:
    return UniverseConfig.model_validate(
        _load_yaml(Path(path) if path else CONFIG_DIR / "universe.yaml")
    )


def load_strategy(path: Path | str | None = None) -> StrategyConfig:
    return StrategyConfig.model_validate(
        _load_yaml(Path(path) if path else CONFIG_DIR / "strategy.yaml")
    )


def load_risk_limits(path: Path | str | None = None) -> RiskLimitsConfig:
    return RiskLimitsConfig.model_validate(
        _load_yaml(Path(path) if path else CONFIG_DIR / "risk_limits.yaml")
    )
