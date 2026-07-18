"""SOFR / OIS swap curve — databento source (plan §6.3, D3).

The Fixed-Income Regime Filter (FIRF, plan §6.3) monitors the slope of the SOFR
Forward Curve (2y vs 10y OIS swap spread) alongside real rates. If the curve
aggressively flattens or inverts while real rates surge, a macro circuit breaker
caps growth/duration buckets (Tech, Real Estate, Crypto) at 20% NAV.

Source of record: **databento** (SOFR/OIS forward curve). FRED is a free
sanity cross-check only, not the system of record.

For P0-1 the ``DatabentoRatesProvider`` is wired but integration-gated (needs
``DATABENTO_API_KEY``); ``SyntheticRatesProvider`` is a deterministic provider
for tests/dev so the IV proxy and FIRF can run without spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

__all__ = [
    "DEFAULT_RISK_FREE_RATE",
    "DatabentoRatesProvider",
    "RatesCurve",
    "RatesProvider",
    "SyntheticRatesProvider",
    "risk_free_continuous",
]

# Conservative default used when no curve is available (IV proxy, FIRF tests).
# ~5% continuous — a reasonable recent-regime placeholder; overridden by the live curve.
DEFAULT_RISK_FREE_RATE: float = 0.05


@dataclass(frozen=True, slots=True)
class RatesCurve:
    """SOFR/OIS forward curve snapshot (plan §6.3)."""

    as_of: date
    tenors_years: list[float] = field(default_factory=list)  # e.g. [0.25, 0.5, 1, 2, 5, 10, 20]
    ois_rates: list[float] = field(default_factory=list)  # continuous annualized rates
    sofr_rates: list[float] = field(default_factory=list)  # continuous annualized rates
    real_rates: list[float] = field(default_factory=list)  # real rates per tenor (optional)

    @property
    def ois_slope_2y10y(self) -> float | None:
        """2y-10y OIS swap spread (plan §6.3 FIRF trigger). Positive = normal curve."""
        r2 = self._rate_at(2.0, self.ois_rates)
        r10 = self._rate_at(10.0, self.ois_rates)
        if r2 is None or r10 is None:
            return None
        return r10 - r2

    def _rate_at(self, tenor: float, rates: list[float]) -> float | None:
        if not self.tenors_years or not rates:
            return None
        try:
            idx = self.tenors_years.index(tenor)
        except ValueError:
            # linear interpolation between bracketing tenors
            idx_lo = max(i for i, t in enumerate(self.tenors_years) if t <= tenor)
            idx_hi = min(i for i, t in enumerate(self.tenors_years) if t >= tenor)
            if idx_lo == idx_hi:
                return rates[idx_lo]
            t_lo, t_hi = self.tenors_years[idx_lo], self.tenors_years[idx_hi]
            r_lo, r_hi = rates[idx_lo], rates[idx_hi]
            if t_hi == t_lo:
                return r_lo
            return r_lo + (r_hi - r_lo) * (tenor - t_lo) / (t_hi - t_lo)
        return rates[idx]


class RatesProvider(Protocol):
    """Rates source protocol (plan §6.3, D3)."""

    def get_curve(self, as_of: date) -> RatesCurve: ...


class SyntheticRatesProvider:
    """Deterministic rates provider for tests/dev (no databento spend)."""

    def __init__(
        self,
        ois: list[float] | None = None,
        sofr: list[float] | None = None,
        tenors: list[float] | None = None,
    ) -> None:
        self.tenors = tenors if tenors is not None else [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        self.ois = ois if ois is not None else [0.045, 0.045, 0.046, 0.045, 0.047, 0.050, 0.052]
        self.sofr = sofr if sofr is not None else [0.045, 0.045, 0.046, 0.045, 0.047, 0.050, 0.052]

    def get_curve(self, as_of: date) -> RatesCurve:
        return RatesCurve(
            as_of=as_of,
            tenors_years=list(self.tenors),
            ois_rates=list(self.ois),
            sofr_rates=list(self.sofr),
        )


class DatabentoRatesProvider:
    """databento SOFR/OIS provider (plan §6.3, D3). Integration-gated (needs API key)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def get_curve(self, as_of: date) -> RatesCurve:  # pragma: no cover - integration
        if not self.api_key:
            raise RuntimeError(
                "DatabentoRatesProvider needs DATABENTO_API_KEY (plan §6.3). "
                "Use SyntheticRatesProvider for tests/dev."
            )
        raise NotImplementedError(
            "databento SOFR/OIS timeseries fetch is wired in P0-1's integration test "
            "path; the production loader lands with the live ingest job. Use "
            "SyntheticRatesProvider until creds are set."
        )


def risk_free_continuous(provider: RatesProvider, as_of: date, tenor_years: float = 1.0) -> float:
    """The continuous risk-free rate used in Black-Scholes (IV proxy) / FIRF.

    Falls back to DEFAULT_RISK_FREE_RATE if the curve is unavailable.
    """
    curve = provider.get_curve(as_of)
    r = curve._rate_at(tenor_years, curve.ois_rates)
    return r if r is not None else DEFAULT_RISK_FREE_RATE
