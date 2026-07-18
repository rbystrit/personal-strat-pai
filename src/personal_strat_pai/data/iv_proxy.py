"""Self-built IV proxy from EOD options chain snapshot — NO OPRA (plan §6.2, D2).

We build implied volatility ourselves from a narrow end-of-day options chain
snapshot (databento EOD options dataset — a fraction of OPRA cost) for the
~45-ETF universe: 30-day and 90-day at-the-money IV plus a small put-skew
sample. Black-Scholes inversion via scipy. No OPRA spend in v1.

Accuracy tradeoff (flagged, plan §6.2/§19): a self-built IV proxy from EOD
snapshots is less precise than live OPRA. Phase 0 quantifies proxy-vs-OPRA
divergence; if it materially changes smoke-detector verdicts, escalate to CEO
to reconsider D2 (OPRA is the higher-accuracy option).

The interface (``IvProvider`` protocol) is parameterized so the source can be
swapped to OPRA later WITHOUT touching the sieve:
  - ``EodChainIvProvider`` (default, D2) — computes IV from an EOD chain snapshot.
  - ``OpraIvProvider`` (stub) — the parameterized swap target; raises
    NotImplementedError until D2 is revisited. The sieve calls the protocol, so
    swapping is a one-line wiring change in the composition root.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import erf, exp, log, sqrt
from typing import Protocol

import polars as pl
from scipy.optimize import brentq

__all__ = [
    "CHAIN_COLUMNS",
    "EXTREME_PUT_SKEW_PERCENTILE",
    "PUT_SKEW_MONEYNESS",
    "EodChainIvProvider",
    "IvProvider",
    "IvSnapshot",
    "OpraIvProvider",
    "OptionType",
    "black_scholes_price",
    "compute_iv_from_chain",
    "implied_vol",
    "rolling_put_skew_percentile",
]

# --- Chain schema (EOD options chain snapshot) --- #
CHAIN_COLUMNS: tuple[str, ...] = (
    "symbol",  # underlying symbol
    "expiration",  # option expiration date (date)
    "strike",  # strike price
    "type",  # "C" | "P"
    "bid",  # bid price
    "ask",  # ask price
)

PUT_SKEW_MONEYNESS: float = 0.90  # OTM put strike = 0.90 * spot (proxy for ~25-delta put)
EXTREME_PUT_SKEW_PERCENTILE: float = 90.0  # plan §6.2: extreme put skew >= 90th pct of 12m


# --- Black-Scholes --- #
OptionType = str  # "C" or "P"


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (scipy.stats.norm is heavier; use erf for a single call)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType, *, q: float = 0.0
) -> float:
    """Black-Scholes-Merton price. ``T`` in years, ``r``/``q`` continuous rates.

    ``option_type`` is "C" (call) or "P" (put). Returns the option price.
    """
    if T <= 0 or sigma <= 0:
        # At/after expiry or zero vol: intrinsic value.
        if option_type == "C":
            return max(S * (1.0 - q) - K, 0.0)
        return max(K - S, 0.0)
    sqrtT_sigma = sqrt(T) * sigma
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sqrtT_sigma
    d2 = d1 - sqrtT_sigma
    disc_K = K * exp(-r * T)
    spot_adj = S * exp(-q * T)
    if option_type == "C":
        return spot_adj * _norm_cdf(d1) - disc_K * _norm_cdf(d2)
    if option_type == "P":
        return disc_K * _norm_cdf(-d2) - spot_adj * _norm_cdf(-d1)
    raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    *,
    q: float = 0.0,
    bracket: tuple[float, float] = (1e-4, 5.0),
) -> float:
    """Invert Black-Scholes to recover implied vol from a market price (scipy brentq).

    Returns the implied volatility (annualized, decimal). Raises ValueError if
    the price is outside the no-arbitrage bounds or brentq fails to bracket.
    """
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")

    # No-arb bounds: |S*exp(-qT) - K*exp(-rT)| bounds option value.
    intrinsic_call = max(S * (1.0 - q) - K, 0.0)
    intrinsic_put = max(K - S, 0.0)
    if option_type == "C":
        lower, upper = intrinsic_call, S * (1.0 - q)
    elif option_type == "P":
        lower, upper = intrinsic_put, K
    else:
        raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")
    if price < lower - 1e-9 or price > upper + 1e-9:
        raise ValueError(
            f"price {price} outside no-arb bounds [{lower}, {upper}] for "
            f"{option_type} S={S} K={K} T={T} r={r}"
        )

    def objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type, q=q) - price

    # Ensure the bracket straddles zero.
    lo, hi = bracket
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"implied_vol: brentq bracket [{lo}, {hi}] does not straddle zero "
            f"(f(lo)={f_lo}, f(hi)={f_hi}) for price={price}"
        )
    return float(brentq(objective, lo, hi, xtol=1e-8, rtol=1e-8, maxiter=200))


# --- IV snapshot --- #
@dataclass(frozen=True, slots=True)
class IvSnapshot:
    """Per-symbol daily IV snapshot (plan §6.2) — persisted to Object Storage + NoSQL."""

    symbol: str
    as_of: date
    iv_30d: float  # 30-day ATM IV (decimal, annualized)
    iv_90d: float  # 90-day ATM IV (decimal, annualized)
    term_slope: float  # iv_30d - iv_90d  (>0 => backwardation, plan §6.2)
    put_skew_30d: float  # 30D OTM-put IV - 30D ATM IV (proxy for 25-delta skew)
    put_skew_percentile_12m: float | None  # pct of 30D put skew vs trailing 12m (0-100)
    backwardation: bool  # iv_30d > iv_90d  (plan §6.2 smoke-detector trigger)
    extreme_put_skew: bool  # put_skew_percentile_12m >= 90 (plan §6.2 smoke-detector trigger)


def _nearest_expiry(chain: pl.DataFrame, target_days: float, as_of: date) -> date | None:
    """Pick the expiration nearest to ``target_days`` from ``as_of``."""
    avail = chain["expiration"].unique().sort()
    dates = [d for d in avail.to_list() if isinstance(d, date)]
    if not dates:
        return None
    return min(dates, key=lambda d: abs((d - as_of).days - target_days))


def _nearest_strike(
    strikes: list[float], target: float, *, prefer_le: bool = False
) -> float | None:
    """Pick the strike nearest to ``target``. If ``prefer_le``, prefer strikes <= target."""
    if not strikes:
        return None
    if prefer_le:
        below = [s for s in strikes if s <= target]
        pool = below if below else strikes
    else:
        pool = strikes
    return min(pool, key=lambda s: abs(s - target))


def _atm_iv(
    chain: pl.DataFrame, expiry: date | None, spot: float, r: float, as_of: date
) -> float | None:
    """ATM IV at ``expiry`` — average of the nearest-strike call IV and put IV."""
    if expiry is None:
        return None
    sub = chain.filter(pl.col("expiration") == expiry)
    if sub.is_empty():
        return None
    atm_strike = _nearest_strike(sub["strike"].unique().to_list(), spot)
    if atm_strike is None:
        return None
    T = max((expiry - as_of).days, 1) / 365.0
    mid = sub.filter(pl.col("strike") == atm_strike).with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid")
    )
    ivs: list[float] = []
    for row in mid.filter(pl.col("mid") > 0).iter_rows(named=True):
        try:
            ivs.append(implied_vol(row["mid"], spot, atm_strike, T, r, row["type"]))
        except ValueError:
            continue
    if not ivs:
        return None
    return sum(ivs) / len(ivs)


def _otm_put_iv(
    chain: pl.DataFrame, expiry: date | None, spot: float, r: float, as_of: date, moneyness: float
) -> float | None:
    """IV of the OTM put nearest to ``moneyness * spot`` (PUT_SKEW_MONEYNESS=0.90)."""
    if expiry is None:
        return None
    target_strike = moneyness * spot
    puts = chain.filter((pl.col("expiration") == expiry) & (pl.col("type") == "P"))
    if puts.is_empty():
        return None
    atm_strike = _nearest_strike(puts["strike"].unique().to_list(), target_strike, prefer_le=True)
    if atm_strike is None:
        return None
    T = max((expiry - as_of).days, 1) / 365.0
    row = (
        puts.filter(pl.col("strike") == atm_strike)
        .with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"))
        .filter(pl.col("mid") > 0)
    )
    if row.is_empty():
        return None
    r0 = row.row(0, named=True)
    try:
        return implied_vol(r0["mid"], spot, atm_strike, T, r, "P")
    except ValueError:
        return None


def compute_iv_from_chain(
    chain: pl.DataFrame,
    spot: float,
    r: float,
    as_of: date,
    symbol: str,
    *,
    put_skew_moneyness: float = PUT_SKEW_MONEYNESS,
) -> IvSnapshot:
    """Compute an IV snapshot from an EOD options chain snapshot (plan §6.2, D2).

    ``chain`` conforms to CHAIN_COLUMNS. ``spot`` is the underlying close on
    ``as_of``. ``r`` is the risk-free continuous rate (from data/rates.py).
    Returns the raw snapshot (12m percentile is None here; the provider fills it
    from history via rolling_put_skew_percentile).
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    my_chain = chain.filter(pl.col("symbol") == symbol)
    if my_chain.is_empty():
        raise ValueError(f"no options in chain for {symbol!r}")

    exp_30 = _nearest_expiry(my_chain, 30.0, as_of)
    exp_90 = _nearest_expiry(my_chain, 90.0, as_of)

    iv_30d = _atm_iv(my_chain, exp_30, spot, r, as_of)
    iv_90d = _atm_iv(my_chain, exp_90, spot, r, as_of)
    if iv_30d is None or iv_90d is None:
        raise ValueError(
            f"could not compute ATM IV for {symbol} on {as_of} (iv_30d={iv_30d}, iv_90d={iv_90d})"
        )

    atm_30 = iv_30d
    otm_put_30 = _otm_put_iv(my_chain, exp_30, spot, r, as_of, put_skew_moneyness)
    put_skew_30d = (otm_put_30 - atm_30) if otm_put_30 is not None else 0.0

    term_slope = iv_30d - iv_90d
    return IvSnapshot(
        symbol=symbol,
        as_of=as_of,
        iv_30d=iv_30d,
        iv_90d=iv_90d,
        term_slope=term_slope,
        put_skew_30d=put_skew_30d,
        put_skew_percentile_12m=None,  # filled by the provider from history
        backwardation=term_slope > 0.0,
        extreme_put_skew=False,  # filled by the provider from history
    )


def rolling_put_skew_percentile(
    history_put_skew: list[float],
    current_put_skew: float,
    *,
    window: int = 252,  # ~12 months of trading days
) -> float:
    """Percentile (0-100) of current 30D put skew vs trailing 12m window (plan §6.2)."""
    if not history_put_skew:
        return 50.0  # no history => neutral (not extreme)
    windowed = history_put_skew[-window:] if len(history_put_skew) > window else history_put_skew
    n = len(windowed)
    rank = sum(1 for v in windowed if v <= current_put_skew)
    return 100.0 * rank / n


# --- Provider protocol (D2 parameterization for later OPRA swap) --- #
class IvProvider(Protocol):
    """IV source protocol (plan §6.2). The sieve calls this, never the chain directly.

    Swapping EOD-chain-IV -> OPRA is a one-line wiring change in the composition
    root; the sieve is untouched (D2 parameterization).
    """

    def get_snapshot(self, symbol: str, as_of: date) -> IvSnapshot: ...


class EodChainIvProvider:
    """Default IV provider: self-built IV from an EOD chain snapshot (plan §6.2, D2).

    The chain source is injected (``chain_loader``) so this provider is testable
    with a synthetic chain and swappable to databento EOD options in production.
    History of put-skew values is injected for the 12m percentile.
    """

    def __init__(
        self,
        chain_loader: ChainLoader,
        spot_loader: SpotLoader,
        risk_free_rate: float,
        put_skew_history: dict[str, list[float]] | None = None,
    ) -> None:
        self.chain_loader = chain_loader
        self.spot_loader = spot_loader
        self.r = risk_free_rate
        self.put_skew_history = put_skew_history or {}

    def get_snapshot(self, symbol: str, as_of: date) -> IvSnapshot:
        chain = self.chain_loader(symbol, as_of)
        spot = self.spot_loader(symbol, as_of)
        snap = compute_iv_from_chain(chain, spot, self.r, as_of, symbol)
        hist = self.put_skew_history.get(symbol, [])
        if hist:
            pct = rolling_put_skew_percentile(hist, snap.put_skew_30d)
            return IvSnapshot(
                symbol=snap.symbol,
                as_of=snap.as_of,
                iv_30d=snap.iv_30d,
                iv_90d=snap.iv_90d,
                term_slope=snap.term_slope,
                put_skew_30d=snap.put_skew_30d,
                put_skew_percentile_12m=pct,
                backwardation=snap.backwardation,
                extreme_put_skew=pct >= EXTREME_PUT_SKEW_PERCENTILE,
            )
        return snap


class OpraIvProvider:
    """Parameterized OPRA swap target (plan §6.2, D2). NOT wired in v1.

    Raises NotImplementedError until D2 is revisited (Phase 0 validation of
    EOD-chain-IV-vs-OPRA divergence). The sieve codes against ``IvProvider``, so
    enabling OPRA is a one-line composition-root change.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "OpraIvProvider is the D2 parameterized swap target, not wired in v1. "
            "Phase 0 quantifies EOD-chain-IV-vs-OPRA divergence; if it materially "
            "changes smoke-detector verdicts, escalate to CEO to reconsider D2."
        )

    def get_snapshot(self, symbol: str, as_of: date) -> IvSnapshot:  # pragma: no cover
        raise NotImplementedError


# --- Loader protocol types (for type-checking the provider composition) --- #
class ChainLoader(Protocol):
    def __call__(self, symbol: str, as_of: date) -> pl.DataFrame: ...


class SpotLoader(Protocol):
    def __call__(self, symbol: str, as_of: date) -> float: ...
