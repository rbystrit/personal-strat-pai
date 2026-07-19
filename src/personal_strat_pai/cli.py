"""CLI entrypoints (plan §4).

P0-1 surface:
  config-show   — load + print the three configs (validates at startup).
  data-check    — run data-quality checks over a parquet bar store.
  data-ingest   — databento -> parquet store via the caching BarRepo
                  (no piece of data downloaded twice; CEO directive 2026-07-18).
                  Integration-gated; needs DATABENTO_API_KEY.
  rates-ingest  — FRED -> parquet rate-observation store via the caching
                  FredRateRepo (same no-double-download policy; CEO directive
                  2026-07-19). Pulls SOFR + Treasury CMT (OIS proxy) + TIPS
                  (real rates). Integration-gated; needs FRED_API_KEY.

Later phases add: run-backtest, sec-audit, reconcile, arm-backup, lease-show
(plan §4 cli.py). Kept minimal here so P0-1 ships a working entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from personal_strat_pai.config import load_risk_limits, load_strategy, load_universe

__all__ = ["main"]


def _cmd_config_show(args: argparse.Namespace) -> int:
    cfg_dir = Path(args.config_dir) if args.config_dir else None
    universe = load_universe(cfg_dir / "universe.yaml" if cfg_dir else None)
    strategy = load_strategy(cfg_dir / "strategy.yaml" if cfg_dir else None)
    risk = load_risk_limits(cfg_dir / "risk_limits.yaml" if cfg_dir else None)
    print(f"# universe: {len(universe.buckets)} buckets, {len(universe.all_tickers())} ETFs")
    for b in universe.buckets:
        print(f"  {b.id:2d}. {b.name:<22} A={b.etf_a:<6} B={b.etf_b:<6} C={b.etf_c:<6}")
    print(
        f"# strategy: ΔH ST={strategy.delta_h_st} LT={strategy.delta_h_lt} "
        f"stops=({strategy.stop_iv_rank_low},{strategy.stop_iv_rank_high},crypto={strategy.stop_crypto})"
    )
    print(
        f"# risk: monthly_cash_injection=${risk.monthly_cash_injection:,.0f} "
        f"(D8 CEO-SET: starting_capital={risk.starting_capital}, "
        f"dd_stop={risk.account_dd_stop_pct}, max_shares_set={bool(risk.per_ticker_max_shares)})"
    )
    return 0


def _cmd_data_check(args: argparse.Namespace) -> int:
    from personal_strat_pai.data.quality import validate_bars
    from personal_strat_pai.data.store import BarStore

    store = BarStore(args.base_uri)
    df = store.read_bars_eager(kind=args.kind, symbols=args.symbols or None)
    if df.is_empty():
        print(f"no bars in {store._kind_dir(args.kind)}")
        return 1
    report = validate_bars(df, raise_on_fail=False)
    print(report.summary())
    for v in report.violations[:20]:
        print(f"  [{v.check}] {v.symbol}: {v.detail}")
    return 0 if report.passed else 2


def _cmd_data_ingest(args: argparse.Namespace) -> int:
    from personal_strat_pai.data.databento import DatabentoClient
    from personal_strat_pai.data.quality import validate_bars
    from personal_strat_pai.data.repo import BarRepo
    from personal_strat_pai.data.store import BarStore

    universe = load_universe(Path(args.config_dir) / "universe.yaml" if args.config_dir else None)
    client = DatabentoClient()
    store = BarStore(args.base_uri)
    repo = BarRepo(store, client, bootstrap_start=args.bootstrap_start)
    symbols = universe.all_tickers_with_parking()
    print(
        f"ingesting {len(symbols)} symbols from databento -> {args.base_uri} "
        f"(no-double-download; bootstrap_start={args.bootstrap_start}, end={args.end})"
    )
    # get_bars fetches only the missing ranges, upserts into the store, and
    # returns a lazy scan of the requested range. Collect eagerly for the
    # quality gate (boundary -> eager, D14(b)).
    lazy = repo.get_bars(symbols, start=args.start, end=args.end, kind="daily")
    df = lazy.collect()
    if df.is_empty():
        print(f"no rows fetched for {args.start}..{args.end}; check creds/range")
        return 3
    report = validate_bars(df, raise_on_fail=False)
    if not report.passed:
        print(f"data-quality FAIL — quarantining: {report.summary()}")
        return 3
    cov = repo.coverage(kind="daily", symbols=symbols)
    print(
        f"OK: {df.height} rows in [{args.start}, {args.end}); "
        f"{len(cov)} symbols cached. {report.summary()}"
    )
    return 0


def _cmd_rates_ingest(args: argparse.Namespace) -> int:
    from personal_strat_pai.data.fred import ALL_FRED_SERIES_IDS, FredClient
    from personal_strat_pai.data.rates import FredRateRepo
    from personal_strat_pai.data.store import RateSeriesStore

    bootstrap = _parse_date_arg(args.bootstrap_start)
    end = _parse_date_arg(args.end)
    client = FredClient()
    store = RateSeriesStore(args.base_uri)
    repo = FredRateRepo(store, client, bootstrap_start=bootstrap)
    series_ids = ALL_FRED_SERIES_IDS if args.series is None else args.series
    print(
        f"ingesting {len(series_ids)} FRED series -> {args.base_uri} "
        f"(no-double-download; bootstrap_start={bootstrap}, end={end})"
    )
    # get_observations fetches only the missing ranges, upserts into the store,
    # and returns a lazy scan of the requested range. Collect eagerly for the
    # summary (boundary -> eager, D14(b)). The FRED API key is checked lazily
    # on the first fetch — catch RuntimeError for a clean no-creds exit.
    try:
        lazy = repo.get_observations(series_ids, start=args.start, end=end)
        df = lazy.collect()
    except RuntimeError as exc:
        print(f"FRED fetch failed: {exc}")
        return 3
    cov = repo.coverage(series=series_ids)
    if df.is_empty() and not cov:
        print(f"no observations fetched for {args.start}..{end}; check FRED_API_KEY/range")
        return 3
    by_series = df.group_by("series").len().sort("series") if not df.is_empty() else None
    print(f"OK: {df.height} observations in [{args.start}, {end}); {len(cov)} series cached.")
    if by_series is not None:
        for row in by_series.iter_rows(named=True):
            print(f"  {row['series']:<10} {row['len']} rows")
    return 0


def _parse_date_arg(s: str) -> date:
    """Parse an ISO-8601 date string from a CLI arg."""
    from datetime import datetime as _dt

    return _dt.fromisoformat(s).date()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="personal-strat-pai", description="Flat Momentum Strategy CLI."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("config-show", help="load + print the three configs (validates them).")
    p_show.add_argument("--config-dir", default=None, help="override config/ directory")
    p_show.set_defaults(func=_cmd_config_show)

    p_check = sub.add_parser("data-check", help="run data-quality checks over a parquet bar store.")
    p_check.add_argument("--base-uri", default="data/local/bars", help="bar store base uri")
    p_check.add_argument("--kind", default="daily", choices=["daily", "minute"])
    p_check.add_argument("--symbols", nargs="*", default=None)
    p_check.set_defaults(func=_cmd_data_check)

    p_ingest = sub.add_parser(
        "data-ingest",
        help="databento -> parquet store via the caching BarRepo (needs DATABENTO_API_KEY).",
    )
    p_ingest.add_argument("--base-uri", default="data/local/bars")
    p_ingest.add_argument("--start", required=True, help="ISO-8601 start date (inclusive)")
    p_ingest.add_argument("--end", required=True, help="ISO-8601 end date (exclusive)")
    p_ingest.add_argument(
        "--bootstrap-start",
        default="2000-01-01",
        help="max-history floor for first-time pulls (CEO: bootstrap with max range available).",
    )
    p_ingest.add_argument("--config-dir", default=None)
    p_ingest.set_defaults(func=_cmd_data_ingest)

    p_rates = sub.add_parser(
        "rates-ingest",
        help=(
            "FRED -> parquet rate store via the caching FredRateRepo "
            "(needs FRED_API_KEY; same no-double-download policy as bars)."
        ),
    )
    p_rates.add_argument(
        "--base-uri", default="data/local/rates", help="rate-observation store base uri"
    )
    p_rates.add_argument(
        "--start",
        default=None,
        help="ISO-8601 start date (inclusive); defaults to bootstrap_start",
    )
    p_rates.add_argument("--end", required=True, help="ISO-8601 end date (exclusive)")
    p_rates.add_argument(
        "--bootstrap-start",
        default="2000-01-01",
        help="max-history floor for first-time pulls (CEO: bootstrap with max range available).",
    )
    p_rates.add_argument(
        "--series",
        nargs="*",
        default=None,
        help="FRED series ids to pull; defaults to the full FIRF catalog (SOFR + DGS + DFII).",
    )
    p_rates.set_defaults(func=_cmd_rates_ingest)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
