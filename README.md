# personal-strat-pai

Flat Momentum Strategy trading system — IBKR execution, Oracle Cloud durable storage, polars-first data layer.

**Status:** Phase 0 (P0-1 foundation). No live capital. No IBKR. Paper trading only after CEO sign-off (plan §17, D10).

## Quickstart

Toolchain: [`uv`](https://docs.astral.sh/uv/) (D13). Python 3.12.

```bash
uv sync                       # create venv, install deps from uv.lock
uv run pytest                 # run tests (skips live-data integration tests by default)
uv run ruff check .           # lint
uv run mypy src               # typecheck
uv run python -m personal_strat_pai.cli --help
```

CI runs `uv sync --frozen && uv run pytest` + ruff + mypy + gitleaks on every PR.

## Layout

```
src/personal_strat_pai/
  data/        # polars-first data layer (D14): caching, repo (bars), fred, rates (FredRateRepo + FredRatesProvider),
               #                store (BarStore + RateSeriesStore), databento, yfinance, polars_utils, iv_proxy (HV/IBKR), corp_actions, quality
  config.py    # pydantic-settings loaders for config/*.yaml
  cli.py       # entrypoints (config-show, data-check, data-ingest, rates-ingest)
config/
  universe.yaml  strategy.yaml  risk_limits.yaml   # parametrized (D7); # CEO-SET markers on D8 values
tests/
.github/workflows/ci.yml
```

See `docs/technical-design.md` (committed from the approved RBY-2 plan rev 6) for the full architecture.

## Data layer discipline (D14)

- `polars` is the primary engine; `pandas` only at interop boundaries (yfinance, quantstats, ib_async).
- `LazyFrame` + `scan_parquet` for large multi-symbol historical reads; `.collect()` at strategy-decision boundaries.
- Never leak a `LazyFrame` across a module API or into a pre-trade check.
- `data/polars_utils.py` centralizes the lazy/eager boundary.
- The lazy-vs-eager property test (`tests/test_polars_lazy_eager.py`) is a MUST-PASS footgun guard.

## Live data

`databento` (bars) and `fred` (historical rates) modules are wired but gated behind `@pytest.mark.integration` and runtime credential checks. Set `DATABENTO_API_KEY`, `FRED_API_KEY`, and OCI creds to run the real ingest; CI uses synthetic parquet so the data layer is exercised without spend.

**No double download (CEO directive 2026-07-18 / 2026-07-19):** the shared `data/caching.py` `compute_fetch_ranges` drives both `BarRepo` (bars) and `FredRateRepo` (rates). A first request for a key (symbol / FRED series) bootstraps the **max available historical range**; later requests fetch **only** the forward gap from the latest day in storage up to the requested date (plus a one-time front-gap fill). Upserts dedupe by `(symbol, ts)` / `(series, ts)` so overlapping re-fetches never duplicate rows. Re-requesting an already-cached range triggers zero fetcher calls. The same policy applies to FRED (CEO 2026-07-19).

**Rates (CEO directive 2026-07-19):** FRED is the primary source for **historical** SOFR / OIS proxy (Treasury CMT yields) / real rates (TIPS yields), used in backtesting. Series: `SOFR`, `DGS3MO`/`DGS6MO`/`DGS1`/`DGS2`/`DGS5`/`DGS10`/`DGS20`/`DGS30`, `DFII5`/`DFII10`/`DFII30`. `FredRatesProvider` builds a `RatesCurve` from the cache; `DatabentoRatesProvider` remains the live forward-curve system of record (P0-3, D3). FRED publishes percent; the normalizer converts to decimal at the boundary.

**IV proxy (CEO directive 2026-07-18):** for backtesting, IV is proxied by **HV** (realized volatility) computed from the daily bars we already ingest — no OPRA / EOD-options spend (`data/iv_proxy.py`, `HvIvProvider`). For live/paper, IV/options come via **IBKR** (`IbkrIvProvider`, wired in P0-3). The `IvProvider` protocol makes the swap a one-line composition-root change. This supersedes design decision D2.
