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
  data/        # polars-first data layer (D14): databento, yfinance, store, polars_utils, iv_proxy, rates, corp_actions, quality
  config.py    # pydantic-settings loaders for config/*.yaml
  cli.py       # entrypoints
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

`databento` (bars, EOD options for IV, SOFR/OIS) and `yfinance` (fallback) modules are wired but gated behind `@pytest.mark.integration` and runtime credential checks. Set `DATABENTO_API_KEY` + OCI creds to run the real ingest; CI uses synthetic parquet so the data layer is exercised without spend.
