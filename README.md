# personal-strat-pai

Flat Momentum Strategy trading system — IBKR execution, Oracle Cloud durable storage, polars-first data layer.

**Status:** Phase 0 (P0-2 state plane). No live capital. No IBKR. Paper trading only after CEO sign-off (plan §17, D10).

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

The `oci` SDK is an OPTIONAL extra (the OCI NoSQL backend + provisioner). The
default `uv sync --extra dev` install (CI) uses the in-memory NoSQL backend and
does NOT pull the heavy `oci` SDK:

```bash
uv sync --extra oci           # add the OCI SDK for the real tenancy / provisioner
```

## Layout

```
src/personal_strat_pai/
  data/        # polars-first data layer (D14): caching, repo (bars), fred, rates (FredRateRepo + FredRatesProvider),
               #                store (BarStore + RateSeriesStore), databento, yfinance, polars_utils, iv_proxy (HV/IBKR), corp_actions, quality
  state/       # P0-2: NoSQL state plane (plan §8, D5)
               #   nosql   — InMemoryNoSqlStore + OciNoSqlStore; conditional writes (put_if_absent /
               #              update_if_version / update_if_condition / delete_if_version) are the atomicity
               #              boundary for HIFO lot closure + triplet slot advance.
               #   ledger  — HIFO tax-lot selection (highest-cost-first) + ST(<365d)/LT(>=365d) split at the
               #              lot level (brief §1). Pure functions, hypothesis property-tested.
               #   triplet — A->B->C state machine + 60-day immunization + 30-day wash-sale restricted slot
               #              + append-only slot_history (brief §1, plan §8). Pure transitions + NoSQL fence.
  exec/        # P0-2: exec/lease.py is the v1 NO-OP STUB (D12 — single writer, no fencing). The
               #   IBKR session, router, and target_portfolio modules land in P0-3/P0-4.
  infra/oci/   # P0-2: v1 STORAGE-plane NoSQL table provisioning (Python; 8 tables; NO execution_lease — D12).
               #   provision.py is the CLI: `uv run python -m personal_strat_pai.infra.oci.provision --dry-run`.
  config.py    # pydantic-settings loaders for config/*.yaml
  cli.py       # entrypoints (config-show, data-check, data-ingest, rates-ingest)
config/
  universe.yaml  strategy.yaml  risk_limits.yaml   # parametrized (D7); # CEO-SET markers on D8 values
tests/
  state/       # nosql conditional-write tests + HIFO + triplet hypothesis property tests + cross-module atomicity
  exec/        # exec/lease.py v1 no-op stub tests
  infra/       # provision_all idempotency + D12 fence (execution_lease rejected)
.github/workflows/ci.yml
```

See `docs/technical-design.md` (committed from the approved RBY-2 plan rev 6) for the full architecture.

## Data layer discipline (D14)

- `polars` is the primary engine; `pandas` only at interop boundaries (yfinance, quantstats, ib_async).
- `LazyFrame` + `scan_parquet` for large multi-symbol historical reads; `.collect()` at strategy-decision boundaries.
- Never leak a `LazyFrame` across a module API or into a pre-trade check.
- `data/polars_utils.py` centralizes the lazy/eager boundary.
- The lazy-vs-eager property test (`tests/test_polars_lazy_eager.py`) is a MUST-PASS footgun guard.

## State plane (P0-2; plan §8, D5)

System of record = **Oracle NoSQL Database Cloud Service**. Local SQLite is a
read-through cache. The atomicity boundary is NoSQL conditional writes
(`put_if_absent`, `update_if_version`, `update_if_condition`):
HIFO lot closure and triplet slot advance are each guarded by a conditional
write, so two concurrent stop-outs on the same bucket cannot double-close a
lot or double-advance a slot.

v1 STORAGE-plane tables (provisioned by `infra/oci/nosql_tables.py`):
`tax_lots`, `positions`, `triplet_state`, `realized_pnl`, `order_intent`,
`risk_state`, `ibkr_session`, `sec_compliance`. The `execution_lease` table is
**NOT created in v1** (D12 — single writer, no fencing); `exec/lease.py` is a
no-op stub wired in v2 / P0-5.

CI uses `InMemoryNoSqlStore` (an exact mirror of the conditional-write
semantics) so the hypothesis property tests run without the `oci` SDK. The
hypothesis property tests (HIFO qty/proceeds conservation; triplet never
re-enters inside the 60d immunization window; NoSQL conditional-write
transition atomicity) are the plan §16 acceptance criteria.

### Provisioning the v1 NoSQL tables (CEO-gated)

Creating NoSQL tables incurs a small monthly cost; the first real provision
requires CEO sign-off. The provisioner is idempotent and Python (no Terraform)
to keep the v1 toolchain single-language:

```bash
# Dry-run: prints the 8 tables + DDL without touching the tenancy.
uv run --extra oci python -m personal_strat_pai.infra.oci.provision --dry-run

# Real run (after CEO sign-off): requires OCI_NOSQL_COMPARTMENT_ID.
uv run --extra oci python -m personal_strat_pai.infra.oci.provision --compartment-id ocid1.compartment.oc1...
```

## Live data

`databento` (bars) and `fred` (historical rates) modules are wired but gated behind `@pytest.mark.integration` and runtime credential checks. Set `DATABENTO_API_KEY`, `FRED_API_KEY`, and OCI creds to run the real ingest; CI uses synthetic parquet so the data layer is exercised without spend.

**No double download (CEO directive 2026-07-18 / 2026-07-19):** the shared `data/caching.py` `compute_fetch_ranges` drives both `BarRepo` (bars) and `FredRateRepo` (rates). A first request for a key (symbol / FRED series) bootstraps the **max available historical range**; later requests fetch **only** the forward gap from the latest day in storage up to the requested date (plus a one-time front-gap fill). Upserts dedupe by `(symbol, ts)` / `(series, ts)` so overlapping re-fetches never duplicate rows. Re-requesting an already-cached range triggers zero fetcher calls. The same policy applies to FRED (CEO 2026-07-19).

**Rates (CEO directive 2026-07-19):** FRED is the primary source for **historical** SOFR / OIS proxy (Treasury CMT yields) / real rates (TIPS yields), used in backtesting. Series: `SOFR`, `DGS3MO`/`DGS6MO`/`DGS1`/`DGS2`/`DGS5`/`DGS10`/`DGS20`/`DGS30`, `DFII5`/`DFII10`/`DFII30`. `FredRatesProvider` builds a `RatesCurve` from the cache; `DatabentoRatesProvider` remains the live forward-curve system of record (P0-3, D3). FRED publishes percent; the normalizer converts to decimal at the boundary.

**IV proxy (CEO directive 2026-07-18):** for backtesting, IV is proxied by **HV** (realized volatility) computed from the daily bars we already ingest — no OPRA / EOD-options spend (`data/iv_proxy.py`, `HvIvProvider`). For live/paper, IV/options come via **IBKR** (`IbkrIvProvider`, wired in P0-3). The `IvProvider` protocol makes the swap a one-line composition-root change. This supersedes design decision D2.
