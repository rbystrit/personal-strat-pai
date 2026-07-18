# Technical Design & Architecture — Flat Momentum Strategy Trading System

**Status:** Revision 6 — incorporates CEO comment `79201ed3` (2026-07-18) with three changes: **(D12) the OCI Functions backup-execution path is deferred out of v1 — v1 runs on the podman primary only, a single execution path, so split-brain is impossible by construction and the `execution_lease` fencing becomes a v2 concern; (D13) `uv` is the project manager** (venv, deps, lockfile, running scripts, workspace — authoritative, not a side note); **(D14) `polars` is the primary DataFrame engine** for performance-critical paths (bar ingestion, backtest vectorized signals, IV proxy, ROC/term-structure math) with explicit lazy-evaluation discipline, while `pandas` is kept only at interop boundaries (yfinance, quantstats, ib_async DataFrame returns).** The D11 architecture (podman-primary + OCI durable storage) is retained; only the **backup-execution** piece of D11 is pushed to v2. The durable-storage plane (Oracle NoSQL, Object Storage, Vault, Logging/Monitoring) is unchanged and is still part of v1. Awaiting CEO approval of this revision.

Revision 5 (preserved for history): corrected a factual error from earlier revisions — the project repo `personal-strat-pai` is an empty greenfield, not a repo with pre-existing OCI Functions images, EDGAR tooling, backtest harness, Terraform, or CI. Revision 5 removed every `reuse existing X` assumption; all code is authored fresh.

Revision 4 (preserved for history): incorporated CEO decision **D11** (comment `be3c1e75`, 2026-07-18) — primary execution moves to a podman container on the current machine; OCI Functions become the backup-execution path; OCI remains the durable-data-storage plane exactly as planned in revision 3.
**Author:** Founding Quant Trading Engineer
**Scope:** Full-stack technical design for the strategy in the attached brief (document key `using-all-available-documents-formulate-the-stra`). Design-only; implementation is a separate, CEO-approved gate.
**Hard constraints (CEO-set):** Python · **project managed with `uv` (D13)** · **`polars`-first data layer with lazy-eval discipline (D14); `pandas` only at interop boundaries** · IBKR execution · **Compute (v1): a single podman container on the current machine (no hourly VM charge) — no OCI Functions in v1 (D12); the OCI Functions backup-execution path is deferred to v2. Durable storage = Oracle Cloud (Object Storage, NoSQL, Vault, Logging/Monitoring), in v1.** · databento is the primary data feed · yfinance is the fallback · all persistent state in **Oracle NoSQL Database Cloud Service** · options via a **self-built IV proxy (no OPRA)** · SOFR/OIS from **databento** · small PAYGO Object Storage overage is acceptable.

> **What changed in revision 6 (CEO comment `79201ed3`, 2026-07-18) — three changes:**

1. **D12 — OCI Functions backup deferred to v2.** v1 runs on the podman primary only. There is **one execution path** in v1, so the §3.3 `execution_lease` fencing (introduced in D11 to prevent split-brain between primary and backup) is **not needed in v1** — split-brain is impossible by construction with a single writer. The fencing logic, the `functions/` packaging, the ephemeral in-function Gateway, the OCI Resource Scheduler wiring, and the backup failover policy are all **deferred to v2** and re-introduced when the backup comes online. The `execution_lease` NoSQL table is likewise **deferred to v2** (kept in the §8 schema as a forward-compat placeholder, not created in v1). **The durable-storage plane (Oracle NoSQL, Object Storage, Vault, Logging/Monitoring) is unchanged and ships in v1.** Consequence: Phase 0 no longer includes backup-ephemeral-auth or fencing property tests; it is a podman-primary-only validation. This also means **v1 has no failover safety net** — if the primary host is down, trading stops until it recovers (§19 #1 risk, now more acute). The CEO decision to add the backup in v2 is the mitigation.
2. **D13 — `uv` is the project manager.** Not just a dep resolver: `uv` manages the venv, the lockfile (`uv.lock` committed), running scripts (`uv run`), the workspace, and tool installs (`uv tool install ruff/mypy`). `pyproject.toml` is `uv`-managed. Replaces any `pip`/`poetry`/`pip-tools` mention. CI uses `uv sync --frozen && uv run pytest`.
3. **D14 — `polars` is the primary DataFrame engine, with lazy-eval discipline.** `polars` replaces `pandas` on performance-critical paths: bar ingestion and parquet I/O (`scan_parquet`/`write_parquet`), the backtest bar driver and vectorized signal/ROC/term-structure math, the IV proxy chain processing, and reconciliation joins. `pandas` is retained **only at interop boundaries** — yfinance returns pandas, `quantstats` expects a pandas return series, and `ib_async` DataFrame-returning APIs — converted to/from polars at the boundary with `pl.from_pandas()`/`.to_pandas()`. **Lazy-evaluation discipline (flagged by the CEO as tricky):** (a) use `LazyFrame` (`scan_parquet` / `.lazy()`) for large multi-symbol historical pipelines so predicate pushdown and projection pushdown skip unneeded rows/columns; (b) `.collect()` at well-defined strategy boundaries — never let a `LazyFrame` leak across module APIs or into a pre-trade check (eager there for predictability); (c) be explicit about eager-vs-lazy in type signatures — a `pl.DataFrame` and `pl.LazyFrame` are not interchangeable and silent auto-conversion is a footgun; (d) watch for lazy semantics that differ from eager/pandas — filter-pushdown over partitioned parquet, join order, null propagation in `over()` windows, and `maintain_order` defaults; (e) small config/state tables stay eager (no benefit, more readable); (f) tests assert the lazy pipeline's collected output equals an eager reference on a small slice (guard against pushdown surprises). A `data/polars_utils.py` helper centralizes the lazy/eager boundary conventions.

> **What changed in revision 5 (correction, CEO comment `a59e21cc`, 2026-07-18):** the prior revisions wrongly assumed the project could reuse code, OCI Functions images, EDGAR tooling, a backtest harness, Terraform, and CI from an `existing personal-strat repo`. That assumption was false — it came from reading unrelated repositories. The real project repo is **`personal-strat-pai`** (`https://github.com/rbystrit/personal-strat-pai.git`, local dir `~/src/personal-strat-pai`), and it is **empty — a clean greenfield**. Revision 5 removes every such assumption: §4 now treats the repo as greenfield (all modules, the `runtime/` container, the `functions/` packaging, the `infra/` OCI-as-code, and CI are authored fresh); §5 drops `matches existing repo` / `reuse current pattern` language; §9/§10/§14/§18 D9 drop `reusing the existing repo's EDGAR tooling` / `existing backtest harness` claims. **No architectural decision (D1–D11) changes** — only the incorrect `reuse existing X` provenance claims are corrected. The Python package is renamed `personal_strat_pai` to match the repo and avoid confusion with the unrelated `personal_strat` repos.

> **What changed in revision 4 (D11):** the deployment model is no longer "OCI Functions run everything." Per D11, **the live trading workload (Risk Clock, month-end sieve, IBKR execution, corp actions, reconciliation, SEC audit, IV proxy build) runs primarily in a single podman container on the current machine**, with **OCI Functions retained as a warm-standby backup-execution path** that takes over only when the primary is unavailable. **OCI stays exactly as planned in revision 3 for durable storage** — Object Storage (parquet), Oracle NoSQL (system of record), OCI Vault (secrets), OCI Logging/Monitoring/Alarms. The biggest consequence: the **§7 IBKR headless-auth problem — the #1 risk of revision 3 — is largely retired for the primary path**, because the IBKR Gateway now runs as a **persistent process inside the podman container** on a machine with a browser, so the periodic forced re-auth can be handled interactively. The ephemeral in-function Gateway becomes a **backup-only** path where the auth limitation is tolerable for short failovers. New risks introduced by D11 — single-host reliability, split-brain between primary and backup, and NoSQL-on-the-hot-path latency — are addressed in §3.3 (fencing/lease), §7.4 (failover), and §19.

> **What changed in revision 3 (preserved for history):** D1 moved the system to OCI Functions (x86) serverless; D5 moved state to Oracle NoSQL; D2 moved options to a self-built IV proxy; D3 moved rates to databento; D9 deferred the full SEC N-CEN/N-PORT/485BPOS engine to v4 with a static whitelist for v1–v3; D7 set parameters to the most conservative end of the brief's bands and parametrized them; D8 made the $10,000 monthly cash injection literal; D10 confirmed the paper→live gate; D6 deferred the dynamic intra-bucket sieve to v2. Revision 4 keeps all of D1–D10 except the "Functions run everything" reading of D1, which D11 refines.

> **v1 scope reminder (revision 6, D12):** everything above describes the **forward-looking** architecture. For **v1** specifically, only the **podman-primary execution path** and the **OCI durable-storage plane** are built. The OCI Functions backup-execution path, the `execution_lease` fencing, and the backup failover policy are **deferred to v2**. Read every "backup" mention below as "designed for v2; not present in v1" unless a section explicitly says otherwise.

---

## 1. System overview

A single, long-only ETF rotation strategy ("Flat Momentum Strategy") that:
- Rebalances monthly at the close on the last trading day of the month, deploying the cash pool ($10,000 monthly fresh cash injection + 100% intra-month stopped-out cash + trend-failed liquidation proceeds − withdrawals) into the top-3 of 13 ETF "buckets" via a 4-layer sieve (options/fixed-income pre-filters → absolute regime voting → relative-strength ROC → tax-hurdle).
- Defends capital intra-month on a "Risk Clock" with IV-rank-scaled dynamic stops, sweeping stopped-out cash into a tax-equivalent-yield parking vehicle.
- Runs a triplet (A→B→C) state machine per bucket to bypass 30-day wash-sale rules, with global HIFO tax-lot accounting (38.8% ST / 23.8% LT hurdle environment).
- Performs a monthly SEC compliance pre-flight. **v1–v3: static pre-verified whitelist** (PIMCO-free, ≥15-issuer, verified offline). **Full N-CEN/N-PORT/485BPOS XML engine deferred to v4** (D9).
- Adjusts the ledger overnight for splits/corporate actions before the next open.

The engineering stack has six planes: **data**, **state**, **strategy**, **execution**, **risk**, **observability**, plus a **backtester** that reuses the strategy and risk planes on historical data so live and backtest share one code path. **In v1 all planes run in a single podman container on the current machine — one execution path, no split-brain. The OCI Functions backup-execution path (§3) is deferred to v2 (D12); the durable-storage plane (Oracle NoSQL / Object Storage / Vault / Logging) ships in v1.**

## 2. High-level architecture (podman-primary + OCI backup-execution + OCI durable storage)

```
  v1 EXECUTION — current machine, ONE podman container (persistent, single execution path — no backup in v1)
  ┌──────────────────────────────────────────────────────────────────────┐
  │  One container (systemd-managed, --restart=always):                  │
  │   ├─ IBKR Gateway (Java, PERSISTENT in-container)                    │
  │   ├─ ib_async client  ◀──────▶ Gateway (localhost)                   │
  │   ├─ in-process scheduler (APScheduler/asyncio):                     │
  │   │   risk clock (sub-hourly), monthend, corpaction, reconcile,      │
  │   │   sec-audit, iv-proxy                                            │
  │   ├─ strategy + risk + exec + state modules (polars-first data layer)│
  │   └─ healthcheck (Gateway + scheduler liveness). No lease-renewer    │
  │      in v1 (single writer → no fencing needed; deferred to v2).      │
  └───────────────┬─────────────────────────────────────────────────────┘
                  │  read/write (write-through; SQLite read cache); polars scan_parquet over parquet
                  ▼
  DURABLE STORAGE & OBSERVABILITY — Oracle Cloud (Always Free + small PAYGO). v1 writes here from the single primary.
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Oracle NoSQL Database Cloud (system of record): tax_lots, positions, triplet_state, realized_pnl,      │
  │    sec_compliance, order_intent, risk_state, ibkr_session.                                            │
  │    [execution_lease deferred to v2 — single writer in v1 → no fencing needed.]                         │
  │  Object Storage (parquet lake): daily+minute bars, corp actions, IV proxy snapshots (polars-native).   │
  │  OCI Vault: IBKR creds, databento key, OCI/NoSQL creds, IBKR session-token material.                  │
  │  OCI Logging + Monitoring + Alarms + Email: structured logs, metrics, alerts.                         │
  │  [OCI Queue + Connector Hub: deferred to v2 with the backup functions.]                               │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        ▲                                   ▲                              ▲
        │                                   │                              │
   databento (bars, rates, EOD options)   yfinance (fallback)          SEC EDGAR (v4 compliance)

  v2 ADDS (deferred by D12, not in v1): BACKUP EXECUTION — Oracle Cloud, OCI Functions (x86), warm standby,
   same code, gated by the execution_lease; ephemeral in-function Gateway (session restore from Vault);
   OCI Resource Scheduler (hourly floor) + optional sub-hourly relay; OCI Queue + Connector Hub async glue.
   The §3.3 fencing, the execution_lease table, and the backup failover policy are (re-)introduced here.
```

**Key consequence of D11:** the IBKR Gateway is a **persistent process in the podman container on the current machine** — not an ephemeral in-function process. This retires the revision-3 §7.2 headless-re-auth risk for the primary path: the periodic forced re-auth (every ~7 days with "Bypass Enforced Re-auth", ~24h without) can be handled interactively on the current machine because a browser/display is available. Continuous price monitoring (the Risk Clock) still uses **databento** live/snapshot data (not IBKR market data), so the monitoring loop stays cheap and stateless; IBKR is touched only at trade events (month-end rebalance + intra-month stop exits) and nightly reconciliation — same as revision 3, but now from a persistent host. The **OCI Functions backup path keeps the ephemeral-Gateway design** (§7.3); its residual headless-auth limitation is acceptable because it only runs during short failovers while the primary is restored.

## 3. Deployment topology (podman-primary · OCI backup-execution · OCI durable storage)

**No OCI VMs of any kind.** **v1 compute is a single podman container on the current machine** (the user's current box — not an OCI VM, no hourly charge). **The OCI Functions backup-compute path is deferred to v2 (D12) — there is no OCI compute of any kind in v1.** Durable storage and observability are **Oracle Cloud services** (Always Free + small PAYGO overage pre-approved, D4) and are part of v1. This fits the issue's "Oracle Cloud PAYGO/free items only, nothing like VMs with hourly charge" constraint: in v1 the only Oracle footprint is storage/observability (Always Free + small PAYGO); the compute is off-Oracle on the existing machine. The per-invocation OCI Functions compute returns in v2 as the backup.

### 3.1 Component placement

| Component | v1 (podman primary — only path) | v2 backup (deferred, D12) | OCI storage/obs it uses |
|---|---|---|---|
| Risk Clock (sub-hourly OK) | podman, in-process scheduler | `risk-clock-fn` (hourly; relay for sub-hourly) | NoSQL `risk_state`, Object Storage bars |
| Month-end sieve | podman, scheduled | `monthend-fn` | NoSQL state, Object Storage bars/IV |
| IBKR order execution | podman, **persistent Gateway** | `exec-fn`, **ephemeral Gateway** (session restore) | NoSQL `order_intent`/`tax_lots`, Vault creds |
| Corp actions / splits | podman, scheduled (two runs) | `corpaction-fn` | Object Storage, NoSQL `tax_lots` |
| Nightly reconciliation | podman, scheduled | `reconcile-fn` | NoSQL, Object Storage |
| SEC audit | podman, monthly | `sec-audit-fn` | NoSQL `sec_compliance` |
| IV proxy build | podman, scheduled | `iv-proxy-fn` | Object Storage (chain + snapshots), NoSQL |
| Data ingestion (databento/yfinance) | podman, scheduled | (covered by the functions above) | Object Storage parquet, NoSQL |
| Scheduling | **APScheduler / asyncio loop** in container; systemd timers as a second-tier wake | **OCI Resource Scheduler** (hourly floor) | — |
| Fencing / leader election | **n/a in v1** — single writer, no split-brain possible | `execution_lease` NoSQL table (§3.3) | (v2) **Oracle NoSQL** |
| System of record | — (writes here) | — (writes here) | **Oracle NoSQL** |
| Bar / data lake | — (writes here, polars parquet I/O) | — (writes here) | **Object Storage** |
| Secrets | loaded from Vault at container start | loaded from Vault at function start | **OCI Vault** |
| Logs / metrics / alerts | ship to OCI Logging/Monitoring | ship to OCI Logging/Monitoring | **OCI Logging + Monitoring + Alarms** |

### 3.2 OCI service tiers (unchanged from revision 3 unless noted)

| OCI service | Tier / limits | Role |
|---|---|---|
| OCI Functions (x86) — backup | PAYGO, 2M invocations/mo free; sync ≤300s, detached ≤3600s | **Deferred to v2 (D12).** Not provisioned in v1. Backup-execution workloads (warm standby, lease-gated) return when the backup comes online. |
| Oracle NoSQL Database Cloud | Always Free (1 table, 25 GB) + small PAYGO | System of record (shared by both paths). **Replaces Autonomous DB (D5).** |
| Object Storage | Always Free 20 GB + small PAYGO overage (D4) | Parquet lake (bars, corp actions, IV snapshots). |
| OCI Vault | Always Free (20 keys) | All secrets, incl. IBKR session-token material. |
| OCI Resource Scheduler | Free; hourly floor | Triggers backup functions only. |
| OCI Queue + Connector Hub | Always Free quotas | **Deferred to v2 (D12)** with the backup functions. Not used in v1. |
| OCI Logging + Monitoring + Alarms + Email | Always Free quotas | Logs/metrics/alerts from **both** execution paths. |

### 3.3 Fencing & leader election — prevents split-brain (capital-safety critical)

> **v1 status (D12): DEFERRED TO v2.** In v1 there is **one execution path** (the podman primary), so two paths cannot reach the IBKR account simultaneously — split-brain is impossible by construction. The `execution_lease` table is **not created in v1**; the acquire/renew/check/release logic in `strategy/exec/lease.py` is **not wired in v1**; the lease-renewer healthcheck is replaced by a plain Gateway+scheduler liveness check. The design below is the **v2 spec**, re-introduced verbatim when the OCI Functions backup comes online. Keeping it in the plan preserves forward-compatibility and makes the v2 work a pure addition.

Because (in v2) two execution paths can reach the same IBKR account, **only the lease holder may submit orders.** This is the single most important control introduced by D11 for the backup path.

- **NoSQL `execution_lease` table** (key = `account`): `holder` (`primary`|`backup`), `holder_instance_id`, `generation` (monotonic int), `acquired_at`, `expires_at` (TTL), `last_renewed_at`.
- **Primary holds the lease** and renews it every 30 s with a TTL of 90 s while healthy. Every order submission and every scheduled trade event **reads the lease first**; if the holder is not `primary` with a live TTL, the primary **refuses to trade** and logs a fencing violation.
- **Backup functions check the lease before executing.** While the primary holds a live lease, backup `exec-fn` / `monthend-fn` / `risk-clock-fn` are **no-ops** (they observe and stand by). The backup acquires the lease only after the primary's TTL expires (≥ 90 s of missed renewals) — then and only then may the backup route orders.
- **On primary restart**, the container acquires the lease on boot before opening the trading loop; if the backup currently holds it (failover in progress), the primary waits and re-acquires after the backup's TTL or a manual handback.
- **Second layer of defense:** IBKR `client_order_id` idempotency + NoSQL `order_intent` unique-key writes (§11) ensure that even if a race occurs, duplicate orders are rejected at IBKR or de-duped against `order_intent`.
- **Failover policy (v1 default: manual).** For paper trading v1, failover to the backup is **manual** — the CEO or I arm the backup by explicitly releasing the primary's lease (a one-line CLI/config op). **Automatic failover is an explicit opt-in** gated behind a config flag and only enabled for live, after the fencing logic is proven in paper. This is the conservative choice consistent with "capital preservation first"; it is a **CEO decision** (D11-follow-up, §18) whether/when to enable auto-failover.

### 3.4 Current-machine host requirements (flagged for CEO confirmation)

> **v1 status (D12): the current machine is the *only* execution host in v1 — there is no backup safety net.** If it sleeps, loses network, or reboots during market hours, trading stops until it recovers; orders are not lost (state is in NoSQL + `order_intent` idempotency) but the month-end rebalance or a Risk-Clock stop exit can be missed. This makes §19 #1 (single-host SPOF) the **primary operational risk of v1**, and is the core reason the OCI Functions backup is the first v2 deliverable. The mitigations below assume the v2 backup exists; until then, the host-up expectation is stricter.

The current machine is the execution host, so it must meet:
- **Up during market hours** (09:30–16:00 ET) and at month-end close (~15:55 ET), plus the overnight corp-action windows (17:00 ET and 03:45 ET) and nightly reconcile (~16:30 ET). If the machine sleeps/powers down, trading stops until the backup is armed.
- **Reliable outbound network** to IBKR Gateway endpoints, databento, and OCI services. No inbound ports required (the IBKR Gateway binds localhost inside the container).
- **Java + podman** available (bundled in the container image; the host only needs podman).
- **A browser/display reachable for the ~7-day IBKR forced re-auth.** The current machine has this; an OCI Function does not — this is precisely why D11 retires the §7.2 risk for the primary.

Mitigations: `podman run --restart=always` + a systemd unit so the container (and the IBKR Gateway inside) survives host reboots; a host-level watchdog that pages if the container or the lease-renewer is unhealthy; the OCI backup is armed and tested so a primary outage is recoverable. **Residual single-host SPOF risk is flagged in §19**; the CEO should confirm the current machine is acceptable as the primary host or whether a dedicated always-on small box (still not an OCI hourly VM) is preferred — a CEO decision, not blocking this design.

## 4. Repository & module structure

Build inside the **`personal-strat-pai`** repo — `https://github.com/rbystrit/personal-strat-pai.git`, local dir `~/src/personal-strat-pai`. **This repo is empty — a clean greenfield.** There is no pre-existing OCI Functions image, EDGAR tooling, backtest harness, Terraform, or CI to reuse; **every module, container, function package, infra-as-code file, and CI workflow in this plan is authored fresh.** The Python package is **`personal_strat_pai`** (matches the repo; avoids confusion with the unrelated `personal_strat` package). **Revision 4 added a `runtime/` directory for the podman-primary container** (Containerfile, compose, systemd unit, healthcheck); revision 5 adds an `infra/` directory for OCI infra-as-code authored fresh.

> **Repo ground truth (revision 5):** the only allowed local directory is `~/src/personal-strat-pai` and the only allowed repository is `https://github.com/rbystrit/personal-strat-pai.git`. Both are empty as of this revision. Nothing in this plan assumes code, tooling, or infra from any other repo. If a later revision discovers an asset that genuinely exists in `personal-strat-pai`, it will be referenced explicitly; until then, `NEW — authored fresh` is the default provenance.

```
personal-strat-pai/                   # GREENFIELD repo (empty) — everything below is authored fresh in this plan
  pyproject.toml                     # uv-managed (D13): requires-python, deps, ruff/mypy/pytest config
  uv.lock                            # committed (D13); CI runs `uv sync --frozen`
  .python-version                    # 3.12
  README.md                          # quickstart: `uv sync`, `uv run pytest`, `uv run personal-strat-pai ...`
  src/personal_strat_pai/
    backtest/                         # NEW — built fresh; shares sieve/alloc/risk code path with live
      engine.py, costs.py, metrics.py, signals.py, data.py, portfolio.py, risk.py
    strategy/                         # NEW — the Flat Momentum Strategy
      config.py                       # pydantic-settings; Vault/env + NoSQL-backed config
      universe/
        buckets.py                    # 13-bucket × 3-ETF (A/B/C) config loader + whitelist
      sieve/
        options_smoke.py              # L1a — IV proxy term structure + 12m put skew (no OPRA)
        firf.py                       # L1b — fixed-income regime filter, 20% growth cap
        absolute_regime.py            # L2 — 3/6/9/12m voting (75%; crypto 66% on 1/2/3m)
        relative_strength.py          # L3 — Blended ROC = 0.5·ROC3M + 0.5·ROC6M, top-3
        tax_hurdle.py                 # L4 — profit-weighted ΔH (conservative: ST 8%, LT 4%)
        pipeline.py                   # month-end orchestrator (L1→L2→L3→L4)
        dynamic.py                    # OPTIONAL intra-bucket dynamic sieve — v2 only (D6)
      alloc/
        waterfall.py                  # 50/30/20 cash deployment
        nav_cap.py                    # 40% NAV cap (20% under FIRF) + overflow cascade
        beta_ceiling.py               # β_p ≤ 1.35, de-risk into lowest-beta passing assets
      risk/
        stops.py                      # IV-rank-scaled dynamic stops (5% / 8% / 15% — conservative)
        parking_lot.py                # TEY-based SUB/VTES/SGOV sweep, 5–7 day exception
        clock.py                      # Risk Clock logic (pure; scheduling is the runtime's job)
        limits.py                     # hard limits + kill switch
        reconcile.py                  # nightly vs IBKR executions/positions/cash
      exec/
        target_portfolio.py           # diff current → target orders
        ibkr_session.py               # Gateway lifecycle (persistent on primary; ephemeral on backup)
        router.py                     # pre-trade checks → ib_async order routing
        lease.py                      # NEW (D11) — execution_lease acquire/renew/check/release
      withdraw/
        waterfall.py                  # 6-tier tax-aware extraction
      compliance/
        whitelist.py                  # v1–v3: static pre-verified universe whitelist (D9)
        sec_audit.py                  # v4: N-CEN (PIMCO), N-PORT (≥15 issuers), 485BPOS listener
      state/
        nosql.py                      # Oracle NoSQL SDK access (tables, indexes, conditional writes)
        ledger.py                     # HIFO lots, positions, realized P&L
        triplet.py                    # A→B→C state machine + 60-day immunization windows
      data/
        databento.py                  # primary: historical + live bars + EOD options + SOFR/OIS (polars out)
        yfinance.py                   # fallback: daily bars, metadata, splits (cross-validated); pandas at the yfinance boundary -> pl.from_pandas
        store.py                      # parquet I/O via polars (scan_parquet/write_parquet) + local SQLite read cache
        polars_utils.py               # NEW (D14) — lazy/eager boundary helpers, schema defs, collect-at-boundary conventions
        corp_actions.py               # splits/dividends ingestion + ledger hooks
        iv_proxy.py                   # self-built IV from EOD chain snapshot (no OPRA) — D2; chain math in polars
        rates.py                      # SOFR / OIS curve — databento — D3
      observability/
        logging.py, metrics.py, alerts.py
    cli.py                            # entrypoints: run-backtest, sec-audit, reconcile, arm-backup, lease-show
  runtime/                            # NEW (D11) — podman-primary container
    Containerfile                     # image bundling Java + IBKR Gateway + the strategy code
    compose.yaml                      # podman-compose: container, volumes, restart=always, healthcheck
    personal-strat.service            # systemd unit (auto-start on boot, restart on failure)
    healthcheck.sh                    # lease-renewer + Gateway liveness + scheduler liveness
    README.md                         # runbook: start/stop, re-auth, failover, logs
  functions/                          # DEFERRED TO v2 (D12) — OCI Functions packaging for BACKUP execution.
    #   risk_clock/, monthend/, exec/, corpaction/, reconcile/, sec_audit/, iv_proxy/
    # (not created in v1; re-introduced verbatim when the backup comes online)
  infra/oci/                          # NEW — OCI infra-as-code for the v1 STORAGE plane only: Object Storage / NoSQL / Vault / Logging-Monitoring. (Functions infra added in v2.)
  config/
    universe.yaml, strategy.yaml, risk_limits.yaml   # all parameters parametrized (D7)
  tests/                              # pytest + hypothesis property tests
  docs/technical-design.md            # this doc, committed once approved
```

## 5. Technology stack

- **Language:** Python 3.12 (chosen; repo is greenfield — `requires-python` is set fresh in `pyproject.toml`).
- **IBKR client:** `ib_async` (maintained fork of ib_insync). Used with a **persistent Gateway on the primary** (podman) and an **ephemeral Gateway on the backup** (OCI Functions).
- **Market data:** `databento` (primary: historical + live bars, **EOD options chain snapshot for the IV proxy**, **SOFR/OIS curve**), `yfinance` (fallback).
- **Persistence:** **Oracle NoSQL Database Cloud SDK** (`oci-nosqldb`) for the system of record; **`sqlite3`** for local read-through caches (SEC whitelist, conId map, recent bars); **polars** for parquet I/O over Object Storage (lazy `scan_parquet` with pushdown). **No oracledb / SQLAlchemy** (D5 replaces Autonomous DB).
- **Numerics (D14, polars-first):** **`polars`** is the primary DataFrame engine — bar ingestion, parquet I/O, the backtest bar driver, vectorized ROC/term-structure/IV-proxy math, and reconciliation joins are all polars. `numpy` stays for scalar math and `scipy` for Black-Scholes in the IV proxy. `pyarrow` is the parquet on-disk format (polars reads/writes it natively). **`pandas` is retained only at interop boundaries** — yfinance returns pandas, `quantstats` expects a pandas return series, and `ib_async` DataFrame-returning APIs — with `pl.from_pandas()`/`.to_pandas()` conversions at the boundary. **Lazy-eval discipline:** `LazyFrame` + `scan_parquet` for large multi-symbol historical reads (predicate/projection pushdown); `.collect()` at strategy-module boundaries (never leak a `LazyFrame` across an API or into a pre-trade check); eager for small config/state tables; tests assert collected lazy output equals an eager reference on a small slice. Conventions centralized in `data/polars_utils.py`.
- **Config/validation:** `pydantic` v2 + `pydantic-settings`.
- **Scheduling (primary):** **APScheduler** or an **asyncio task loop** in the podman container — sub-hourly monitoring is trivial now that a persistent process exists. **Scheduling (backup):** **OCI Resource Scheduler** (hourly floor) + in-function relay for sub-hourly when on backup.
- **Container/runtime (primary, D11):** **podman** + **podman-compose**; **systemd** unit for boot-time start and restart-on-failure; container image bundles **Java + IBKR Client Portal Gateway** + the strategy code.
- **Function runtime (backup):** a Docker image authored fresh in `personal-strat-pai` + the OCI Functions `fdk` wrapper; `FN_NAME` dispatch (a pattern defined in this repo).
- **Logging/metrics:** `structlog` → OCI Logging (both paths); OCI Monitoring push.
- **Reports:** `quantstats` + custom pandas for tax-drag attribution.
- **Env/tooling (D13, uv is the project manager):** **`uv`** manages the venv, the lockfile (`uv.lock` committed), running scripts (`uv run`), the workspace, and dev tools (`uv tool install ruff/mypy`). `pyproject.toml` is `uv`-managed; no `pip`/`poetry`/`pip-tools`/`requirements.txt`. CI runs `uv sync --frozen && uv run pytest`. `pytest` + `pytest-asyncio` + `hypothesis`, `ruff` + `mypy` (invoked via `uv run`).
- **CI:** GitHub Actions (set up fresh in this repo — PR-level lint+typecheck+tests on every PR; merge requires green; optional push-to-main deploy for the backup functions).

## 6. Data pipeline

### 6.1 Historical + live bars (databento primary, yfinance fallback)
- **databento** `DBEQ` (US equities + ETFs, EOD + intraday) for daily and minute bars across the ~45 ETF universe. Live via `LiveClient` for the Risk Clock; batch `timeseries.get_range` for backfill.
- **yfinance** fallback: daily bars + metadata when databento is down or for a quick sanity cross-check. yfinance split/dividend data is occasionally wrong, so **every yfinance corp action is cross-validated against databento before it touches the ledger**.
- Bars land as **parquet** partitioned by `symbol/year` in **Object Storage**, written and read with **polars** (`write_parquet` eager at ingest; `scan_parquet` lazy on read so predicate pushdown skips unneeded symbols/dates and projection pushdown skips unneeded columns). A **local SQLite read cache** (in the podman container) holds the most-recent rolling window (~ last 260 trading days for 200D SMA / 12m ROC) for sub-minute Risk-Clock reads. NoSQL writes are write-through; hot reads hit SQLite; historical backtest reads go straight to parquet via `scan_parquet`. **Lazy discipline:** the multi-symbol historical scan stays a `LazyFrame` through signal/ROC/term-structure computation and is `.collect()`-ed only at the strategy-decision boundary; pre-trade checks use eager frames for predictability.

### 6.2 Options — Layer 1a (self-built IV proxy, no OPRA) — D2 resolved
- **No OPRA spend.** Build IV ourselves from a **narrow end-of-day options chain snapshot** (databento EOD options dataset, a fraction of OPRA cost) for the ~45 ETF universe: 30-day and 90-day at-the-money plus a small put-skew sample.
- Compute IV from the chain with Black-Scholes (`scipy`); persist a per-bucket daily IV snapshot (term structure + put-skew percentile) to Object Storage + NoSQL so the 12m percentile is a rolling-window query, not a recompute.
- **Accuracy tradeoff, flagged:** a self-built IV proxy from EOD snapshots is less precise than live OPRA. The smoke detector's edge partially depends on IV accuracy. **Phase 0 validation:** quantify proxy-vs-OPRA divergence on a sample window; if divergence materially changes smoke-detector verdicts, escalate to CEO to reconsider D2 (OPRA is the higher-accuracy option). The interface (`iv_proxy.py`) is parameterized so the source can be swapped to OPRA later without touching the sieve.

### 6.3 Fixed-income regime — Layer 1b (FIRF) — D3 resolved
- SOFR / OIS swap curve slope (2y vs 10y) + real rates. Source: **databento** (SOFR/OIS forward curve). FRED kept only as a free sanity cross-check, not the system of record.

### 6.4 Corporate actions / splits (overnight routine, brief §7)
- Primary: **databento reference / corporate-actions** dataset. Fallback: yfinance metadata. The overnight routine ingests next-day splits and recalibrates ledger lots (`qty × ratio`, `basis ÷ ratio`) before 04:00 ET so the Risk Clock never misreads a split gap as a stop breach. On the primary this is two scheduled runs in the container; on backup, two `corpaction-fn` invocations.

### 6.5 Data quality
- Every bar batch is checked: monotonic timestamps, no null OHLC, volume ≥ 0, price within sane bounds, split-adjustment continuity. Failures block the consuming job and page via alerts — no silent data errors (priority #2).

## 7. IBKR integration layer — persistent Gateway on the podman primary (v1); ephemeral backup deferred to v2 (D12)

> **v1 status (D12): only §7.1, §7.2, §7.4 (primary path) are built in v1.** §7.3 (backup ephemeral Gateway) is **deferred to v2** along with the OCI Functions packaging. With a single execution path, the lease check in §7.1 step 2 is a **no-op in v1** (always passes); it is wired in v2 when the backup arrives.

**The IBKR Gateway (Client Portal Web API, Java) runs as a persistent process inside the podman container on the current machine.** Because the current machine has a browser/display, the periodic forced re-auth can be handled interactively — the case that was unsolvable in a serverless function. This retires the revision-3 §7.2 "core design risk" for v1. The **backup OCI Functions ephemeral-Gateway path (§7.3) is the v2 work.**

### 7.1 Primary execution flow (per trade event, podman)
1. The in-process scheduler (APScheduler/asyncio) fires `monthend` or `risk-clock` and decides trades are needed.
2. `exec/lease.py` checks the `execution_lease` (§3.3); the primary must hold a live lease before proceeding. **(v1: this check is a no-op — single writer, no lease table yet; wired in v2.)**
3. The persistent `ib_async.Ib` connection (Gateway already running in-container) qualifies the small static universe (~45 tickers; conIds cached in NoSQL) and routes orders with `client_order_id` idempotency (§11).
4. Waits for fills, tracks partials, writes fills/lots/positions to NoSQL, advances triplet state.
5. Reconciles against NoSQL; Gateway stays up (no teardown on primary).

### 7.2 Primary session & authentication (largely solved by D11)
- **One-time interactive OAuth** at first start (browser on the current machine). Then enable **"Bypass Enforced Re-authentication"** so the gateway resumes for up to 7 days (paper) before forcing re-auth.
- **Session material persisted** to OCI Vault (or NoSQL `ibkr_session`) so the container can restore after a restart without re-OAuth.
- **Forced periodic re-auth (every ~7 days, or ~24h without bypass):** the container surfaces a re-auth-required alert (OCI Alarm + console/structured log) and a desktop notification on the current machine; a human (CEO or me) completes the interactive re-auth in the browser. This is the previously-unsolvable case, now solvable because the primary runs on a machine with a browser. **Operational runbook in `runtime/README.md`.**
- **IBKR OAuth for the Client Portal REST API** (long-lived token, no interactive re-auth) remains the preferred hardening if IBKR enables it for the account — note in Phase 0; not required for v1 given D11.

### 7.3 Backup execution flow (OCI Functions, lease-gated) — **DEFERRED TO v2 (D12)**

> Not built in v1. Retained as the v2 spec. Re-introduced verbatim when the OCI Functions backup comes online; until then the primary is the only path and this section does not execute.
1. A scheduled backup function decides trades are needed **and** the primary's `execution_lease` has expired (TTL > 90 s with no renewal). If the primary still holds the lease, the backup is a no-op.
2. `exec-fn` starts the headless IBKR Gateway in-container with a saved session (restored from Vault), waits for `authenticated`/`connected`.
3. Connects via `ib_async`, routes orders with `client_order_id` idempotency, waits for fills within the detached (≤3600s) window, tracks partials.
4. Reconciles against NoSQL, writes fills/lots/positions, tears down the Gateway, exits.
5. **Known backup limitation:** if a forced re-auth is required while on backup, the function cannot provide a browser. Mitigation: failover windows are expected to be short (primary restoration, not weeks); if re-auth is required on backup, `exec-fn` alerts and **refuses to trade** until the primary is restored or the CEO manually re-auths and refreshes the Vault session. Acceptable for a backup; flagged in §19.

### 7.4 Connection lifecycle (`strategy/exec/ibkr_session.py`)
- **Primary:** `start_gateway()` runs once at container boot (spawn Java gateway, poll `/v1/api/iserver/auth/status` until authenticated); the `ib_async.Ib` connection is kept alive; `disconnectedEvent` triggers in-process exponential-backoff reconnect; the lease-renewer (§3.3) runs as a background task and also serves as the liveness healthcheck. `stop_gateway()` only on container shutdown.
- **Backup:** `start_gateway()` per `exec-fn` invocation; `stop_gateway()` on exit (always, including error paths); if the detached window is expiring, persist state to NoSQL and exit cleanly (a follow-up `exec-fn` resumes from NoSQL).
- **Contract resolution:** ETF contracts qualified once per session via `qualifyContracts`; conIds cached in NoSQL. Universe is small and static.
- **Market data from IBKR is NOT used.** Live prices come from databento (§6.1) on both paths. IBKR `reqMktData`/`reqHistoricalBars` avoided.
- **Account/portfolio:** `reqAccountSummary` + `reqPositions` pulled per trade event into the local account cache used by pre-trade checks and reconciliation; not streamed.

## 8. State & persistence model — Oracle NoSQL (D5)

System of record = **Oracle NoSQL Database Cloud Service**, accessed via the `oci-nosqldb` SDK. Local SQLite is a read-through cache + scratchpad for regenerable data (compliance whitelist, conId map, recent bars). **No Autonomous DB, no oracledb, no SQLAlchemy.** Both the primary and backup paths read/write the same NoSQL tables — NoSQL conditional writes are the concurrency/atomicity boundary.

NoSQL tables (logical model — each is a keyed table with JSON-ish rows):

| Table | Key | Fields |
|---|---|---|
| `tax_lots` | `lot_id` | account, bucket_id, ticker, triplet_slot (A/B/C), qty, cost_basis_per_share, acquired_at, realized_st_gain, realized_lt_gain, status (open/closed/washed), wash_immunity_until |
| `positions` | `account,bucket_id,ticker` | qty, avg_cost, market_value, unrealized_st_gain, unrealized_lt_gain, updated_at |
| `triplet_state` | `bucket_id` | current_slot, last_loss_at, immunized_until, slot_history (list) |
| `realized_pnl` | `lot_id` | closed_at, proceeds, cost, gain, holding_days, st_lt |
| `sec_compliance` | `ticker,month` | pimco_blacklisted (bool), issuer_count, diversified (bool), source (`whitelist` v1–v3 / `edgar` v4), checked_at |
| `order_intent` | `client_order_id` | bucket_id, ticker, side, qty, intent_hash, status, created_at, filled_qty, avg_fill_price |
| `risk_state` | `account` | kill_switch (bool), nav_cap, beta_ceiling, dd_stop_account, updated_at |
| `ibkr_session` | `account` | session_blob_ref (Vault secret ref), issued_at, expires_at, auth_method |
| `execution_lease` (D11) — **v2, deferred (D12)** | `account` | holder (`primary`/`backup`), holder_instance_id, generation, acquired_at, expires_at, last_renewed_at. **Not created in v1** — single writer makes fencing unnecessary. Re-introduced with the OCI Functions backup. |

**HIFO (`strategy/state/ledger.py`):** on a sell, lots are selected highest-cost-first to maximize harvested losses; realized gain/loss split into ST (<365d) vs LT (≥365d) at the lot level. Pure functions, property-tested with `hypothesis` (invariants: qty conservation, proceeds = Σ per-lot proceeds, no lot double-counted). NoSQL stores the lots; HIFO selection is in-Python.

**Triplet state machine (`strategy/state/triplet.py`):** on a loss liquidation in bucket *B*, advance `current_slot` B→C and set `immunized_until = now + 60d`; the slot just exited becomes RESTRICTED for 30 days (wash-sale). Transitions are append-only (`slot_history`) and reversible only by an explicit, logged admin op. NoSQL conditional writes enforce transition atomicity.

**Execution lease (`strategy/exec/lease.py`, D11) — v2 only (D12):** acquire/renew/check/release with NoSQL conditional writes on `(holder, expires_at, generation)`. Renewal is a conditional update that only succeeds if the caller is still the holder; acquisition by the backup only succeeds if `expires_at < now`. This is the fencing boundary that prevents split-brain (§3.3) once the backup exists. **In v1 the module is stubbed (no-op) and the table is not created** — single writer, no split-brain possible.

## 9. Strategy engine — mapping to the 9 brief sections

| Brief § | Module | Notes |
|---|---|---|
| 1 Tax & structural | `state/ledger.py`, `state/triplet.py` | HIFO + triplet machine are core. 38.8% ST / 23.8% LT hurdle env. |
| 2 13-bucket universe | `universe/buckets.py`, `config/universe.yaml`, `compliance/whitelist.py` | Static 13×3 matrix. **v1–v3: static pre-verified whitelist** (PIMCO-free, ≥15 issuers). Full N-CEN/N-PORT/485BPOS engine **deferred to v4** (D9). |
| 3 Month-end sieve | `sieve/pipeline.py` + L1a/L1b/L2/L3/L4 | Orchestrator runs in the primary container at last-trading-day ~15:55 ET. (Backup `monthend-fn` deferred to v2, D12.) |
| 4 Allocation & caps | `alloc/waterfall.py`, `nav_cap.py`, `beta_ceiling.py` | 50/30/20; 40% cap (20% FIRF); β_p ≤ 1.35 with de-risk into lowest-beta passing asset. |
| 5 Real-time risk | `risk/stops.py`, `risk/parking_lot.py`, `risk/clock.py` | IV-rank→stop band; TEY parking sweep with 5–7 day late-month exception. Monitoring via databento. |
| 6 Withdrawal waterfall | `withdraw/waterfall.py` | 6-tier priority queue; planned = month-end, emergency = parking-only + early sieve. |
| 7 Corp actions | `data/corp_actions.py`, scheduled runs / `corpaction-fn` | Split recalibration, two scheduled runs covering 17:00–04:00 ET. |
| 8 Intra-bucket dynamic sieve | `sieve/dynamic.py` | **Deferred to v2** (D6). v1 uses the static triplet matrix. |
| 9 SEC compliance audit | `compliance/whitelist.py` (v1–v3), `compliance/sec_audit.py` (v4) | **v4: full N-CEN/N-PORT/485BPOS XML parse** via an EDGAR tooling module authored fresh in this repo. v1–v3: static whitelist. |

### 9.1 Sieve pipeline pseudocode (month-end close)
```
for bucket in UNIVERSE:
    if options_smoke.blocked(bucket):          # L1a — IV proxy backwardation or extreme put skew
        continue
candidates = [b for b in UNIVERSE if firf.allowed(b)]   # L1b may cap growth/duration to 20%
candidates = [b for b in candidates if absolute_regime.passes(b)]  # L2 — 75% (crypto 66%)
candidates.sort(by=blended_roc, desc=True)    # L3 — 0.5·ROC3M + 0.5·ROC6M
top3 = candidates[:3]
# L4 — for held positions that fell to ranks 5–13:
for held in held_positions:
    if cost_basis < market: liquidate(held)           # harvest loss
    elif blended_roc(top1) - blended_roc(held) > delta_H_blended(held): liquidate(held)
    else: amnesty_holdover(held)                      # keep, no new cash
targets = alloc_waterfall(top3, cash_pool)            # 50/30/20  (cash_pool incl. $10K monthly)
targets = nav_cap(targets, portfolio)                 # 40% / 20% overflow cascade
targets = beta_ceiling(targets, portfolio, cap=1.35)  # de-risk into low-beta
```

### 9.2 Configurable parameters — conservative defaults, all parametrized (D7 resolved)
All in `config/strategy.yaml`, CEO-editable. **Defaults are the most conservative end of each band** (tightest stops, highest hurdles → most capital protection, least turnover/whipsaw).

| Parameter | Band | **Default (conservative)** | Rationale |
|---|---|---|---|
| ΔH — short-term | 6–8% | **8%** | Highest hurdle → fewest replacement trades |
| ΔH — long-term | 3–4% | **4%** | Highest hurdle → fewest replacement trades |
| Stop — IV rank <50 | 5% | **5%** | Tightest stop → most capital protection |
| Stop — IV rank ≥50 | 7–8% | **8%** | Tighter end of the higher-vol band |
| Stop — crypto | 12–15% or 3×ATR | **15%** | Tighter of the two forms |
| Blended ROC weights | given | 0.5 / 0.5 | per brief |
| Absolute regime windows | given | 3/6/9/12m ≥3/4 (sectors); 1/2/3m ≥2/3 (crypto) | per brief |
| NAV cap | given | 40% (20% FIRF) | per brief |
| Beta ceiling | given | 1.35 | per brief |

Everything is parametrized so values can be tuned without code changes; live tuning of any of these is a CEO-approved commit.

## 10. Backtesting framework

One code path shared with live: `backtest/engine.py` calls the same `sieve/`, `alloc/`, `risk/` modules on historical bars instead of live feeds. **This is the single biggest correctness lever — no "backtest-only" strategy logic drift.** The backtest harness is **built fresh** in this plan (`backtest/` under `src/personal_strat_pai/`); it is authored alongside the strategy so live and backtest share one code path from day one — there is no pre-existing harness to extend or fork. The backtester runs anywhere (dev shell, CI, or the podman container) and reads the same Object Storage parquet via **polars `scan_parquet`** — it does not depend on the live IBKR session. **Performance (D14):** the bar driver and all vectorized signal math (ROC, term-structure, IV-rank percentile, beta) are **polars `LazyFrame` pipelines** that stay lazy through the sieve and are `.collect()`-ed only at the per-day decision boundary; this is the main payoff of polars for the ~45-ETF × multi-year daily+intraday dataset. `pandas` appears only at the `quantstats` report boundary.

- **Bar driver:** event loop over daily bars (decision frequency) with optional intraday bar overlays for stop simulation on the Risk Clock. Month-end decisions use daily closes; stops are checked against intraday highs/low if available, else daily. **Bar reads are polars `scan_parquet` (lazy)** with date/symbol predicate pushdown; the per-day slice is `.collect()`-ed into an eager frame before the sieve runs so pre-trade checks see predictable eager data.
- **Costs (`backtest/costs.py`):** IBKR commissions (tiered $0.0035/share min $0.35, configurable), slippage model (half-spread + square-root impact), no borrow (long-only), margin = cash (no leverage; intraday vs overnight margin moot for unlevered long ETFs).
- **Constraints enforced in backtest:** NAV cap, beta ceiling, kill switch, drawdown stops — same code as live.
- **Reproducibility:** each run snapshots (databento dataset+range, config hash, code git sha, parquet data hash) into a `run_manifest.json`; reports reference the manifest.
- **Reports:** trade-level + aggregate — total/period P&L, per-bucket and per-lot P&L, Sharpe, Sortino, max drawdown, turnover, exposure over time, portfolio beta, **tax-drag attribution** (ST/LT gains realized, losses harvested, wash-sale bypass events count), SEC compliance pass/fail per month. Output: HTML (quantstats) + parquet of trades + JSON metrics.
- **Validation gate:** before any live paper run, the backtester must reproduce a hand-checked scenario on a small universe and reconcile to the penny against an independently computed P&L.

## 11. Execution & order management

- **Target portfolio (`exec/target_portfolio.py`):** diff current positions vs sieve targets → order list (sells first to free cash, then buys).
- **Pre-trade checks (`exec/router.py`), all hard-gated:** lease held by this path (§3.3) **(v1: no-op, single writer — wired in v2)**; kill-switch off; within market hours; position limit per bucket; NAV cap respected post-fill; beta ceiling respected post-fill; cash available; wash-sale lock not triggered for the ticker's triplet slot; order not duplicate vs `order_intent` (idempotency).
- **Idempotency:** every order gets a `client_order_id` derived from `(bucket_id, ticker, side, intent_hash, run_id)`. The intent row is written to NoSQL `order_intent` **before** submit. Retries look up by `client_order_id` instead of re-submitting. This prevents duplicate orders across primary/backup or restarts.
- **Partial fills:** tracked via `execDetailsEvent` / `commissionReportEvent`; open qty updated incrementally in NoSQL; a partial fill never re-fires the full order. On primary, the persistent connection rides out partials. (Backup `exec-fn` resume-after-detached-window behavior deferred to v2, D12.)
- **Routing:** marketable limit (limit at bid/ask touch) to avoid crossing spreads aggressively; configurable. No market orders by default.
- **Reconciliation (`risk/reconcile.py` → scheduled run):** nightly (~16:30 ET), `reqExecutions`/`reqTrades` from IBKR vs `order_intent`/`tax_lots` in NoSQL; any gap pages and blocks next session's new orders until resolved. (Backup `reconcile-fn` deferred to v2, D12.)

## 12. Risk controls architecture

Capital preservation is priority #1. All limits live in `config/risk_limits.yaml`, versioned, and changes require an explicit CEO-approved commit (logged).

- **Hard position/exposure limits:** per-bucket max weight (≤40%, ≤20% under FIRF), per-ticker max shares (CEO-set — see §18), max gross exposure = NAV (long-only, unlevered).
- **Portfolio beta ceiling:** β_p ≤ 1.35 enforced pre-trade and re-checked post-fill.
- **Drawdown stops:** per-strategy and account-level; on breach → liquidate per CEO policy (configurable: flat-all vs hold-and-block-new) + alert. **Account-level DD stop %: CEO-set — placeholder until confirmed (§18).**
- **Kill switch:** a `risk_state.kill_switch` flag (Vault/env/NoSQL) that, when armed, blocks all new orders and (policy-configurable) liquidates. Armable by CEO or by automated breach detectors. Honored on both primary and backup.
- **Execution lease fencing (D11, §3.3) — v2 only (D12):** only the lease holder may place orders — prevents split-brain between the podman primary and the OCI backup from double-executing against the same account. **In v1 there is a single execution path, so this control is not wired and the `execution_lease` table is not created; split-brain is impossible by construction.** Re-introduced with the v2 backup.
- **Daily reconciliation:** positions/cash/NAV vs IBKR; mismatch > tolerance pages and freezes new orders.
- **Pre-trade vs post-trade:** limits checked pre-trade (reject) and re-verified post-fill (auto-de-risk if a fill somehow breaches).
- **No live capital** until paper sign-off + risk review (priority #1, see §17).

## 13. Observability

- **Logging:** `structlog` JSON to OCI Logging from **both** the primary container and the backup functions. Every order, sieve decision, stop event, reconciliation, IBKR session start/stop, lease acquire/renew/fail, and state-machine transition is logged with a correlation id.
- **Metrics (OCI Monitoring):** P&L (realized/unrealized, per-bucket), exposure, portfolio beta, cash-pool size, slippage-vs-arrival, fill latency, order-rejection rate, reconciliation-gap count, IBKR session start/fail count + re-auth-required events, primary container liveness, data-quality failure count. **(v2 metrics, deferred: lease state/holder/TTL/renew-failures, backup function cold-start time.)**
- **Alerts (OCI Alarms → email/webhook):** risk-limit breach, kill-switch armed, DD stop hit, reconciliation gap, IBKR session auth failure / **re-auth required (primary)**, primary container down, data-quality failure, SEC whitelist blacklist hit on a held ticker. **(v2 alerts, deferred: lease fencing violation, lease-renew missed, function invocation failure.)**
- **Execution quality:** slippage vs arrival price, implementation shortfall, reported per order and aggregated monthly.
- **P&L attribution:** monthly report decomposing return into bucket selection, allocation waterfall, tax-loss harvesting, parking-lot yield, and execution slippage.

## 14. Scheduling & the "Risk Clock"

**v1 (podman primary, only path):** an **in-process scheduler** (APScheduler or an asyncio task loop) — sub-hourly monitoring is trivial with a persistent process. **Backup scheduling (OCI Resource Scheduler hourly floor + sub-hourly relay) is deferred to v2 (D12).** Below, each bullet's "Backup:" note is the v2 spec.

- **Month-end sieve:** last trading day, ~15:55 ET (before the 16:00 close), then order routing at/after 16:00. Primary: scheduled in-container. (Backup `monthend-fn` deferred to v2, D12.)
- **Risk Clock:** **sub-hourly on primary** (e.g., every 1–5 min during 09:30–16:00 ET) — reads databento prices, checks each open position's stop against its IV-rank band; on breach, routes a liquidation via HIFO, advances triplet, sweeps to parking per the 5–7-day rule. **(Backup `risk-clock-fn` hourly + sub-hourly relay deferred to v2, D12; v1 has no backup, so the primary sub-hourly cadence is the only one.)**
- **Overnight corp-action routine:** two scheduled runs — ~17:00 ET (ingest next-day splits, recalibrate ledger) and ~03:45 ET (verify before open). Primary: in-container. (Backup `corpaction-fn` deferred to v2, D12.)
- **Nightly reconciliation:** ~16:30 ET after close; pulls executions/positions from IBKR, reconciles against NoSQL. Primary: in-container. (Backup `reconcile-fn` deferred to v2, D12.)
- **Monthly SEC audit:** scheduled 1st of month. **v1–v3: no-op beyond refreshing the static whitelist. v4: full EDGAR parse** via the EDGAR tooling module authored fresh in this repo (`compliance/sec_audit.py`).

Market-holiday/calendar awareness via `pandas_market_calendars` (XNYS). No orders outside sessions.

## 15. Security & secrets

- **Secrets in OCI Vault** (IBKR password/2FA path, databento key, OCI creds, NoSQL creds, **IBKR session-token material** per §7.2). Loaded at primary container start and at backup function start; never written to disk or logs. No account numbers in the repo or log output (masked).
- **Repo hygiene:** `.env.example` only contains non-secret keys; pre-commit hook + CI scan for secrets (`detect-secrets` / `gitleaks`).
- **IBKR least privilege:** paper account first; live account read+trade, no withdrawal API. IBKR credentials scoped to the trading user.
- **Network:** the primary container needs outbound egress to IBKR Gateway endpoints, databento, EDGAR, and OCI services; **no inbound** (the IBKR Gateway binds localhost inside the container). Backup functions run in OCI's managed network with the same egress, no inbound. No SSH/Bastion (no VM).

## 16. Testing strategy

- **Unit:** per-module pytest; pure functions for HIFO, triplet, ROC, TEY, delta_H, IV proxy, **polars lazy/eager boundary** (collected lazy == eager reference on a small slice). **(v2: lease acquire/renew/fence with concurrent-acquisition edge cases.)**
- **Property tests (hypothesis):** HIFO qty/proceeds conservation; triplet machine never allows re-entry inside immunization window; beta ceiling never exceeds cap post-de-risk; NoSQL conditional-write transition atomicity; **polars: a `scan_parquet`-based lazy pipeline's collected output equals an eager `read_parquet` reference on a sample slice (guards against pushdown/projection surprises — D14).** **(v2: lease — two concurrent acquirers never both hold; backup never trades while primary TTL is live.)**
- **Backtest validation:** reproduce a hand-computed scenario on a 3-bucket, 2-year slice; assert P&L to the cent.
- **Integration (paper) — the Phase 0 IBKR gate (§7.2) — v1 scope:**
  - **Primary (v1 must-pass):** from the podman container, start the persistent Gateway, authenticate (interactive one-time OAuth + bypass), place/confirm/cancel a 1-share paper order, reconcile. Prove the Gateway + scheduler liveness healthcheck works.
  - **Backup + Fencing (v2, deferred — D12):** `exec-fn` ephemeral headless auth + 1-share paper order within the detached window; simulate primary-lease expiry and prove the backup acquires the lease and trades while the primary stands down; simulate primary returning and prove it re-acquires and the backup stands down. **Not in v1.**
- **Reconciliation test:** inject a synthetic gap, assert the reconciler detects and freezes.
- **CI:** GitHub Actions (set up fresh) — ruff, mypy, pytest, backtest smoke on every PR; merge requires green.

## 17. Phased rollout (paper → live gate) — D10 confirmed

| Phase | Scope | Exit gate |
|---|---|---|
| 0 | Data pipeline (databento + IV proxy + rates, **polars-first**), NoSQL state/ledger/triplet (**no `execution_lease` table in v1**), backtester on historical data, **podman-primary container with persistent IBKR Gateway + liveness healthcheck**. **No backup in v1; no fencing.** No live capital. | Backtest reconciles vs hand-checked scenario; primary proves persistent IBKR auth + 1-share paper order; Gateway + scheduler liveness healthcheck passes; polars lazy-vs-eager property tests pass; reports render. |
| 0.5 (v2) | **OCI Functions backup-execution path + `execution_lease` fencing** (D12 lift): `functions/` packaging, ephemeral in-function Gateway, session restore from Vault, OCI Resource Scheduler wiring, backup failover policy, fencing property tests. | Backup proves ephemeral headless IBKR auth + 1-share paper order within the detached window; fencing test passes (no split-brain: backup acquires lease on primary TTL expiry, primary re-acquires on return); failover drill signed off. |
| 1 | IBKR paper: full loop (sieve → alloc → risk → exec → reconcile) on paper account with placeholder capital/limits, **running on the podman primary only (no backup in v1 — single-host risk acknowledged, §19 #1)**. CEO reviews. | **CEO paper-trading sign-off + risk-controls review + N clean paper reconciliation sessions (D10).** |
| 2 | Live, minimal capital, tight CEO-set limits, kill-switch tested. (Auto-failover to backup only relevant after v2 backup is built and signed off — D11-follow-up.) | CEO ongoing review; expand per CEO discretion. |

**No live capital moves before the Phase-1 exit gate.** Non-negotiable (priority #1, role boundary).

## 18. CEO decisions — resolved D1–D11

| # | Decision | **CEO answer** | How it's reflected |
|---|---|---|---|
| D1 | VM vs no-VM | **No OCI VMs of any kind. Compute: podman-primary (current machine) + OCI Functions backup (x86); OCI for durable storage.** (Refined by D11.) | §2, §3, §7, §14 — persistent Gateway in podman; ephemeral in backup functions; OCI storage unchanged. |
| D2 | Options data | **Self-built IV proxy; OPRA is expensive.** | §6.2 — IV from EOD chain snapshot; `iv_proxy.py` parameterized for later OPRA swap; accuracy tradeoff flagged for Phase 0. |
| D3 | SOFR/OIS source | **databento.** | §6.3 — databento for the forward curve; FRED only as cross-check. |
| D4 | Object Storage cap | **Small PAYGO overage is OK.** | §3.2 — 20 GB free + small PAYGO pre-approved. |
| D5 | Ledger system of record | **Oracle NoSQL Database Cloud Service.** | §8 — NoSQL replaces Autonomous DB; `state/nosql.py`; no oracledb/SQLAlchemy. |
| D6 | Dynamic intra-bucket sieve | **(OK) defer to v2.** | §9 — `sieve/dynamic.py` v2-only; v1 uses static triplet matrix. |
| D7 | Parameter exact values | **Most conservative within bands, parametrized.** | §9.2 — ΔH ST 8% / LT 4%; stops 5% / 8% / 15%; all in `config/strategy.yaml`, CEO-editable. |
| D8 | Capital & risk limits | **$10,000 monthly cash injection is literal.** | §1, §9.1 — $10K/mo in the cash pool. **Starting capital, account-level DD stop %, and per-ticker max shares are not in the brief and remain CEO-set** — placeholders in `config/risk_limits.yaml` marked `# CEO-SET`; to confirm before Phase 1 live capital (do not block Phase 0). |
| D9 | SEC compliance phasing | **Static pre-verified whitelist OK until v4.** | §9, §14 — `compliance/whitelist.py` for v1–v3; full N-CEN/N-PORT/485BPOS engine deferred to v4 (EDGAR tooling authored fresh in this repo). |
| D10 | Paper→live gate | **(OK) confirmed.** | §17 — CEO sign-off + risk-controls review + N clean paper reconciliation sessions. |
| D11 | Execution placement (revision 4) | **Primary = podman on the current machine; OCI Functions = backup execution only; OCI = durable data storage as planned.** (Refined by D12: backup deferred to v2.) | §2, §3 (incl. §3.3 fencing — v2, §3.4 host reqs), §4 (`runtime/`), §5, §7 (persistent primary in v1; ephemeral backup in v2), §8 (`execution_lease` — v2), §11, §12, §14, §17, §19. |

**D11 follow-ups needing CEO confirmation (not blocking this design):**
- **Failover policy:** manual (v1 paper default) vs automatic (opt-in for live). Proposed default: manual for paper, auto only after CEO sign-off. → §3.3. **(Moot in v1 — no backup; revisited at v2.)**
- **Current-machine as primary host:** confirm the current machine is acceptable (uptime during market hours, reliable network) or whether a dedicated always-on small box (still not an OCI hourly VM) is preferred. → §3.4, §19. **Stricter in v1: with no backup, host downtime = trading downtime.**
- **Lease TTL / heartbeat thresholds:** proposed 90 s TTL, 30 s renewal. CEO-adjustable. → §3.3. **(v2 only.)**

| D12 | OCI Functions backup timing (revision 6) | **Deferred out of v1 — v1 runs on the podman primary only, single execution path, no fencing. Backup + `execution_lease` + failover policy are the first v2 deliverable.** | §2 (diagram v2 note), §3.1–§3.3 (fencing deferred), §3.4 (stricter host req), §4 (`functions/` deferred), §7.3 deferred, §8 lease table/row deferred, §11/§12/§13/§14 backup refs marked v2, §16 fencing tests → v2, §17 Phase 0.5, §19 #1 risk sharpened, §20. |
| D13 | Project manager (revision 6) | **`uv`** — venv, lockfile (`uv.lock` committed), `uv run`, workspace, dev tools. No pip/poetry/pip-tools. CI: `uv sync --frozen && uv run pytest`. | §4 (`pyproject.toml` + `uv.lock`), §5 (env/tooling), §16 (CI). |
| D14 | Data layer engine (revision 6) | **`polars`-first with lazy-eval discipline; `pandas` only at interop boundaries (yfinance, quantstats, ib_async).** | §4 (`data/polars_utils.py`), §5 (numerics, persistence), §6.1 (scan_parquet), §10 (backtest lazy pipelines), §16 (lazy-vs-eager property test), §19 (lazy-eval correctness risk). |

**Remaining CEO-set values (not blocking this design; needed before Phase 1 live capital):**
- Starting capital (initial account balance).
- Account-level drawdown stop %.
- Per-ticker max shares.

These are config placeholders (`# CEO-SET`) for paper trading. I'll collect them in a single `ask_user_questions` interaction before Phase 1 — not now.

## 19. Risks & flags

- **⚠⚠ Single-host SPOF — v1 #1 operational risk (sharpened by D12).** In v1 the current machine is the **only** execution host and **there is no backup**. If it is down, asleep, or loses network during market hours or at month-end close, trading stops until it recovers — a month-end rebalance or a Risk-Clock stop exit can be **missed** (orders are not lost: state is in NoSQL + `order_intent` idempotency, but the opportunity/risk event passes). Mitigations in v1: `--restart=always` + systemd, a host watchdog, a Gateway+scheduler liveness healthcheck, and a CEO operational discipline to keep the machine up during market hours. **The v2 OCI Functions backup is the structural mitigation and is the first v2 deliverable.** CEO decision (§3.4): is the current machine acceptable as the sole v1 host? This replaces the retired §7.2 auth risk.
- **Split-brain / double execution — v1: not present (single execution path, D12).** Two paths can reach the IBKR account only once the v2 backup exists. **v1 mitigation: by construction** — one writer, no `execution_lease` table, no backup functions. The `client_order_id` idempotency + `order_intent` dedup remain as the second layer against restart-time duplicates. **v2 re-introduces this risk and the §3.3 fencing as a Phase 0.5 must-pass gate (property-tested, §16).**
- **NoSQL latency on the primary hot path.** The podman primary reads/writes NoSQL over the internet. Hot reads (recent bars, conId map, whitelist) are served from the local SQLite read cache; NoSQL writes are write-through. Stop checks read databento prices (cached) + a small NoSQL `positions`/`risk_state` read per cycle. Phase 0 measures the NoSQL round-trip from the current machine and flags if it materially affects the Risk Clock cadence. (Unchanged by D12; true in v1.)
- **IBKR headless auth on the backup path — v2 only (D12).** The backup `exec-fn` cannot do the periodic forced re-auth (no browser). Acceptable for short failovers; if re-auth is required on backup, `exec-fn` alerts and refuses to trade until the primary is restored or the session is manually refreshed (§7.3). **Not present in v1 (no backup).**
- **Hourly Risk Clock granularity on backup — v2 only (D12).** The backup scheduler floor is hourly; sub-hourly on backup needs the relay pattern. **Not present in v1 (no backup; primary is sub-hourly).**
- **IV proxy accuracy (D2).** Self-built IV from EOD snapshots is less precise than OPRA and could change smoke-detector verdicts. Phase 0 quantifies divergence; escalate to CEO if material.
- **Tax/regulatory ownership.** HIFO + triplet wash-sale bypass is an engineering implementation of the brief's approach. Tax/regulatory/compliance review is the CEO's domain (per role boundaries). I recommend an independent tax review before any live capital. Flagging, not deciding.
- **yfinance reliability.** Fallback only; corp actions cross-validated vs databento.
- **⚠ polars lazy-evaluation correctness (new with D14).** Lazy frames defer work and can change semantics in subtle ways: predicate/projection pushdown over partitioned parquet, join order, null propagation in `over()` windows, and `maintain_order` defaults differ from eager/pandas. A silent auto-conversion between `pl.DataFrame` and `pl.LazyFrame` across a module boundary is a footgun. Mitigations: `data/polars_utils.py` centralizes the eager/lazy boundary; `.collect()` at strategy-decision boundaries (pre-trade checks always eager); type signatures explicit; a §16 property test asserts `scan_parquet`-based lazy output equals an eager `read_parquet` reference on a sample slice. Flagged because the CEO explicitly called this out as tricky.
- **⚠ v1 has no failover safety net (new with D12).** Until the v2 backup ships, a primary-host outage during market hours or at month-end close means a missed trading event (not a lost order — state is durable in NoSQL). The mitigation is operational (host up during market hours) and structural (v2 backup). Flagged as the v1 #1 risk above; restated here as its own line so it is not read as a minor consequence.
- **NoSQL transaction boundaries.** HIFO selection + lot closure + triplet advance + lease transitions must be atomic; NoSQL conditional writes enforce this but the model is simpler than SQL — Phase 0 validates the transaction patterns.
- **Function cold start on backup trade events — v2 only (D12).** `exec-fn` cold start + Gateway boot may add seconds to the first trade of a failover event. **Not present in v1 (no backup).**

## 20. What this issue does NOT include (out of scope here)

- **Implementation code** — deferred to child build issues after this design is approved. The issue objective is design-only.
- **The `roadmap-30d` implementation task breakdown** — separate deliverable, to be cut once this design is approved.
- **OCI Functions backup-execution path + `execution_lease` fencing + backup failover policy (D12)** — deferred to v2 (Phase 0.5). The design is in this plan; the build is a v2 child issue.
- **Final CEO-set values** (starting capital, account-level DD stop %, per-ticker max shares) — collected before Phase 1, not now.
- **Tax/legal sign-off** — CEO/external.
- **D11 follow-ups** (failover policy auto-vs-manual, current-machine-as-host confirmation, lease TTL) — moot in v1 (no backup); revisited at v2; none block this design.

---

**Next step:** CEO approval of this **revision 6**. On approval, I cut the `roadmap-30d` document with the **v1 Phase 0** task breakdown (podman-primary only — no backup, no fencing) and spawn child build issues. **v1 Phase 0 includes: (a) `uv`-managed project skeleton + polars-first data layer with the lazy/eager boundary tests, (b) the podman-primary persistent-IBKR-Gateway validation + 1-share paper order, (c) NoSQL state/ledger/triplet (no `execution_lease` table in v1), (d) the backtester reconciled to a hand-checked scenario.** The OCI Functions backup, `execution_lease` fencing, and backup failover policy are a **separate v2 (Phase 0.5) child issue**, cut now but not started until v1 paper-trades. If v1 Phase 0 shows the current machine is unsuitable as the sole host (§19 #1), I escalate to you before proceeding — the v2 backup becomes urgent, or we reconsider the host.

