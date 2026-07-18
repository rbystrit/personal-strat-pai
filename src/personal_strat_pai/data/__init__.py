"""Polars-first data layer (plan §4, §6, D14).

Modules:
    polars_utils — lazy/eager boundary helpers + canonical schemas (D14 centerpiece).
    store        — parquet I/O via polars (scan_parquet/write_parquet/upsert_bars/coverage)
                   + SQLite read cache.
    repo         — caching BarRepo: no piece of data downloaded twice (CEO 2026-07-18).
    databento    — primary: bars + SOFR/OIS (integration-gated). IV no longer sourced here.
    yfinance     — fallback: daily bars + metadata + splits (cross-validated).
    rates        — SOFR/OIS curve (databento, D3).
    iv_proxy     — HV (realized vol) IV proxy for backtesting; IBKR for live/paper
                   (CEO 2026-07-18; supersedes D2 EOD-chain approach — no OPRA spend).
    corp_actions — splits/dividends ingestion + cross-validation + ledger hooks.
    quality      — data-quality checks (plan §6.5); failures block consuming jobs.
"""

from __future__ import annotations
