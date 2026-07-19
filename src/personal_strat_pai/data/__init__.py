"""Polars-first data layer (plan §4, §6, D14).

Modules:
    polars_utils — lazy/eager boundary helpers + canonical schemas (D14 centerpiece).
    store        — parquet I/O via polars (BarStore + RateSeriesStore:
                   scan/write/upsert/coverage) + SQLite read cache.
    caching      — shared no-double-download fetch-range math (CEO 2026-07-18/-19).
    repo         — caching BarRepo: no piece of data downloaded twice (CEO 2026-07-18).
    databento    — primary bars source (integration-gated). IV no longer sourced here.
    fred         — primary historical rates source: SOFR / OIS proxy (Treasury CMT)
                   / real rates (TIPS). No-double-download (CEO 2026-07-19).
    yfinance     — fallback: daily bars + metadata + splits (cross-validated).
    rates        — SOFR/OIS curve + real rates (FRED primary for history,
                   databento for live forward curve; D3).
    iv_proxy     — HV (realized vol) IV proxy for backtesting; IBKR for live/paper
                   (CEO 2026-07-18; supersedes D2 EOD-chain approach — no OPRA spend).
    corp_actions — splits/dividends ingestion + cross-validation + ledger hooks.
    quality      — data-quality checks (plan §6.5); failures block consuming jobs.
"""

from __future__ import annotations
