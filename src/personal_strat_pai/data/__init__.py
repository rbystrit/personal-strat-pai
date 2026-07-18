"""Polars-first data layer (plan §4, §6, D14).

Modules:
    polars_utils — lazy/eager boundary helpers + canonical schemas (D14 centerpiece).
    store        — parquet I/O via polars (scan_parquet/write_parquet) + SQLite read cache.
    databento    — primary: bars + EOD options (IV) + SOFR/OIS (integration-gated).
    yfinance     — fallback: daily bars + metadata + splits (cross-validated).
    rates        — SOFR/OIS curve (databento, D3).
    iv_proxy     — self-built IV from EOD chain snapshot, NO OPRA (D2).
    corp_actions — splits/dividends ingestion + cross-validation + ledger hooks.
    quality      — data-quality checks (plan §6.5); failures block consuming jobs.
"""

from __future__ import annotations
