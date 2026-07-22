"""Run manifest — reproducibility for backtest runs (plan §10).

The manifest captures everything needed to verify and reproduce a backtest:
  * Dataset: source path, symbols, date range.
  * Config hash: SHA-256 of the ``BacktestConfig``.
  * Git SHA: the current commit (for code reproducibility).
  * Parquet data hash: SHA-256 of the parquet files (for data reproducibility).

Written as ``run_manifest.json`` alongside the reports. The manifest is the
acceptance criterion: "run_manifest.json captures dataset+range, config hash,
git sha, parquet data hash" (plan §10).
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

__all__ = [
    "build_manifest",
    "compute_parquet_hash",
    "get_git_sha",
]


def get_git_sha() -> str:
    """Current git commit SHA (short). Returns 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_parquet_hash(source: str | Path | list[str | Path] | None) -> str:
    """SHA-256 hash of the parquet files at ``source``.

    For a directory: hashes all ``*.parquet`` files (sorted by path) together.
    For a list of paths: hashes each file. For ``None``: returns 'no-data'.
    """
    if source is None:
        return "no-data"

    paths: list[Path] = []
    if isinstance(source, str | Path):
        p = Path(source)
        if p.is_dir():
            paths = sorted(p.rglob("*.parquet"))
        elif p.is_file():
            paths = [p]
    elif isinstance(source, list):
        for s in source:
            p = Path(s)
            if p.is_dir():
                paths.extend(sorted(p.rglob("*.parquet")))
            elif p.is_file():
                paths.append(p)
        paths.sort()

    if not paths:
        return "no-data"

    hasher = hashlib.sha256()
    for p in paths:
        hasher.update(p.name.encode())
        hasher.update(b"\0")
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def build_manifest(
    *,
    config: Any,
    symbols: list[str],
    trading_days: list[date],
    source: str | Path | list[str | Path] | None,
) -> dict[str, Any]:
    """Build the run manifest dict (plan §10).

    ``config``: the ``BacktestConfig`` (has ``to_dict`` and ``config_hash``).
    ``symbols``: all symbols in the data.
    ``trading_days``: the sorted trading dates.
    ``source``: the parquet source path (for hashing).
    """
    return {
        "dataset": {
            "source": str(source) if source is not None else None,
            "symbols": sorted(symbols),
            "n_symbols": len(symbols),
            "start_date": trading_days[0].isoformat() if trading_days else None,
            "end_date": trading_days[-1].isoformat() if trading_days else None,
            "n_trading_days": len(trading_days),
        },
        "config_hash": config.config_hash(),
        "config": config.to_dict(),
        "git_sha": get_git_sha(),
        "parquet_data_hash": compute_parquet_hash(source),
    }
