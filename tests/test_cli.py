"""Smoke test for the CLI (plan §4)."""

from __future__ import annotations

import os

import pytest

from personal_strat_pai.cli import main


def test_config_show_runs_and_returns_zero(capsys: pytest.CaptureFixture[str]):
    rc = main(["config-show"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "buckets" in out
    assert "XLB" in out
    assert "monthly_cash_injection" in out


def test_data_check_empty_store_returns_nonzero(tmp_path, capsys: pytest.CaptureFixture[str]):
    rc = main(["data-check", "--base-uri", str(tmp_path / "empty"), "--kind", "daily"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "no bars" in out


def test_rates_ingest_without_api_key_returns_nonzero(tmp_path, capsys: pytest.CaptureFixture[str]):
    # No FRED_API_KEY -> the caching repo's first fetch raises RuntimeError;
    # the CLI catches it and returns 3 with a helpful message.
    old = os.environ.pop("FRED_API_KEY", None)
    try:
        rc = main(["rates-ingest", "--base-uri", str(tmp_path / "rates"), "--end", "2024-01-31"])
        out = capsys.readouterr().out
        assert rc == 3
        assert "FRED" in out
    finally:
        if old is not None:
            os.environ["FRED_API_KEY"] = old


def test_rates_ingest_help_lists_subcommand(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        main(["rates-ingest", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "FRED" in out
    assert "bootstrap-start" in out
    assert "series" in out
