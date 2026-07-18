"""Smoke test for the CLI (plan §4)."""

from __future__ import annotations

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
