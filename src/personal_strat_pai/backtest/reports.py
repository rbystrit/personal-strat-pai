"""Reports — quantstats HTML + parquet trades + JSON metrics (plan §10).

Renders three output artifacts for a ``BacktestResult``:
  1. **quantstats HTML** — a full HTML tear-sheet from the equity curve.
     Uses ``quantstats`` (pandas interop boundary, D14) when available;
     falls back to a simple HTML summary if quantstats is not installed.
  2. **parquet trades** — the trade-level report as a parquet file.
  3. **JSON metrics** — aggregate stats + tax-drag + cost attribution.

All three are written to an output directory alongside the ``run_manifest.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from personal_strat_pai.backtest.metrics import (
    compute_cost_attribution,
    compute_tax_drag,
)
from personal_strat_pai.backtest.portfolio import Portfolio

__all__ = [
    "render_reports",
]


def render_reports(
    result: Any,
    portfolio: Portfolio,
    output_dir: str | Path,
    *,
    quantstats_html: bool = True,
) -> dict[str, str]:
    """Render all reports to ``output_dir``. Returns ``{artifact: path}``.

    ``result``: a ``BacktestResult``.
    ``portfolio``: the portfolio after the run (for tax-drag + cost attribution).
    ``output_dir``: the directory to write to (created if missing).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    # 1. Parquet trades.
    trades_path = out / "trades.parquet"
    result.trades_df().write_parquet(trades_path)
    artifacts["trades"] = str(trades_path)

    # 2. Parquet equity curve.
    equity_path = out / "equity_curve.parquet"
    result.equity_curve.write_parquet(equity_path)
    artifacts["equity_curve"] = str(equity_path)

    # 3. JSON metrics + tax-drag + cost attribution.
    tax_drag = compute_tax_drag(portfolio)
    cost_attr = compute_cost_attribution(portfolio)
    metrics_json = {
        "stats": result.stats,
        "tax_drag": tax_drag,
        "cost_attribution": cost_attr,
        "risk_events": result.risk_events,
        "manifest": result.manifest,
        "n_trades": len(result.trades),
    }
    metrics_path = out / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    artifacts["metrics"] = str(metrics_path)

    # 4. Run manifest.
    manifest_path = out / "run_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(result.manifest, f, indent=2, default=str)
    artifacts["manifest"] = str(manifest_path)

    # 5. quantstats HTML (pandas interop boundary, D14).
    if quantstats_html:
        html_path = out / "report.html"
        try:
            _render_quantstats_html(result, html_path)
        except Exception:
            _render_simple_html(result, html_path)
        artifacts["html"] = str(html_path)

    return artifacts


def _render_quantstats_html(result: Any, path: Path) -> None:
    """Render a quantstats HTML tear-sheet from the equity curve.

    Converts the polars equity curve to a pandas Series (D14 interop boundary)
    and passes it to ``quantstats.reports.html``. Falls back to simple HTML
    if quantstats is not installed.
    """

    try:
        import quantstats as qs
    except ImportError as exc:
        raise ImportError("quantstats not installed") from exc

    # Convert equity curve to a pandas return series.
    eq = result.equity_curve.to_pandas()
    eq = eq.set_index("date")
    returns = eq["nav"].pct_change().dropna()
    returns.name = "Strategy"

    qs.reports.html(returns, output=str(path), title="Backtest Report")


def _render_simple_html(result: Any, path: Path) -> None:
    """Fallback HTML report when quantstats is not available."""
    s = result.stats
    html = f"""<!DOCTYPE html>
<html>
<head><title>Backtest Report</title>
<style>
body {{ font-family: monospace; margin: 2em; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #eee; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<h2>Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Final NAV</td><td>{result.final_nav:,.2f}</td></tr>
<tr><td>Total Return</td><td>{s.get('total_return', 0):.2%}</td></tr>
<tr><td>CAGR</td><td>{s.get('cagr', 0):.2%}</td></tr>
<tr><td>Volatility</td><td>{s.get('volatility', 0):.2%}</td></tr>
<tr><td>Sharpe</td><td>{s.get('sharpe', 0):.3f}</td></tr>
<tr><td>Sortino</td><td>{s.get('sortino', 0):.3f}</td></tr>
<tr><td>Max Drawdown</td><td>{s.get('max_drawdown', 0):.2%}</td></tr>
<tr><td>Calmar</td><td>{s.get('calmar', 0):.3f}</td></tr>
<tr><td>Win Rate</td><td>{s.get('win_rate', 0):.2%}</td></tr>
<tr><td>Avg Gross Exposure</td><td>{s.get('avg_gross_exposure', 0):.2%}</td></tr>
<tr><td>Turnover</td><td>{s.get('turnover', 0):.2f}x</td></tr>
<tr><td>Total Commission</td><td>${s.get('total_commission', 0):,.2f}</td></tr>
<tr><td>Total Fees</td><td>${s.get('total_fees', 0):,.2f}</td></tr>
<tr><td>Total Slippage</td><td>${s.get('total_slippage', 0):,.2f}</td></tr>
<tr><td>Total Costs</td><td>${s.get('total_costs', 0):,.2f}</td></tr>
<tr><td>N Trades</td><td>{len(result.trades)}</td></tr>
</table>
<h2>Manifest</h2>
<pre>{json.dumps(result.manifest, indent=2, default=str)}</pre>
</body>
</html>"""
    path.write_text(html)
