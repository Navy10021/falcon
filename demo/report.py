from __future__ import annotations

from typing import Dict, Iterable


def render_aar_html(summary: Dict[str, float], metrics_rows: Iterable[Dict[str, float]]) -> str:
    rows = list(metrics_rows)
    metrics_table = "\n".join(
        "<tr>"
        f"<td>{row['episode']}</td>"
        f"<td>{row['outcome']}</td>"
        f"<td>{float(row['friendly_loss']):.3f}</td>"
        f"<td>{float(row['enemy_loss']):.3f}</td>"
        f"<td>{int(row['roe_violations'])}</td>"
        f"<td>{row.get('event_type', '-')}</td>"
        f"<td>{row.get('selected_action', '-')}</td>"
        f"<td>{row.get('event_rationale', '-')}</td>"
        "</tr>"
        for row in rows
    )

    blocked_rows = [row for row in rows if row.get("event_type") == "ROE_BLOCKED"]
    blocked_items = "".join(
        f"<li>Episode {row['episode']}: action={row.get('selected_action', '-')}, "
        f"risk={float(row.get('civilian_risk', 0.0)):.3f}, rationale={row.get('event_rationale', '-')}</li>"
        for row in blocked_rows
    )
    blocked_section = (
        f"<h2>ROE Blocked Actions & Rationale</h2><ul>{blocked_items}</ul>"
        if blocked_rows
        else "<h2>ROE Blocked Actions & Rationale</h2><p>No blocked actions in this run.</p>"
    )

    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\" />
  <title>FALCON AAR</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
    th {{ background: #f4f6f8; }}
  </style>
</head>
<body>
  <h1>FALCON AAR - {summary['scenario']}</h1>
  <ul>
    <li>Seed: {summary['seed']}</li>
    <li>Success Rate: {summary['success_rate']:.3f}</li>
    <li>Friendly Loss: {summary['friendly_loss']:.3f}</li>
    <li>ROE Violation Rate: {summary['roe_violation_rate']:.3f}</li>
    <li>ROE Mode: {summary.get('roe_mode', 'on')}</li>
    <li>ROE Blocked Events: {summary.get('roe_blocked_events', 0)}</li>
    <li>HITL Preference: {summary.get('preference', 'balanced')}</li>
    <li>Loss Tradeoff: {summary.get('loss_tradeoff', 0.0):.3f}</li>
    <li>Time Tradeoff: {summary.get('time_tradeoff', 0.0):.3f}</li>
    <li>Runtime (sec): {summary['runtime_sec']:.3f}</li>
  </ul>
  {blocked_section}
  <h2>Episode Metrics</h2>
  <table>
    <thead>
      <tr><th>Episode</th><th>Outcome</th><th>Friendly Loss</th><th>Enemy Loss</th><th>ROE Violations</th><th>Event</th><th>Selected Action</th><th>Rationale</th></tr>
    </thead>
    <tbody>
      {metrics_table}
    </tbody>
  </table>
</body>
</html>
"""
