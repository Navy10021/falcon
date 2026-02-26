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
        "</tr>"
        for row in rows
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
    <li>Runtime (sec): {summary['runtime_sec']:.3f}</li>
  </ul>
  <h2>Episode Metrics</h2>
  <table>
    <thead>
      <tr><th>Episode</th><th>Outcome</th><th>Friendly Loss</th><th>Enemy Loss</th><th>ROE Violations</th></tr>
    </thead>
    <tbody>
      {metrics_table}
    </tbody>
  </table>
</body>
</html>
"""
