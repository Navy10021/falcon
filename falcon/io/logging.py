from __future__ import annotations

from datetime import datetime


def log_info(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [falcon-demo] {message}")
