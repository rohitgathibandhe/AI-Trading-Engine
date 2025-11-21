from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


class EngineLogger:
    def __init__(self, path: Path):
        self.path = path
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "timestamp",
                        "regime",
                        "strategy",
                        "scores",
                        "strike_details",
                        "entry_price",
                        "exit_price",
                        "pnl",
                        "reason",
                    ],
                )
                writer.writeheader()

    def log(self, row: Dict[str, str]):
        with self.path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=row.keys())
            writer.writerow(row)
