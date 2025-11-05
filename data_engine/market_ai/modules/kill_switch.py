# market_ai/modules/kill_switch.py
from __future__ import annotations
import os

class KillSwitch:
    """
    Create an empty file (default: KILL) to request graceful shutdown.
    """
    def __init__(self, path: str = "KILL"):
        self.path = path

    def tripped(self) -> bool:
        return os.path.exists(self.path)
