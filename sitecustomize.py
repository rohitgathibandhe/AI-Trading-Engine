"""
Ensure the in-repo market_ai package (data_engine/market_ai) is the one that gets imported.

Having multiple working copies on the same machine can cause Python to pick up a
different installed copy from site-packages. By inserting the local src path at
the front of sys.path we force imports to use this checkout first.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data_engine"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
