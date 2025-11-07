"""
Compatibility wrapper so legacy imports like ``modules.data_fetch`` resolve to
``market_ai.modules.data_fetch`` during tests.
"""

from __future__ import annotations

import importlib
import sys

_BASE = "market_ai.modules"
_ALIASES = ("data_fetch", "adapters", "agents", "strategies")

# expose the root package as ``modules``
sys.modules.setdefault(__name__, importlib.import_module(_BASE))

for name in _ALIASES:
    target = f"{_BASE}.{name}"
    try:
        sys.modules[f"{__name__}.{name}"] = importlib.import_module(target)
    except ModuleNotFoundError:
        continue
