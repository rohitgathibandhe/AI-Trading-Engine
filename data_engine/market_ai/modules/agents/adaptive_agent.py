"""
Compatibility shim for legacy imports.

Older modules referenced ``market_ai.modules.agents.adaptive_agent`` directly.
The project now keeps the live implementation in ``market_ai.agents.adaptive_agent``.
To avoid having two divergent copies of the class (and increasingly stale code),
this module simply re-exports the canonical implementation.

Usage elsewhere in the repo (or in a user's extensions) continues to work:

    from market_ai.modules.agents.adaptive_agent import AdaptiveAgent

Behind the scenes it is the same class used by ``start_agent.py`` and the UI.
"""

from __future__ import annotations

from market_ai.agents.adaptive_agent import (  # re-export canonical agent
    AdaptiveAgent,
    AgentConfig,
    AgentState,
    PaperBroker,
    DEFAULT_VIRTUAL_CASH,
)

__all__ = [
    "AdaptiveAgent",
    "AgentConfig",
    "AgentState",
    "PaperBroker",
    "DEFAULT_VIRTUAL_CASH",
]
