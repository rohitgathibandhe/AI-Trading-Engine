# market_ai/modules/notifier.py
from __future__ import annotations
import logging

LOG = logging.getLogger(__name__)

class Notifier:
    """
    Minimal notifier. For now, logs to console.
    Later: plug email/Telegram/Slack.
    """
    def __init__(self, channel: str = "console"):
        self.channel = channel

    def send(self, title: str, text: str):
        LOG.info("NOTIFY [%s] %s | %s", self.channel, title, text)
