# data_engine/market_ai/infra/command_bus.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = PROJECT_ROOT / "data_engine" / "market_ai"
CMD_DIR = AGENT_DIR / "agent_commands"
CMD_QUEUE = CMD_DIR / "queue.jsonl"
OFFSET_FILE = CMD_DIR / ".queue.offset"

class CommandBus:
    """
    File-based JSONL command bus.
    - Append-only queue written by UI.
    - Agent polls/reads new lines and processes them.
    - Uses a simple byte-offset file so we don't re-read old lines.
    """

    def __init__(self, queue_path: Path = CMD_QUEUE, offset_path: Path = OFFSET_FILE):
        self.queue_path = Path(queue_path)
        self.offset_path = Path(offset_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.queue_path.exists():
            self.queue_path.touch()

    def _read_offset(self) -> int:
        try:
            return int(self.offset_path.read_text().strip())
        except Exception:
            return 0

    def _write_offset(self, offset: int) -> None:
        try:
            self.offset_path.write_text(str(offset))
        except Exception:
            pass

    def poll(self, block: bool = False, sleep_sec: float = 0.5) -> Iterator[Dict]:
        """
        Yields dict commands as they arrive.
        If block=True, keeps waiting for new commands.
        """
        while True:
            start = self._read_offset()
            size = self.queue_path.stat().st_size
            if size > start:
                with open(self.queue_path, "r", encoding="utf-8") as f:
                    f.seek(start)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            cmd = json.loads(line)
                            yield cmd
                        except Exception:
                            # skip bad lines
                            pass
                    self._write_offset(f.tell())
            if not block:
                return
            time.sleep(sleep_sec)

    def send(self, command: Dict) -> Tuple[bool, str]:
        """Handy for tests—lets the agent also push commands if needed."""
        try:
            line = json.dumps(command, separators=(",", ":")) + "\n"
            with open(self.queue_path, "a", encoding="utf-8") as f:
                f.write(line)
            return True, "ok"
        except Exception as e:
            return False, str(e)
