"""
modules/utils/config_reloader.py

Watches the config/ folder for YAML file changes and triggers live reloads.
Use this in agent/backtest loops for adaptive reconfiguration without restart.
"""

import os
import time
import threading
from typing import Callable, Dict
from market_ai.modules.utils.config_loader import load_all_configs


class ConfigReloader:
    """
    Watches config/ directory for changes in any .yaml file.
    When change detected, automatically reloads all configs and triggers callback.
    """

    def __init__(self, config_dir: str = "config", check_interval: int = 10):
        self.config_dir = config_dir
        self.check_interval = check_interval
        self.last_mtimes: Dict[str, float] = {}
        self.stop_flag = threading.Event()
        self.callback: Callable[[Dict], None] = None
        self._thread = None

    def _scan_files(self) -> Dict[str, float]:
        mtimes = {}
        for root, _, files in os.walk(self.config_dir):
            for f in files:
                if f.endswith(".yaml"):
                    path = os.path.join(root, f)
                    try:
                        mtimes[path] = os.path.getmtime(path)
                    except FileNotFoundError:
                        pass
        return mtimes

    def _watch_loop(self):
        while not self.stop_flag.is_set():
            current_mtimes = self._scan_files()
            # Compare to previous snapshot
            for path, mtime in current_mtimes.items():
                if path not in self.last_mtimes or mtime != self.last_mtimes[path]:
                    print(f"🔁 Detected change in {path} → reloading configs ...")
                    try:
                        new_cfg = load_all_configs()
                        if self.callback:
                            self.callback(new_cfg)
                        print("✅ Configs reloaded successfully.")
                    except Exception as e:
                        print(f"⚠️ Failed to reload config: {e}")
                    break  # Avoid multiple triggers within same interval
            self.last_mtimes = current_mtimes
            time.sleep(self.check_interval)

    def start(self, callback: Callable[[Dict], None]):
        """
        Starts background watcher thread and sets callback.
        The callback will be passed the new config dict.
        """
        self.callback = callback
        self.last_mtimes = self._scan_files()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        print(f"👁️  ConfigReloader started (watching {self.config_dir})")

    def stop(self):
        """Stops the watcher thread."""
        self.stop_flag.set()
        if self._thread:
            self._thread.join(timeout=3)
        print("🛑 ConfigReloader stopped.")
