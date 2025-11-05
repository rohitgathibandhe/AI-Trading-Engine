import os
import json
import datetime
from typing import Dict, Any, Optional, List

class IVStore:
    """
    Minimal rolling IV store backed by JSON.
    Structure on disk:
      {
        "NIFTY": [{"date":"2025-10-23", "iv": 0.1523}, ...],
        "BANKNIFTY": [...]
      }
    """
    def __init__(self, path: str, max_days: int = 60):
        self.path = path
        self.max_days = max_days
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._data: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)

    def add_today_iv(self, symbol: str, iv_value: float) -> None:
        if iv_value is None or iv_value <= 0:
            return
        today = datetime.date.today().isoformat()
        series = self._data.get(symbol, [])
        # replace today if exists
        series = [row for row in series if row.get("date") != today]
        series.append({"date": today, "iv": float(iv_value)})
        # trim
        series = sorted(series, key=lambda r: r["date"])[-self.max_days:]
        self._data[symbol] = series
        self._save()

    def percentile(self, symbol: str, current_iv: float) -> Optional[float]:
        """
        Return IV percentile (0-100) of current_iv vs stored history.
        If insufficient history (<10), return None.
        """
        series = self._data.get(symbol, [])
        vals = [float(r["iv"]) for r in series if r.get("iv") is not None]
        if len(vals) < 10 or current_iv is None or current_iv <= 0:
            return None
        below = sum(1 for v in vals if v <= current_iv)
        return 100.0 * below / max(1, len(vals))
