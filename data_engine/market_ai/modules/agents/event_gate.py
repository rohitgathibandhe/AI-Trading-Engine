import json
import os
import datetime
from typing import Optional, Dict, Any, List

def _parse_event(dt_str: str) -> datetime.datetime:
    # dt_str like "2025-10-24 10:00"
    return datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M")

def _load_calendar(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def is_blackout_now(now: datetime.datetime, calendar_path: str, minutes_before: int) -> Optional[Dict[str, Any]]:
    """
    Returns the blocking event dict if 'now' is within [t_event - minutes_before, t_event].
    Calendar JSON shape: [{ "date":"YYYY-MM-DD", "time":"HH:MM", "title":"...", "severity":"high|med|low"}]
    """
    cal = _load_calendar(calendar_path)
    window = datetime.timedelta(minutes=minutes_before)

    for ev in cal:
        dt = _parse_event(f'{ev.get("date")} {ev.get("time","00:00")}')
        if dt - window <= now <= dt:
            out = dict(ev)
            out["window_start"] = (dt - window).isoformat()
            out["window_end"] = dt.isoformat()
            return out
    return None
