#!/usr/bin/env python3
"""
Lightweight HTTP server for a flicker-free Paper P&L frontend.

Serves:
  - /api/paper_positions : JSON snapshot of paper positions (aggregated from trade_blotter.csv)
  - static files under web/paper_pnl/ (index.html uses the API above)

Usage:
  python scripts/paper_pnl_server.py --port 8000
Then open http://localhost:8000 in your browser.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import signal

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "data_engine" / "market_ai" / "state"
AGENT_LOG = STATE_DIR / "agent.log"
BLOTTER_CSV = STATE_DIR / "trade_blotter.csv"
STRATEGY_STATE = STATE_DIR / "strategy_state.json"
SETTINGS_JSON = STATE_DIR / "agent_settings.json"
CREDS_FILE = STATE_DIR / "creds.json"
LAST_STRATEGY_FILE = STATE_DIR / "last_strategy.json"
PID_FILE = STATE_DIR / "agent.pid"
AGENT_ENTRY = ROOT / "data_engine" / "market_ai" / "start_agent.py"
# New unified frontend location
STATIC_DIR = ROOT / "web" / "app"


def _parse_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _parse_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val))
    except Exception:
        return default


def load_positions(blotter_path: Path, mode: str = "paper") -> Dict[str, Any]:
    """
    Aggregate OPEN + latest MTM rows into per-leg positions and total P&L.
    """
    legs: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    latest_expiry: Optional[str] = None
    mode = (mode or "").lower()

    if not blotter_path.exists():
        return {"positions": [], "total_pnl": 0.0, "as_of": datetime.now().isoformat(), "blotter_tail": []}

    tail_rows: List[Dict[str, Any]] = []
    with blotter_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        tail_rows = rows[-30:] if len(rows) > 30 else rows

    # determine latest expiry in selected mode rows
    for row in rows:
        if (row.get("trade_mode") or "").lower() != mode:
            continue
        exp = row.get("expiry") or ""
        if exp and (latest_expiry is None or exp > latest_expiry):
            latest_expiry = exp

    for row in rows:
        if (row.get("trade_mode") or "").lower() != mode:
            continue
        exp = row.get("expiry") or ""
        if latest_expiry and exp != latest_expiry:
            continue
        sec_id = str(row.get("security_id") or "")
        strike = row.get("strike") or ""
        side = str(row.get("side") or "").upper()
        qty = _parse_int(row.get("quantity"), 0)
        price = _parse_float(row.get("price"), 0.0)
        notes = str(row.get("notes") or "").upper()
        key = (sec_id, str(strike), str(exp))
        leg = legs.setdefault(
            key,
            {
                "entry": None,
                "entry_qty": 0,
                "ltp": None,
                "qty": 0,
                "side": side,
                "strike": strike,
                "expiry": exp,
                "sec_id": sec_id,
            },
        )

        if notes == "OPEN":
            prev_qty = leg.get("entry_qty", 0)
            signed = qty if side == "SELL" else -qty
            new_qty = prev_qty + signed
            if new_qty != 0:
                prev_entry = leg.get("entry") or 0.0
                leg["entry"] = ((prev_entry * prev_qty) + (price * signed)) / new_qty
                leg["entry_qty"] = new_qty
            leg["qty"] = new_qty
            leg["side"] = "SELL" if new_qty > 0 else "BUY"
        elif notes == "MTM":
            leg["ltp"] = price
            leg["qty"] = qty if side == "SELL" else -qty
            leg["side"] = "SELL" if leg["qty"] > 0 else "BUY"

    positions: List[Dict[str, Any]] = []
    total_pnl = 0.0
    for leg in legs.values():
        entry_px = leg.get("entry")
        ltp_px = leg.get("ltp") if leg.get("ltp") not in (None, "") else entry_px
        if entry_px is None:
            continue
        qty = abs(int(leg.get("qty") or leg.get("entry_qty") or 0))
        side = leg.get("side")
        pnl = None
        if ltp_px is not None:
            if side == "SELL":
                pnl = (entry_px - ltp_px) * qty
            else:
                pnl = (ltp_px - entry_px) * qty
            total_pnl += pnl
        positions.append(
            {
                "side": side,
                "strike": leg.get("strike"),
                "expiry": leg.get("expiry"),
                "sec_id": leg.get("sec_id"),
                "qty": qty,
                "entry": entry_px,
                "ltp": ltp_px,
                "pnl": pnl,
            }
        )

    return {
        "positions": positions,
        "total_pnl": total_pnl,
        "as_of": datetime.now().isoformat(),
        "blotter_tail": tail_rows,
    }


def _json_read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _json_write(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


class PaperHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # type: ignore[override]
        if self.path.startswith("/api/settings"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
                current = {}
                if SETTINGS_JSON.exists():
                    try:
                        current = json.loads(SETTINGS_JSON.read_text())
                    except Exception:
                        current = {}
                current.update(data or {})
                SETTINGS_JSON.write_text(json.dumps(current, indent=2))
                self._send_json({"ok": True, "settings": current})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/creds"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    data = {}
                current = _json_read(CREDS_FILE)
                current.update(data or {})
                CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
                _json_write(CREDS_FILE, current)
                self._send_json({"ok": True, "creds": current})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/strategy"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
                if data.get("strategy_file"):
                    _json_write(LAST_STRATEGY_FILE, {"strategy_file": data["strategy_file"]})
                settings = _json_read(SETTINGS_JSON)
                settings.update(data.get("settings", {}))
                _json_write(SETTINGS_JSON, settings)
                self._send_json({"ok": True, "strategy": data})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/control/start"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
                trade_mode = str(data.get("trade_mode", "paper"))
                # prevent double-start
                if PID_FILE.exists():
                    try:
                        pid_data = json.loads(PID_FILE.read_text())
                        pid = int(pid_data.get("pid", 0))
                        if pid and _is_process_alive(pid):
                            self._send_json({"ok": False, "error": "agent already running", "pid": pid})
                            return
                    except Exception:
                        pass
                env = os.environ.copy()
                env["TRADE_MODE"] = trade_mode
                proc = subprocess.Popen(
                    ["python3", str(AGENT_ENTRY)],
                    cwd=str(ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
                pid_payload = {"pid": proc.pid, "trade_mode": trade_mode, "started_at": datetime.now().isoformat()}
                _json_write(PID_FILE, pid_payload)
                self._send_json({"ok": True, "pid": proc.pid})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/control/stop"):
            try:
                pid_data = _json_read(PID_FILE)
                pid = int(pid_data.get("pid", 0)) if pid_data else 0
                if pid and _is_process_alive(pid):
                    os.kill(pid, signal.SIGTERM)
                _json_write(PID_FILE, {})
                self._send_json({"ok": True})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        return super().do_POST()

    def do_GET(self) -> None:  # type: ignore[override]
        if self.path.startswith("/api/paper_positions"):
            try:
                payload = load_positions(BLOTTER_CSV, mode="paper")
                # also include strategy status for context
                try:
                    state = json.loads(STRATEGY_STATE.read_text())
                except Exception:
                    state = {}
                payload["strategy_state"] = state
                self._send_json(payload)
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/live_positions"):
            try:
                payload = load_positions(BLOTTER_CSV, mode="live")
                self._send_json(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/strategy_state"):
            try:
                state = json.loads(STRATEGY_STATE.read_text()) if STRATEGY_STATE.exists() else {}
                self._send_json({"strategy_state": state})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/settings"):
            try:
                settings = json.loads(SETTINGS_JSON.read_text()) if SETTINGS_JSON.exists() else {}
                self._send_json({"settings": settings})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/creds"):
            try:
                creds = _json_read(CREDS_FILE)
                self._send_json({"creds": creds})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/strategy"):
            try:
                last = _json_read(LAST_STRATEGY_FILE)
                settings = _json_read(SETTINGS_JSON)
                self._send_json({"strategy": last, "settings": settings})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/control/status"):
            try:
                pid_data = _json_read(PID_FILE)
                pid = int(pid_data.get("pid", 0)) if pid_data else 0
                running = pid and _is_process_alive(pid)
                self._send_json({"running": bool(running), "pid": pid})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/logs"):
            try:
                tail = 200
                try:
                    from urllib.parse import urlparse, parse_qs
                    qs = parse_qs(urlparse(self.path).query)
                    if "tail" in qs:
                        tail = int(qs["tail"][0])
                except Exception:
                    pass
                lines = []
                if AGENT_LOG.exists():
                    with AGENT_LOG.open("r") as f:
                        lines = f.readlines()[-tail:]
                self._send_json({"lines": lines})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        # Serve static frontend
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
        return super().do_GET()

    def translate_path(self, path: str) -> str:
        # Serve files from STATIC_DIR by default
        root = STATIC_DIR if STATIC_DIR.exists() else Path.cwd()
        # Adapted from SimpleHTTPRequestHandler: map URL to local file under root
        # while preventing path traversal.
        import posixpath
        path = path.split("?",1)[0].split("#",1)[0]
        path = posixpath.normpath(path)
        words = path.split("/")
        words = [_f for _f in words if _f]
        resolved = root
        for word in words:
            resolved = resolved / word
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper P&L mini server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PAPER_PNL_PORT", "8000")))
    args = parser.parse_args()
    os.chdir(STATIC_DIR)
    httpd = HTTPServer(("", args.port), PaperHandler)
    print(f"Serving Paper P&L frontend on http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
