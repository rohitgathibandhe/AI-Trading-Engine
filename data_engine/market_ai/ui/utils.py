from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[3]        # .../Algotrade_ai
DATA_ENGINE = ROOT / "data_engine"
LOG_DIR = DATA_ENGINE / "logs"
ENV_FILE = DATA_ENGINE / ".env.dhan"
STATE_DIR = DATA_ENGINE / "state"

KILL_FILE = STATE_DIR / "kill.switch"             # kill-switch file
PID_FILE = LOG_DIR / "daemon.pid"                 # daemon pid
MANUAL_EXIT_FLAG = STATE_DIR / "manual_exit.flag" # manual exit request


# =========================
# NSE calendar helpers
# =========================
def _is_tuesday(d: date) -> bool:
    return d.weekday() == 1

def last_tuesday_of_month(y: int, m: int) -> date:
    if m == 12:
        nm = date(y + 1, 1, 1)
    else:
        nm = date(y, m + 1, 1)
    d = nm - timedelta(days=1)
    while not _is_tuesday(d):
        d -= timedelta(days=1)
    return d

def next_tuesday(d: Optional[date] = None) -> date:
    d = d or date.today()
    ahead = (1 - d.weekday()) % 7
    if ahead == 0:
        ahead = 7
    return d + timedelta(days=ahead)

def normalize_expiry(mode: str, user_expiry: Optional[str] = None) -> str:
    """Return ISO date string for desired expiry mode."""
    if mode == "auto_weekly":
        return next_tuesday().isoformat()
    if mode == "auto_monthly":
        today = date.today()
        d = last_tuesday_of_month(today.year, today.month)
        if d < today:
            # move to next month
            yy, mm = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
            d = last_tuesday_of_month(yy, mm)
        return d.isoformat()
    # manual
    if user_expiry:
        try:
            y, m, dd = map(int, user_expiry.split("-"))
            return date(y, m, dd).isoformat()
        except Exception:
            pass
    # fallback
    return next_tuesday().isoformat()


# =========================
# Env helpers
# =========================
def read_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out

def write_env(values: Dict[str, str]) -> None:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = read_env()
    existing.update(values)
    ENV_FILE.write_text("\n".join(f"{k}={v}" for k, v in existing.items()))

def get_mode_in_env() -> str:
    return read_env().get("TRADE_MODE", "paper")

def set_mode_in_env(mode: str) -> None:
    write_env({"TRADE_MODE": mode})


# =========================
# Kill-switch & manual exit
# =========================
def kill_switch_on() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    KILL_FILE.write_text("KILL")

def kill_switch_off() -> None:
    try:
        KILL_FILE.unlink()
    except FileNotFoundError:
        pass

def kill_switch_status() -> bool:
    return KILL_FILE.exists()

def signal_manual_exit() -> None:
    """Create a flag that tells the daemon to close the current trade."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_EXIT_FLAG.write_text("EXIT")

def clear_manual_exit() -> None:
    try:
        MANUAL_EXIT_FLAG.unlink()
    except FileNotFoundError:
        pass

def manual_exit_pending() -> bool:
    return MANUAL_EXIT_FLAG.exists()


# =========================
# Daemon management & logs
# =========================
def _env_for_daemon(mode: str) -> dict:
    env = os.environ.copy()
    env.update(read_env())
    env["TRADE_MODE"] = mode
    return env

def daemon_status() -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        raw = (PID_FILE.read_text().strip() or "0")
        try:
            pid = int(raw)
        except ValueError:
            pid = 0
        if pid > 0:
            try:
                os.kill(pid, 0)
                return {"running": True, "pid": pid, "log_dir": str(LOG_DIR)}
            except ProcessLookupError:
                PID_FILE.unlink(missing_ok=True)
    return {"running": False, "pid": None, "log_dir": str(LOG_DIR)}

def start_daemon(mode: str = "paper") -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"daemon_{mode}_{os.getpid()}.log"
    cmd = [
        "nohup",
        "python3", "-m", "market_ai.ai_trade_daemon",
        "--mode", mode, "--debug"
    ]
    proc = subprocess.Popen(
        " ".join(cmd) + f" > {logfile} 2>&1 & echo $!",
        cwd=DATA_ENGINE,
        shell=True,
        env=_env_for_daemon(mode),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid = proc.stdout.read().decode().strip()
    PID_FILE.write_text(pid)
    return f"Daemon ({mode}) started, PID {pid}. Log: {logfile.name}"

def stop_daemon() -> str:
    if PID_FILE.exists():
        raw = (PID_FILE.read_text().strip() or "0")
        try:
            pid = int(raw)
        except ValueError:
            pid = 0
        try:
            if pid > 0:
                os.kill(pid, 15)
            PID_FILE.unlink(missing_ok=True)
            clear_manual_exit()
            return f"Stopped daemon PID {pid}"
        except Exception as e:
            return f"Error stopping daemon: {e}"
    return "No daemon running."

def tail_log(n: int = 200) -> str:
    logs = sorted(LOG_DIR.glob("daemon_*.log"), key=os.path.getmtime, reverse=True)
    if not logs:
        return "No daemon logs found."
    try:
        lines = logs[0].read_text(errors="ignore").splitlines()[-n:]
        return "\n".join(lines)
    except Exception as e:
        return f"(log read error: {e})"

def latest_entry_strikes_from_log() -> Tuple[Optional[float], Optional[float]]:
    """
    Parse daemon tail for most recent 'ENTRY picker → CE xxxx @ ... | PE yyyy @ ...'.
    Returns (ceK, peK) or (None, None) if not found.
    """
    txt = tail_log(400)
    ceK = peK = None
    # support unicode arrow and also ascii variants
    pat = re.compile(r"ENTRY picker\s*(?:→|-|->|=>)?\s*CE\s*(\d+).+?\|\s*PE\s*(\d+)", re.IGNORECASE)
    for line in reversed(txt.splitlines()):
        m = pat.search(line)
        if m:
            ceK = float(m.group(1))
            peK = float(m.group(2))
            break
    return ceK, peK


# =========================
# CSV + analytics
# =========================
def list_trade_logs(log_dir: str | os.PathLike = LOG_DIR) -> List[str]:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return sorted(glob.glob(str(Path(log_dir) / "managed_trades_*.csv")))

def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # timestamp
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    # numeric coercions
    for c in ["spot","ceK","peK","payback","entry_credit","delta_ce","delta_pe","ivp","trend"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    # derived
    df["mtm"] = df["entry_credit"] - df["payback"]
    df["realized_pnl"] = np.where(df["action"].astype(str).str.upper()=="BOOK_PROFIT", df["mtm"], 0.0)
    df["cum_realized"] = df["realized_pnl"].cumsum()
    return df

def daily_realized(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date","realized"])
    tmp = df[["ts","realized_pnl"]].copy()
    tmp["date"] = tmp["ts"].dt.tz_convert("Asia/Kolkata").dt.date
    out = tmp.groupby("date", dropna=False)["realized_pnl"].sum().reset_index()
    out.rename(columns={"realized_pnl":"realized"}, inplace=True)
    return out

def equity_curve(df: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts","equity","cum_realized"])
    ec = df[["ts","cum_realized"]].copy()
    ec["equity"] = starting_capital + ec["cum_realized"]
    return ec
