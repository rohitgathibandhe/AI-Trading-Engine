import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_ENGINE = ROOT / "data_engine"
LOG_DIR = DATA_ENGINE / "logs"
PID_FILE = LOG_DIR / "daemon.pid"

def daemon_status() -> dict:
    """Return {'running': bool, 'pid': int|None, 'log': str|None}"""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip() or 0)
        try:
            os.kill(pid, 0)
            return {"running": True, "pid": pid, "log": str(LOG_DIR)}
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
    return {"running": False, "pid": None, "log": None}

def start_daemon(mode: str = "paper") -> str:
    """Start the ai_trade_daemon in background."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"daemon_{mode}_{os.getpid()}.log"
    cmd = [
        "nohup",
        "python3", "-m", "market_ai.ai_trade_daemon",
        "--mode", mode, "--debug"
    ]
    proc = subprocess.Popen(
        " ".join(cmd) + f" > {log_path} 2>&1 & echo $!",
        shell=True,
        cwd=DATA_ENGINE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid = proc.stdout.read().decode().strip()
    PID_FILE.write_text(pid)
    return f"Started daemon ({mode}) with PID {pid}"


def start_batman_daemon(mode: str = "paper") -> str:
    """Start the ai_batman_daemon in background."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"batman_daemon_{mode}_{os.getpid()}.log"
    cmd = [
        "nohup",
        "python3", "scripts/ai_batman_daemon.py",
    ]
    env = os.environ.copy()
    env["TRADE_MODE"] = mode
    proc = subprocess.Popen(
        " ".join(cmd) + f" > {log_path} 2>&1 & echo $!",
        shell=True,
        cwd=DATA_ENGINE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid = proc.stdout.read().decode().strip()
    PID_FILE.write_text(pid)
    return f"Started Batman daemon ({mode}) with PID {pid}"

def stop_daemon() -> str:
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip() or 0)
        try:
            os.kill(pid, 15)
            PID_FILE.unlink(missing_ok=True)
            return f"Stopped daemon PID {pid}"
        except Exception as e:
            return f"Error stopping daemon: {e}"
    return "No daemon running"

def tail_log(n=20) -> str:
    """Return last n lines of the most recent daemon log."""
    logs = sorted(LOG_DIR.glob("daemon_*.log"), key=os.path.getmtime, reverse=True)
    if not logs:
        return "No daemon logs found."
    log_path = logs[0]
    lines = log_path.read_text().splitlines()[-n:]
    return "\n".join(lines)
