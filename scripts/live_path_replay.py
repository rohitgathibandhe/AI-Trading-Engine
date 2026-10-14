#!/usr/bin/env python3
"""Replay recorded market days through the REAL live loop (`run_live`) — the honest "what would the agent
have done" test. Unlike `cli run_backtest`, every live gate, entry wait, paper entry and live exit runs.

It never touches the live state folder: the code runs from a throw-away git worktree whose own
data_engine/market_ai/state/ is seeded with copies of the live config files.

  python scripts/live_path_replay.py --start 2026-10-12 --end 2026-10-14 [--data <dataset dir>] [--keep]

Dataset: built by scripts/build_live_dataset.py from rolling_option_live (real chains). Settings: the env
vars of the live launchd job (~/Library/LaunchAgents/com.algoagent.intraday_v83.plist).
"""
from __future__ import annotations

import argparse
import json
import plistlib
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_STATE = ROOT / "data_engine" / "market_ai" / "state"
PY = "/Users/Rohit/.venvs/data_engine_py310/bin/python"
PLIST = Path.home() / "Library" / "LaunchAgents" / "com.algoagent.intraday_v83.plist"
SEED_FILES = ("intraday_v83_run_live_config.json", "intraday_v83_runtime_config.json", "agent_settings.json",
              "promotion_state.json", "creds.json", "dhan_scrip_master.csv")

RUNNER = r'''
import json, sys
from market_ai.intraday_defined_risk import replay_provider as rp
from market_ai.intraday_defined_risk import monitor
rp.install_clock()
cfg = json.load(open(sys.argv[1]))
try:
    monitor.run_live(cfg)
except rp.ReplayFinished:
    print("REPLAY_FINISHED")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--ref", default="HEAD", help="git ref of the code to replay (default: current HEAD)")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="livereplay_"))
    wt = work / "repo"
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(wt), a.ref], check=True,
                   capture_output=True)
    try:
        data = Path(a.data) if a.data else work / "dataset"
        if not a.data:
            subprocess.run([PY, str(ROOT / "scripts" / "build_live_dataset.py"), str(data)], check=True,
                           capture_output=True)
        st = wt / "data_engine" / "market_ai" / "state"
        st.mkdir(parents=True, exist_ok=True)
        (wt / "data_engine" / "state").mkdir(parents=True, exist_ok=True)     # day_thesis.py writes here
        for name in SEED_FILES:
            if (LIVE_STATE / name).exists():
                shutil.copy2(LIVE_STATE / name, st / name)
        (st / "intraday_v83_reconciliation_status.json").write_text('{"status": "NO_POSITIONS"}')
        cfg = json.loads((st / "intraday_v83_run_live_config.json").read_text())
        cfg.update({"provider_class": "market_ai.intraday_defined_risk.replay_provider:ReplayMarketDataProvider",
                    "replay_data_path": str(data), "replay_start": a.start, "replay_end": a.end})
        assert "PaperOnly" in str(cfg.get("executor_class")), "refusing: executor is not PaperOnly"
        cfg_path = work / "replay_run_live_config.json"
        cfg_path.write_text(json.dumps(cfg))
        env = {k: str(v) for k, v in plistlib.load(open(PLIST, "rb")).get("EnvironmentVariables", {}).items()}
        env.update({"PYTHONPATH": str(wt / "data_engine"), "TRADE_MODE": "paper", "HOME": str(Path.home()),
                    "PATH": "/usr/bin:/bin"})
        proc = subprocess.run([PY, "-c", RUNNER, str(cfg_path)], cwd=str(wt / "data_engine"), env=env,
                              capture_output=True, text=True, timeout=3600)
        if "REPLAY_FINISHED" not in proc.stdout:
            print(proc.stdout[-2000:], proc.stderr[-4000:])
            return 1

        trades = [json.loads(l) for l in (st / "intraday_v83_paper_live_trades.jsonl").read_text().splitlines()
                  if l.strip()] if (st / "intraday_v83_paper_live_trades.jsonl").exists() else []
        decisions = [json.loads(l) for l in (st / "intraday_v83_paper_live_validation_decisions.jsonl").read_text()
                     .splitlines() if l.strip()] if (st / "intraday_v83_paper_live_validation_decisions.jsonl").exists() else []
        by_day = defaultdict(list)
        for t in trades:
            by_day[str(t.get("timestamp", ""))[:10]].append(t)
        blocks = defaultdict(Counter)
        for d in decisions:
            if d.get("raw_action") == "TRADE" or str(d.get("block_reason") or "") not in ("SELECTOR_STAND_ASIDE", "INSUFFICIENT_DATA", "None"):
                blocks[str(d.get("timestamp", ""))[:10]][str(d.get("block_reason"))[:40]] += 1
        total = 0.0
        print(f"LIVE-PATH REPLAY {a.start}..{a.end} (code {a.ref}, real run_live loop, live plist env)")
        for day in sorted(set(by_day) | set(blocks)):
            pnl = sum(float(t.get("realized_paper_pnl") or 0) for t in by_day[day] if t.get("event") == "PAPER_EXIT")
            total += pnl
            print(f"\n{day}  P&L Rs {pnl:,.0f}")
            for t in by_day[day]:
                if t.get("event") == "PAPER_ENTRY":
                    print(f"  ENTRY {t['timestamp'][11:16]} {t.get('strategy')} {t.get('playbook')} spot {t.get('spot_at_entry')}")
                else:
                    print(f"  EXIT  {t['timestamp'][11:16]} {t.get('exit_reason')} Rs {float(t.get('realized_paper_pnl') or 0):,.0f}")
            if blocks[day]:
                print("  blockers:", dict(blocks[day].most_common(5)))
        print(f"\nTOTAL Rs {total:,.0f}")
        (work / "summary.json").write_text(json.dumps({"total": total, "trades": trades}, default=str))
        print(f"artifacts: {work}")
        return 0
    finally:
        if not a.keep:
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)], capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
