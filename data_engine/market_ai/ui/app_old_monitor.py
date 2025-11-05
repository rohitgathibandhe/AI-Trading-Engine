from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from market_ai.ui.utils import (
    list_trade_logs, load_log, compute_trade_spans, equity_curve, last_open_snapshot
)


st.set_page_config(page_title="Alog Trading – Strangle Manager", layout="wide")

st.title("Alog Trading – Trade Monitor 📈")


# --- Sidebar controls ---
st.sidebar.header("Settings")
log_files = list_trade_logs("logs")
if not log_files:
    st.sidebar.warning("No logs found in ./logs. Start the daemon to generate `managed_trades_*.csv`.")
    st.stop()

default_file = log_files[-1]
file_choice = st.sidebar.selectbox("Trade log file", log_files, index=log_files.index(default_file))
start_capital = st.sidebar.number_input("Starting Capital (₹)", min_value=0, value=int(os.environ.get("STARTING_CAPITAL", "500000")))
refresh_sec = st.sidebar.slider("Auto-refresh every (sec)", min_value=0, max_value=120, value=15, help="0 disables auto refresh")
date_from = st.sidebar.date_input("From date", value=None)
date_to   = st.sidebar.date_input("To date", value=None)

if refresh_sec > 0:
    st_autorefresh = st.experimental_rerun  # placeholder to satisfy type checker
    st.experimental_set_query_params(_=str(int(time.time())))
    st.experimental_memo.clear()

# --- Data load ---
df_raw = load_log(file_choice)

# Filter by date range if provided
if date_from:
    df_raw = df_raw[df_raw["ts"].dt.date >= date_from]
if date_to:
    df_raw = df_raw[df_raw["ts"].dt.date <= date_to]

if df_raw.empty:
    st.info("No rows in selected range/file.")
    st.stop()

# --- KPIs ---
open_spot, open_ceK, open_peK, open_mtm = last_open_snapshot(df_raw)
realized = float(df_raw["realized_pnl"].sum())
open_unrealized = float(df_raw.iloc[-1]["mtm"]) if df_raw.iloc[-1]["action"] != "BOOK_PROFIT" else 0.0
num_trades_closed = int((df_raw["action"] == "BOOK_PROFIT").sum())
num_actions = len(df_raw)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Realized P&L (₹)", f"{realized:,.0f}")
col2.metric("Unrealized MTM (₹)", f"{open_unrealized:,.0f}")
col3.metric("Trades Closed", f"{num_trades_closed}")
col4.metric("Last Spot", f"{open_spot:,.1f}")
col5.metric("Open CE/PE", f"{int(open_ceK) if not np.isnan(open_ceK) else '-'} / {int(open_peK) if not np.isnan(open_peK) else '-'}")

st.markdown("---")

# --- Charts ---
ec = equity_curve(df_raw, start_capital)

left, right = st.columns((2,1), gap="large")

with left:
    st.subheader("Capital Growth (Equity Curve)")
    if ec.empty:
        st.info("No equity data yet.")
    else:
        st.line_chart(ec.set_index("ts")[["equity"]])

    st.subheader("MTM vs Time")
    tmp = df_raw[["ts","mtm"]].copy().set_index("ts")
    st.line_chart(tmp)

with right:
    st.subheader("Delta Tracks (abs)")
    dtmp = df_raw[["ts","delta_ce","delta_pe"]].copy()
    dtmp["|ΔCE|"] = dtmp["delta_ce"].abs()
    dtmp["|ΔPE|"] = dtmp["delta_pe"].abs()
    dtmp = dtmp.drop(columns=["delta_ce","delta_pe"]).set_index("ts")
    st.line_chart(dtmp)

    st.subheader("Strikes Selected")
    ktmp = df_raw[["ts","ceK","peK"]].copy().set_index("ts")
    st.line_chart(ktmp)

st.markdown("---")

# --- Per-trade summary ---
st.subheader("Per-Trade Summary")
summary = compute_trade_spans(df_raw)
if summary.empty:
    st.info("No completed trades yet. Current cycle is open.")
else:
    show_cols = ["trade_id","entry_ts","exit_ts","duration","entry_credit","final_payback","exit_mtm","realized_pnl","last_ceK","last_peK"]
    st.dataframe(summary[show_cols], use_container_width=True)

# --- Ledger ---
st.subheader("Action Ledger")
st.dataframe(
    df_raw[["ts","action","spot","ceK","peK","entry_credit","payback","mtm","delta_ce","delta_pe","ivp","trend","notes"]],
    use_container_width=True
)

# --- Footer / Help ---
with st.expander("Notes"):
    st.write("""
- **Equity Curve** uses **realized P&L** only (from `BOOK_PROFIT` events).
- **MTM** is estimated as `entry_credit - payback` on each log tick.
- If you are running **paper** mode, the “Broker” fills are assumed at logged LTP; in live mode use your broker fills.
- Multiple runs create multiple `managed_trades_*.csv` files; pick the one you want from the sidebar.
- Set environment variable `STARTING_CAPITAL` to change the base capital shown in the equity curve.
""")
