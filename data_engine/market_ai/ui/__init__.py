def _init_state():
    ss = st.session_state
    ss.setdefault("client_id", os.getenv("DHAN_CLIENT_ID", ""))
    ss.setdefault("access_token", os.getenv("DHAN_ACCESS_TOKEN", ""))
    ss.setdefault("trade_mode", "paper")  # 'paper' or 'live'
    ss.setdefault("virtual_cash", 500000.0)
    ss.setdefault("selected_indices", ["NIFTY"])
    ss.setdefault("agent_running", False)
    ss.setdefault("agent_pid", None)
    ss.setdefault("refresh_sec", 5)
