from typing import Dict, Any, Optional, Tuple
import pandas as pd

def compute_pcr(df: pd.DataFrame) -> Dict[str, float]:
    """
    Expects columns: oi_ce, oi_pe, vol_ce, vol_pe (already normalized by StrategySelector if needed).
    """
    p_oi = float(df["oi_pe"].sum()) if "oi_pe" in df else 0.0
    c_oi = float(df["oi_ce"].sum()) if "oi_ce" in df else 0.0
    p_vol = float(df["vol_pe"].sum()) if "vol_pe" in df else 0.0
    c_vol = float(df["vol_ce"].sum()) if "vol_ce" in df else 0.0

    pcr_oi = (p_oi / c_oi) if c_oi > 0 else 1.0
    pcr_vol = (p_vol / c_vol) if c_vol > 0 else 1.0
    return {"pcr_oi": pcr_oi, "pcr_vol": pcr_vol}

def symmetry_check(ce_price: float, pe_price: float, max_misprice_pct: float) -> bool:
    """
    Require CE and PE premiums to be reasonably symmetric around ATM.
    """
    if ce_price <= 0 or pe_price <= 0:
        return False
    mid = (ce_price + pe_price) / 2.0
    mis_ce = abs(ce_price - mid) / mid * 100.0
    mis_pe = abs(pe_price - mid) / mid * 100.0
    return (mis_ce <= max_misprice_pct) and (mis_pe <= max_misprice_pct)

def _pick_by_delta(df: pd.DataFrame, col_delta: str, target_abs: float, tol: float, min_abs: float, side: str) -> Optional[dict]:
    """
    Chooses a single row where |delta| is close to target_abs within ±tol, and ≥ min_abs.
    For CE, delta is positive; for PE, delta is negative.
    """
    if col_delta not in df.columns:
        return None
    d = df[[col_delta, "strike"] + [c for c in df.columns if c.startswith("ltp_")]].dropna()
    if d.empty:
        return None

    # enforce sign & floor
    if side == "CE":
        d = d[d[col_delta] >= min_abs]
    else:
        d = d[d[col_delta] <= -min_abs]
    if d.empty:
        return None

    # distance to target
    target = target_abs if side == "CE" else -target_abs
    d["delta_dist"] = (d[col_delta] - target).abs()
    d = d.sort_values("delta_dist")
    top = d.iloc[0].to_dict()
    # Make sure within tolerance
    if abs(top[col_delta] - target) > tol:
        return None
    return top

def pick_delta_strikes(df: pd.DataFrame, target_abs_delta: float, tol: float, min_abs_delta: float) -> Optional[Tuple[dict, dict]]:
    """
    Return (CE_row, PE_row) dicts with keys: strike, ltp_ce/ltp_pe, delta_ce/delta_pe
    """
    ce = _pick_by_delta(df, "delta_ce", target_abs_delta, tol, min_abs_delta, side="CE")
    pe = _pick_by_delta(df, "delta_pe", target_abs_delta, tol, min_abs_delta, side="PE")
    if not ce or not pe:
        return None

    # Normalize expected keys
    ce_out = {"strike": float(ce["strike"]), "ltp_ce": float(ce.get("ltp_ce", 0.0)), "delta_ce": float(ce["delta_ce"])}
    pe_out = {"strike": float(pe["strike"]), "ltp_pe": float(pe.get("ltp_pe", 0.0)), "delta_pe": float(pe["delta_pe"])}
    return ce_out, pe_out
