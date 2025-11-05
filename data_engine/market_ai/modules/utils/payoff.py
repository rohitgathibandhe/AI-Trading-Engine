# market_ai/modules/utils/payoff.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def plot_strangle_payoff(spot: float,
                         ce_strike: float,
                         pe_strike: float,
                         ce_premium: float,
                         pe_premium: float,
                         lots: int = 1,
                         lot_size: int = 75,
                         title: str = "Short Strangle Payoff",
                         save_path: str = None) -> Tuple[float, float]:
    """
    Plot payoff for a short strangle at expiry.
    Returns (breakeven_low, breakeven_high).
      - ce_strike: strike where call is sold (higher than spot)
      - pe_strike: strike where put is sold (lower than spot)
      - ce_premium, pe_premium: premiums received per unit
    """

    units = lots * lot_size
    # range: from 60% spot to 140% spot (broad)
    left = spot * 0.6
    right = spot * 1.4
    xs = np.linspace(left, right, 800)

    # payoff per unit at expiry for short call = -max(0, S-K) + premium
    ce_payoff_unit = np.array([(-max(0.0, s - ce_strike) + ce_premium) for s in xs])
    pe_payoff_unit = np.array([(-max(0.0, pe_strike - s) + pe_premium) for s in xs])

    total_unit = ce_payoff_unit + pe_payoff_unit
    total = total_unit * units

    plt.figure(figsize=(10, 6))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.plot(xs, total, label=f"Short Strangle ({lots} lots × {lot_size} units)", linewidth=2)
    # annotate short strikes
    plt.axvline(ce_strike, linestyle="--", label=f"CE strike {ce_strike}", color="tab:blue")
    plt.axvline(pe_strike, linestyle="--", label=f"PE strike {pe_strike}", color="tab:orange")
    plt.fill_between(xs, total, where=(total > 0), alpha=0.12, facecolor="green")
    plt.fill_between(xs, total, where=(total <= 0), alpha=0.12, facecolor="red")
    plt.title(title)
    plt.xlabel("Underlying price at expiry")
    plt.ylabel("Payoff (₹)")
    plt.legend()

    # compute breakevens: solve total_unit == 0 either side
    # to approximate, find roots in xs
    sign = np.sign(total_unit)
    idx = np.where(np.diff(sign) != 0)[0]
    breakevens = []
    for i in idx:
        # linear interpolation
        x1, x2 = xs[i], xs[i+1]
        y1, y2 = total_unit[i], total_unit[i+1]
        if (y2 - y1) != 0:
            be = x1 - y1 * (x2 - x1) / (y2 - y1)
            breakevens.append(be)
    # sort and take two (low, high)
    breakevens_sorted = sorted(breakevens)
    be_low = breakevens_sorted[0] if breakevens_sorted else None
    be_high = breakevens_sorted[-1] if breakevens_sorted else None

    # Annotate breakevens
    if be_low:
        plt.scatter([be_low], [0], color="black")
        plt.annotate(f"BE low: {be_low:.2f}", (be_low, 0), xytext=(5, -20), textcoords="offset points")
    if be_high and be_high != be_low:
        plt.scatter([be_high], [0], color="black")
        plt.annotate(f"BE high: {be_high:.2f}", (be_high, 0), xytext=(5, 5), textcoords="offset points")

    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return be_low, be_high
