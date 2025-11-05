# tools/plot_payoff.py
import numpy as np
import matplotlib.pyplot as plt

def payoff_short_strangle(spot_grid, ce_strike, pe_strike, ce_prem, pe_prem, units=75):
    # short CE: premium received = +ce_prem*units ; payoff = -max(spot-ce_strike,0)*units + premium
    # short PE: payoff = -max(pe_strike-spot,0)*units + premium
    total = np.zeros_like(spot_grid, dtype=float)
    total += (-np.maximum(spot_grid - ce_strike, 0.0) * units + ce_prem * units)
    total += (-np.maximum(pe_strike - spot_grid, 0.0) * units + pe_prem * units)
    return total

def payoff_long_option(spot_grid, strike, premium, side='ce', units=75):
    if side=='ce':
        return np.maximum(spot_grid - strike, 0.0) * units - premium * units
    else:
        return np.maximum(strike - spot_grid, 0.0) * units - premium * units

if __name__ == "__main__":
    spot0 = 30000
    grid = np.linspace(spot0*0.7, spot0*1.3, 601)
    ce_strike = 30500
    pe_strike = 29500
    ce_prem = 100.0
    pe_prem = 90.0
    units = 75

    base = payoff_short_strangle(grid, ce_strike, pe_strike, ce_prem, pe_prem, units=units)
    # optional weekly hedge (long a cheap call)
    hedge = payoff_long_option(grid, 30700, premium=3.0, side='ce', units=units)

    total_with_hedge = base + hedge

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(grid, base, label='Short strangle (no hedge)')
    plt.plot(grid, total_with_hedge, label='With weekly hedge (long CE)')
    plt.axvline(ce_strike, color='r', linestyle='--', alpha=0.6)
    plt.axvline(pe_strike, color='g', linestyle='--', alpha=0.6)
    plt.title('Payoff at expiry')
    plt.xlabel('Spot at expiry')
    plt.ylabel('P/L (INR)')
    plt.legend()
    plt.grid(True)
    plt.show()
