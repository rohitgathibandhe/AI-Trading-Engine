# test_dhan_wrapper.py
from market_ai.modules.data_fetch import dhan_wrapper as dw

print("CLIENT_ID:", dw.CLIENT_ID is not None)
print("ACCESS_TOKEN set:", dw.ACCESS_TOKEN is not None)

print("Expiry list (underlying 13):")
exps = dw.get_expiry_list(13)
print(" ->", exps)

if exps:
    print("Option chain sample (first expiry):")
    oc = dw.get_option_chain(13, exps[0])
    print("Keys top-level:", list(oc.keys())[:10] if oc else None)

    print("Try price sample from chain (first strike if available):")
    # attempt to pull a single strike
    if isinstance(oc, dict) and oc:
        # many chains have oc->strike -> {ce, pe}
        oc_container = oc.get("oc") or oc
        if isinstance(oc_container, dict):
            first_strike = next(iter(oc_container.keys()))
            print("First strike:", first_strike)
            price_ce = dw.get_option_price(13, exps[0], float(first_strike), "CE")
            price_pe = dw.get_option_price(13, exps[0], float(first_strike), "PE")
            print("CE price:", price_ce, "PE price:", price_pe)
