#!/usr/bin/env python3
"""
tmp_inspect_chain_test.py

Usage:
  python3 tmp_inspect_chain_test.py --uid 26000 --seg NSE_FNO --debug

This will attempt to call expiry_list(underlying, seg) and then option_chain(underlying, seg, expiry).
It expects your module to be importable as market_ai.modules.data_fetch.dhan_option_chain
and your DHAN credentials to be in env DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID (if needed by your module).
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import traceback

def find_module_candidate():
    names = [
        "market_ai.modules.data_fetch.dhan_option_chain",
        "market_ai.modules.data_fetch.dhan_optionchain",
        "modules.data_fetch.dhan_option_chain",
        "market_ai.modules.data_fetch.dhan_api",
        "modules.data_fetch.dhan_optionchain",
    ]
    for n in names:
        try:
            m = importlib.import_module(n)
            return m, n
        except Exception as e:
            # continue
            pass
    return None, None

def make_wrapper_if_needed(mod):
    """
    If module is functional (provides get_expiry_list / get_option_chain), return a thin wrapper
    with methods expiry_list(underlying, seg) and option_chain(underlying, seg, expiry).
    Otherwise return object/class/instance found in module that looks like an OptionChain client.
    """
    # prefer explicit class names
    for candidate in ("OptionChain", "DhanOptionChain", "OptionChainClient"):
        obj = getattr(mod, candidate, None)
        if obj is not None:
            # if class, instantiate; if instance, return
            try:
                return obj() if isinstance(obj, type) else obj
            except Exception:
                return obj

    # fallback to module-level functions
    get_expiry = getattr(mod, "get_expiry_list", None) or getattr(mod, "expiry_list", None)
    get_option_chain = getattr(mod, "get_option_chain", None) or getattr(mod, "option_chain", None)
    get_option_price = getattr(mod, "get_option_price", None)
    if callable(get_expiry) and callable(get_option_chain):
        class _Wrapper:
            def __init__(self, module):
                self._mod = module
            def expiry_list(self, uid, seg):
                return get_expiry(uid, seg)
            def option_chain(self, uid, seg, expiry):
                return get_option_chain(uid, seg, expiry)
            def option_price(self, *a, **kw):
                if callable(get_option_price):
                    return get_option_price(*a, **kw)
                raise AttributeError("no option_price")
        return _Wrapper(mod)

    # nothing usable found
    return None

def pretty_print_small_chain(resp):
    # resp expected to be dict with 'data' maybe nested
    if not isinstance(resp, dict):
        print("Option chain response is not a dict:", type(resp))
        print(resp)
        return
    print("--- option chain top-level keys ---")
    print(list(resp.keys()))
    data = resp.get("data") if "data" in resp else resp
    if not isinstance(data, dict):
        print("Option chain 'data' not a dict; printing whole:")
        print(json.dumps(resp, indent=2)[:2000])
        return
    strikes = sorted([k for k in data.keys() if k not in ("status",)], key=lambda x: float(x) if x.replace('.','',1).isdigit() else x)
    print("strikes_count:", len(strikes))
    sample = strikes[:5] if len(strikes) > 5 else strikes
    for s in sample:
        item = data.get(s)
        print(f"strike: {s} -> keys: {list(item.keys()) if isinstance(item, dict) else type(item)}")
        if isinstance(item, dict):
            ce = item.get("ce")
            pe = item.get("pe")
            if ce:
                print("  CE: last_price:", ce.get("last_price"), "iv:", ce.get("implied_volatility"), "delta:", ce.get("greeks",{}).get("delta"))
            if pe:
                print("  PE: last_price:", pe.get("last_price"), "iv:", pe.get("implied_volatility"), "delta:", pe.get("greeks",{}).get("delta"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uid", type=int, default=26000, help="Dhan underlying id (example: 26000 for NIFTY)")
    p.add_argument("--seg", type=str, default="NSE_FNO", help="segment (NSE_FNO / NSE_EQ / IDX_I / NSE_INDEX etc)")
    p.add_argument("--expiry", type=str, default=None, help="explicit expiry (optional), format YYYY-MM-DD")
    p.add_argument("--debug", action="store_true", help="debug prints")
    args = p.parse_args()

    # show environment vars
    if args.debug:
        print("ENV DHAN_ACCESS_TOKEN:", bool(os.environ.get("DHAN_ACCESS_TOKEN")))
        print("ENV DHAN_CLIENT_ID:", bool(os.environ.get("DHAN_CLIENT_ID")))

    mod, modname = find_module_candidate()
    if mod is None:
        print("Could not import any dhan optionchain module. Tried several names. Please ensure module is importable.")
        sys.exit(2)
    print("Imported module:", modname)

    client = make_wrapper_if_needed(mod)
    if client is None:
        print("Module found but no usable OptionChain client or functions were detected. Exports:", [n for n in dir(mod) if not n.startswith("_")])
        sys.exit(3)

    # call expiry list
    try:
        print(f"Calling expiry_list(uid={args.uid}, seg={args.seg}) ...")
        expiries_resp = client.expiry_list(args.uid, args.seg)
        if args.debug:
            print("expiry_list raw:", type(expiries_resp))
        # handle module that returns dict {'data': [...]}
        expiries = None
        if isinstance(expiries_resp, dict) and "data" in expiries_resp:
            expiries = expiries_resp["data"]
        elif isinstance(expiries_resp, (list, tuple)):
            expiries = list(expiries_resp)
        elif isinstance(expiries_resp, dict):
            expiries = expiries_resp.get("data") or []
        else:
            expiries = []
        print("Expiry list (first 10):", expiries[:10])
    except Exception as e:
        print("Expiry call raised:", repr(e))
        traceback.print_exc()
        sys.exit(4)

    if not expiries:
        print("No expiries returned. The API might have returned a 'failed' status or invalid uid/segment.")
        sys.exit(5)

    # choose expiry to fetch
    chosen_expiry = args.expiry or (expiries[0] if expiries else None)
    print("Chosen expiry:", chosen_expiry)

    try:
        print(f"Calling option_chain(uid={args.uid}, seg={args.seg}, expiry={chosen_expiry}) ...")
        oc_resp = client.option_chain(args.uid, args.seg, chosen_expiry)
        if args.debug:
            print("option_chain raw (top):", type(oc_resp))
        pretty_print_small_chain(oc_resp)
    except Exception as e:
        print("option_chain call raised:", repr(e))
        traceback.print_exc()
        sys.exit(6)

if __name__ == "__main__":
    main()
