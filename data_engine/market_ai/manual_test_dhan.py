# market_ai/manual_test_dhan.py
"""
Quick manual test to exercise modules.data_fetch.dhan_option_chain
Run with: python -m market_ai.manual_test_dhan
"""

import os
import json
import logging

# set up logging so we can see helpful DEBUG messages from the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from market_ai.modules.data_fetch.dhan_option_chain import get_expiry_list, get_option_chain

def main():
    scrip = int(os.environ.get("TEST_UNDERLYING_SCRIP", 13))
    seg = os.environ.get("TEST_UNDERLYING_SEG", "IDX_I")

    print("DHAN_CLIENT_ID present?", bool(os.environ.get("DHAN_CLIENT_ID")))
    print("DHAN_ACCESS_TOKEN present?", bool(os.environ.get("DHAN_ACCESS_TOKEN")))

    # 1) expiry list
    try:
        exp = get_expiry_list(UnderlyingScrip=scrip, UnderlyingSeg=seg, use_cache=False)
        print("Expiry list sample:", exp.get("data")[:6])
    except Exception as e:
        print("Failed to fetch expiry list:", repr(e))
        raise

    # choose first expiry if available
    expiry = None
    if isinstance(exp, dict) and exp.get("data"):
        expiry = exp["data"][0]
        print("Using expiry:", expiry)

    # 2) fetch chain using expiry (or None to see error help)
    try:
        chain = get_option_chain(UnderlyingScrip=scrip, UnderlyingSeg=seg, Expiry=expiry, use_cache=False)
        print("Option chain top-level keys:", list(chain.keys()))
        print("Underlying LTP:", chain.get("data", {}).get("last_price"))
        strikes = list(chain.get("data", {}).get("oc", {}).keys())[:6]
        print("Sample strikes:", strikes)
        if strikes:
            print("Example strike (first) snippet:", json.dumps(chain["data"]["oc"][strikes[0]])[:800])
    except Exception as e:
        print("Failed to fetch option chain:", repr(e))
        raise

if __name__ == "__main__":
    main()
