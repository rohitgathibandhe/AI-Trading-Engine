from market_ai.intraday_defined_risk.live_runtime import _chain_last_price, _chain_price_mismatch


def _payload(price):
    return {"status": "success", "data": {"data": {"last_price": price, "oc": {}}}}


def test_wrong_underlying_chain_is_rejected():
    assert _chain_price_mismatch(_payload(6914), 22729.3)


def test_matching_chain_is_kept():
    assert not _chain_price_mismatch(_payload(22710.75), 22729.3)


def test_missing_price_is_not_treated_as_a_mismatch():
    assert _chain_last_price({"data": {}}) is None
    assert not _chain_price_mismatch({"data": {}}, 22729.3)
    assert not _chain_price_mismatch(_payload(6914), 0.0)
