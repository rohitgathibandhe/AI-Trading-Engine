import dataclasses

import pytest

from market_ai.intraday_defined_risk.data_models import DecisionOutput


def test_decision_output_is_frozen_so_clamp_must_use_replace():
    fields = {f.name: None for f in dataclasses.fields(DecisionOutput)}
    d = DecisionOutput.__new__(DecisionOutput)
    for k in fields:
        object.__setattr__(d, k, None)
    object.__setattr__(d, "lots", 9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.lots = 6                                   # the old monitor code -> crashed run_live
    assert dataclasses.replace(d, lots=6).lots == 6  # what monitor does now


def test_monitor_no_longer_assigns_to_frozen_decision():
    import inspect
    from market_ai.intraday_defined_risk import monitor
    src = inspect.getsource(monitor.run_live)
    assert "decision.lots = ops_max" not in src


def test_clamp_rescales_leg_quantities():
    import inspect
    from market_ai.intraday_defined_risk import monitor
    src = inspect.getsource(monitor.run_live)
    assert "quantity=int(round(float(l.get(\"quantity\") or 0) * ops_max / _old))" in src
