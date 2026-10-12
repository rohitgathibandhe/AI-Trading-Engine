from market_ai.intraday_defined_risk import strategy_selector as ss


def test_level_inside_a_failed_fade_zone_is_broken_same_day_only():
    ss._BROKEN_FADE_ZONES.clear()
    ss.record_broken_fade("2026-10-12", "FADE_SHORT", entry_spot=22376.0, stop_spot=22390.0)
    assert ss._level_broken_today("2026-10-12", "FADE_SHORT", 22382.0)       # the 10-12 re-fade
    assert not ss._level_broken_today("2026-10-12", "FADE_LONG", 22382.0)    # other side unaffected
    assert not ss._level_broken_today("2026-10-13", "FADE_SHORT", 22382.0)   # next session is fresh
    assert not ss._level_broken_today("2026-10-12", "FADE_SHORT", 22500.0)   # a different level is fine
    ss._BROKEN_FADE_ZONES.clear()
