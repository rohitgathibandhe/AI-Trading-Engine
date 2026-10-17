from market_ai.dhan_wrapper import _DefaultLogger


def test_default_logger_has_every_level_the_wrapper_uses():
    log = _DefaultLogger()
    for level in ("debug", "info", "warning", "error", "exception"):
        getattr(log, level)("msg %s", 1)
