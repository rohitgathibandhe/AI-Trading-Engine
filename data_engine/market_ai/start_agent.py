# start_agent.py — unified entry
from __future__ import annotations

import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import Tuple, Any
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import csv

import pandas as pd

# ───────────────────────── Path resolver (must be first) ─────────────────────
THIS_FILE = Path(__file__).resolve()                   # .../data_engine/market_ai/start_agent.py
ENGINE_DIR = THIS_FILE.parent                          # .../data_engine/market_ai
DATA_ENGINE_DIR = ENGINE_DIR.parent                    # .../data_engine
PROJECT_ROOT = DATA_ENGINE_DIR.parent                  # .../Algotrade_ai
STATE_DIR = ENGINE_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
AGENT_LOG_PATH = STATE_DIR / "agent.log"

# Your correct strategies location
PREFERRED_STRAT_DIR = ENGINE_DIR / "modules" / "strategies"

# Other places we may accept as fallback (lower priority)
ALT_STRAT_DIRS = [
    ENGINE_DIR / "strategies",
    PROJECT_ROOT / "strategies",
    PROJECT_ROOT,
]

# Optional override via env
env_dir = os.getenv("STRATEGIES_DIR")
if env_dir:
    ALT_STRAT_DIRS.insert(0, Path(env_dir).expanduser().resolve())

# Ensure our code and strategies are importable
for p in (ENGINE_DIR, DATA_ENGINE_DIR, PROJECT_ROOT, PREFERRED_STRAT_DIR, *ALT_STRAT_DIRS):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

# ───────────────────────── Logging config (timestamps) ───────────────────────
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES = int(os.getenv("AGENT_LOG_MAX_BYTES", 5 * 1024 * 1024))
LOG_BACKUP_COUNT = int(os.getenv("AGENT_LOG_BACKUP_COUNT", 5))

formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
file_handler = RotatingFileHandler(
    AGENT_LOG_PATH,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = [file_handler, console_handler]

log = logging.getLogger("start_agent")

# ───────────────────────── Helpers ──────────────────────────────────────────
def _load_module_from_path(mod_name: str, file_path: Path):
    """Import a module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {mod_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

def _lazy_symbol(file_path: Path, symbol: str):
    """Load one symbol from a file path."""
    mod = _load_module_from_path(f"dyn_{file_path.stem}", file_path)
    return getattr(mod, symbol)


def _load_agent_settings() -> dict:
    path = os.getenv("AGENT_SETTINGS_JSON")
    if not path:
        return {}
    try:
        settings_path = Path(path)
        if not settings_path.exists():
            return {}
        data = json.loads(settings_path.read_text())
        if isinstance(data, dict):
            return data
    except Exception as exc:
        log.warning("Failed to read agent settings (%s): %s", path, exc)
    return {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    try:
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)
# ───────────────────────── DhanWrapper import (robust) ───────────────────────
try:
    from data_engine.market_ai.dhan_wrapper import DhanWrapper  # type: ignore
    log.info("DhanWrapper import: data_engine.market_ai.dhan_wrapper")
except ModuleNotFoundError:
    dw_path = ENGINE_DIR / "dhan_wrapper.py"
    if not dw_path.exists():
        raise
    DhanWrapper = _lazy_symbol(dw_path, "DhanWrapper")  # type: ignore
    log.info("DhanWrapper import: file %s", dw_path)

# ───────────────────────── Strategy import (robust) ─────────────────────────
STRAT_FILENAME = "monthly_strangle_with_weekly_hedge.py"

def _import_strategy() -> Tuple[Any, Any, Any]:
    """
    Returns (run_live, LiveConfig, BatmanConfig).
    Prefers package imports; falls back to loading the .py file from known dirs.
    """
    # 1) package-style, preferred: modules/strategies
    try:
        from data_engine.market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (  # type: ignore
            run_live, LiveConfig, BatmanConfig
        )
        log.info("Strategy import: data_engine.market_ai.modules.strategies")
        return run_live, LiveConfig, BatmanConfig
    except Exception:
        pass

    # 2) older package paths
    try:
        from data_engine.market_ai.strategies.monthly_strangle_with_weekly_hedge import (  # type: ignore
            run_live, LiveConfig, BatmanConfig
        )
        log.info("Strategy import: data_engine.market_ai.strategies")
        return run_live, LiveConfig, BatmanConfig
    except Exception:
        pass
    try:
        from strategies.monthly_strangle_with_weekly_hedge import run_live, LiveConfig, BatmanConfig  # type: ignore
        log.info("Strategy import: 'strategies' on sys.path")
        return run_live, LiveConfig, BatmanConfig
    except Exception:
        pass

    # 3) file fallbacks
    search_dirs = [PREFERRED_STRAT_DIR, *ALT_STRAT_DIRS]
    for d in search_dirs:
        path = (Path(d) / STRAT_FILENAME).resolve()
        if path.exists():
            mod = _load_module_from_path("dyn_strategy", path)
            log.info("Strategy import: file '%s'", path)
            return mod.run_live, mod.LiveConfig, mod.BatmanConfig  # type: ignore[attr-defined]

    # 4) nothing found -> informative error
    lines = "\n".join(f" - {(Path(d)/STRAT_FILENAME).resolve()}" for d in search_dirs)
    raise FileNotFoundError(
        f"Could not locate {STRAT_FILENAME}. Place it at one of:\n{lines}\n"
        "Or set STRATEGIES_DIR to the folder that contains it."
    )

# Try now so any error appears in the log immediately
from market_ai.modules.strategies.strategy_selector import select_preferred_strategy
from market_ai.modules.analytics import MarketRegimeAnalyzer
from market_ai.modules.agents.strategy_recommender import StrategyRecommender

run_live, LiveConfig, BatmanConfig = _import_strategy()
strategy_module = sys.modules.get(run_live.__module__)
if strategy_module is None or not hasattr(strategy_module, "FEATURE_FIELDS"):
    FEATURE_FIELDS = [
        "timestamp",
        "strategy",
        "expiry",
        "exchange_seg",
        "spot",
        "ce_strike",
        "pe_strike",
        "ce_ltp",
        "pe_ltp",
        "ce_delta",
        "pe_delta",
        "ce_entry",
        "pe_entry",
        "net_credit",
        "warn_only",
        "trade_mode",
        "context",
    ]
else:
    FEATURE_FIELDS = list(getattr(strategy_module, "FEATURE_FIELDS"))

if strategy_module is None or not hasattr(strategy_module, "BLOTTER_FIELDS"):
    BLOTTER_FIELDS = [
        "timestamp",
        "trade_mode",
        "warn_only",
        "executed",
        "side",
        "order_type",
        "exchange_seg",
        "product_type",
        "security_id",
        "quantity",
        "price",
        "strike",
        "tag",
        "notes",
    ]
else:
    BLOTTER_FIELDS = list(getattr(strategy_module, "BLOTTER_FIELDS"))


def _ensure_csv_file(path: Path, headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()

# ───────────────────────── Optional adaptive agent support ───────────────────
def _maybe_run_adaptive() -> bool:
    """
    If AGENT_MODE=adaptive, attempt to load agents/adaptive_agent.py and run it.
    Returns True if adaptive was started, else False.
    """
    if os.getenv("AGENT_MODE", "strategy").lower() != "adaptive":
        return False

    # Locate file
    cand1 = ENGINE_DIR / "agents" / "adaptive_agent.py"
    cand2 = ENGINE_DIR / "agent" / "adaptive_agent.py"
    agent_file = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)
    if agent_file is None:
        hits = list(PROJECT_ROOT.rglob("adaptive_agent.py"))
        if hits:
            # prefer a file under market_ai if present
            agent_file = next((h for h in hits if "market_ai" in str(h)), hits[0])
    if agent_file is None:
        raise FileNotFoundError("AGENT_MODE=adaptive but 'adaptive_agent.py' not found.")

    log.info("Loading adaptive agent from: %s", agent_file)
    mod = _load_module_from_path("adaptive_agent_mod", agent_file)
    AdaptiveAgent = getattr(mod, "AdaptiveAgent")
    AgentConfig   = getattr(mod, "AgentConfig")

    state_dir = ENGINE_DIR / "state"
    control_dir = ENGINE_DIR / "control"
    logs_dir = ENGINE_DIR / "logs"
    for d in (state_dir, control_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = AgentConfig(
        state_path=str(state_dir / "trade_state.json"),
        control_dir=str(control_dir),
    )
    risk_cfg = {
        "max_loss_pct_trade": 0.015,
        "max_loss_pct_day": 0.025,
        "ivpct_min": 10,
        "ivpct_max": 85,
        "vix_min": 10.0,
        "vix_max": 25.0,
        "gap_thr_pct": 0.5,
        "default_account_equity": 500000.0,
    }
    AdaptiveAgent(cfg, risk_cfg).start()
    return True

# ───────────────────────── Main (strategy mode by default) ───────────────────
def main() -> None:
    if _maybe_run_adaptive():
        return

    # Strategy mode (default)
    # DhanWrapper gets DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN from environment
    dw = DhanWrapper()

    settings = _load_agent_settings()

    def pick(env_name: str, setting_name: str, default: Any, caster) -> Any:
        if env_name in os.environ:
            return caster(os.environ[env_name], default)
        if setting_name in settings:
            return caster(settings[setting_name], default)
        return default

    def pick_bool(env_name: str, setting_name: str, default: bool) -> bool:
        if env_name in os.environ:
            return _as_bool(os.environ[env_name], default)
        if setting_name in settings:
            return _as_bool(settings[setting_name], default)
        return default

    blotter_path = os.getenv("BLOTTER_PATH", str(ENGINE_DIR / "state" / "trade_blotter.csv"))
    summary_path = os.getenv("BLOTTER_SUMMARY_PATH", str(ENGINE_DIR / "state" / "trade_blotter_summary.json"))
    trade_mode = os.getenv("TRADE_MODE", settings.get("trade_mode", "live"))
    feature_log = os.getenv("FEATURE_LOG_PATH", settings.get("feature_log_path", str(ENGINE_DIR / "state" / "feature_history.csv")))

    blotter_path_obj = Path(blotter_path)
    _ensure_csv_file(blotter_path_obj, BLOTTER_FIELDS)

    cfg = LiveConfig(
        max_strangles=pick("MAX_STRANGLES", "max_legs", 1, _as_int),
        lot_size=pick("NIFTY_LOT", "lot_size", 75, _as_int),
        sl_pct=pick("SL_PCT", "leg_sl_pct", 0.025, _as_float),
        tp_pct=pick("TP_PCT", "profit_pct", 0.0225, _as_float),
        hedge_enable=pick_bool("HEDGE_ENABLE", "hedge_enable", True),
        hedge_price_min=pick("HEDGE_PRICE_MIN", "hedge_price_min", 2.5, _as_float),
        hedge_price_max=pick("HEDGE_PRICE_MAX", "hedge_price_max", 3.5, _as_float),
        poll_sec=pick("POLL_SEC", "poll_sec", 5.0, _as_float),
        warn_only=pick_bool("WARN_ONLY", "warn_only", False),
        trade_mode=trade_mode,
        blotter_path=blotter_path,
        blotter_summary_path=summary_path,
        batman=BatmanConfig(
            enabled=pick_bool("BATMAN_ENABLED", "batman_enabled", True),
            delta_breach=pick("BATMAN_DELTA_BREACH", "batman_delta_breach", 0.30, _as_float),
            premium_hard_x=pick("BATMAN_PREMIUM_HARD_X", "batman_premium_hard_x", 2.0, _as_float),
            roll_distance=pick("BATMAN_ROLL_DISTANCE", "batman_roll_distance", 150.0, _as_float),
            hedge_delta_max=pick("BATMAN_HEDGE_DELTA_MAX", "batman_hedge_delta_max", 0.12, _as_float),
            hedge_price_max=pick("BATMAN_HEDGE_PRICE_MAX", "batman_hedge_price_max", 35.0, _as_float),
            salvage_wing_ltp=pick("BATMAN_SALVAGE_WING_LTP", "batman_salvage_wing_ltp", 5.0, _as_float),
        ),
        feature_log_path=feature_log,
    )
    cfg.regime_refresh_minutes = pick("REGIME_REFRESH_MINUTES", "regime_refresh_minutes", cfg.regime_refresh_minutes, _as_float)
    cfg.auto_disable_strangle_when_unfavored = pick_bool(
        "AUTO_DISABLE_STRANGLE",
        "auto_disable_strangle",
        cfg.auto_disable_strangle_when_unfavored,
    )

    selector_model_path = Path(
        os.getenv(
            "STRATEGY_MODEL_PATH",
            settings.get("strategy_model_path", str(ENGINE_DIR / "state" / "strategy_selector_model.json")),
        )
    )
    feature_path = Path(feature_log)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if not feature_path.exists():
        header = ",".join(FEATURE_FIELDS) + "\n"
        feature_path.write_text(header)

    lookback_minutes = pick("REGIME_LOOKBACK_MINUTES", "regime_lookback_minutes", 180.0, _as_float)
    regime_analyzer = MarketRegimeAnalyzer(lookback=timedelta(minutes=float(lookback_minutes)))
    recommender = StrategyRecommender(
        selector_model_path if selector_model_path.exists() else None,
        feature_path,
    )

    initial_state = None
    if feature_path.exists() and feature_path.stat().st_size > 0:
        try:
            history_df = pd.read_csv(feature_path).tail(600)
            initial_state = regime_analyzer.analyze_feature_history(history_df)
        except Exception as exc:
            log.debug("Failed to compute initial market state: %s", exc)

    if initial_state is None and feature_path.exists() and feature_path.stat().st_size == 0:
        spot_val = _as_float(dw.get_ltp_once("IDX_I", 13), 0.0)
        spot = spot_val if spot_val > 0 else None
        if spot is not None:
            history_df = pd.DataFrame([
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "spot": spot,
                }
            ])
            initial_state = regime_analyzer.analyze_feature_history(history_df, min_required=1)

    if initial_state:
        setattr(cfg, "_last_market_state", initial_state)
        candidates = recommender.recommend(initial_state)
        setattr(cfg, "_last_strategy_candidates", candidates)
        if candidates:
            top = candidates[0]
            log.info(
                "Initial environment trend=%s vol=%s -> top strategy=%s score=%.2f",
                initial_state.trend,
                initial_state.volatility,
                top.name,
                top.score,
            )
            if cfg.auto_disable_strangle_when_unfavored:
                cfg.enable_strangle_entry = top.name == "monthly_strangle_with_weekly_hedge"
    else:
        if selector_model_path.exists():
            recommendation = select_preferred_strategy(selector_model_path, feature_path)
        else:
            recommendation = None
        if recommendation:
            log.info(
                "Selector recommendation: %s (expected_credit=%.2f, samples=%d, confidence=%.2f)",
                recommendation.name,
                recommendation.expected_credit,
                recommendation.samples,
                recommendation.confidence,
            )
            if recommendation.name.lower() == "strangle":
                cfg.enable_strangle_entry = True
            elif recommendation.name.lower() == "batman" and cfg.auto_disable_strangle_when_unfavored:
                cfg.enable_strangle_entry = False
                if not cfg.batman.enabled:
                    log.warning("Selector favours Batman but it is disabled via settings")
        else:
            log.info("Strategy selector model unavailable; using configured defaults")

    _debug_positions_once(dw)
    log.info("Starting strategy run_live with cfg=%s", cfg)
    run_live(
        dw,
        cfg,
        regime_analyzer=regime_analyzer,
        recommender=recommender,
        feature_history_path=feature_path,
    )


def _debug_positions_once(dw) -> None:
    """Log the raw type/shape of broker positions once on startup for diagnostics."""
    try:
        raw = dw.get_positions_live()
        t = type(raw).__name__
        log.info("[start_agent] dw.get_positions_live() -> %s", t)
        if isinstance(raw, list) and raw and not isinstance(raw[0], dict):
            log.warning("[start_agent] first row type is %s ; sample=%r", type(raw[0]).__name__, raw[0])
        if isinstance(raw, str):
            snippet = raw[:300] + ("..." if len(raw) > 300 else "")
            log.warning("[start_agent] got string response: %r", snippet)
    except Exception:
        log.exception("[start_agent] probe positions failed")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Agent crashed with an unhandled exception")
        raise
