"""
modules/utils/config_loader.py
Loads all configuration YAMLs for the trading agent.
"""

import os
import yaml
from typing import Any, Dict

def load_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def load_all_configs(base_dir: str = "config") -> Dict[str, Any]:
    """
    Loads and merges base_config.yaml + risk_config.yaml + broker_config.yaml
    and strategy-specific YAMLs.
    Returns dict with structure:
    {
        'base': {...},
        'risk': {...},
        'broker': {...},
        'strategies': {'iron_condor': {...}, 'hedged_strangle': {...}, ...}
    }
    """
    configs = {}
    configs['base'] = load_yaml(os.path.join(base_dir, "base_config.yaml"))
    configs['risk'] = load_yaml(os.path.join(base_dir, "risk_config.yaml"))

    broker_cfg = os.path.join(base_dir, "broker_config.yaml")
    configs['broker'] = load_yaml(broker_cfg) if os.path.exists(broker_cfg) else {}

    # Load strategy-specific configs
    strategies_dir = os.path.join(base_dir, "strategies")
    configs['strategies'] = {}
    if os.path.isdir(strategies_dir):
        for f in os.listdir(strategies_dir):
            if f.endswith(".yaml"):
                name = f.replace(".yaml", "")
                configs['strategies'][name] = load_yaml(os.path.join(strategies_dir, f))

    return configs
