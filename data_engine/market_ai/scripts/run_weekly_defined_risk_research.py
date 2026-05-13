from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from market_ai.weekly_defined_risk import WeeklyResearchConfig, run_weekly_defined_risk_research


def main() -> int:
    parser = argparse.ArgumentParser(description="Run weekly NIFTY defined-risk research backtest.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to the historical intraday research dataset.")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory for weekly research artifacts.")
    parser.add_argument("--output-tag", type=str, default=None, help="Tag appended to generated artifact file names.")
    parser.add_argument("--allowed-strategy", action="append", default=None, help="Limit research to one or more specific strategy names.")
    parser.add_argument("--allowed-regime", action="append", default=None, help="Limit research to one or more specific regime labels.")
    parser.add_argument("--research-variant", type=str, default=None, help="Optional label stored in the output config.")
    args = parser.parse_args()

    config = WeeklyResearchConfig()
    if args.dataset_root is not None:
        config.dataset_root = args.dataset_root
    if args.output_root is not None:
        config.output_root = args.output_root
    if args.output_tag is not None:
        config.output_tag = args.output_tag
    if args.allowed_strategy:
        config.allowed_strategies = tuple(args.allowed_strategy)
    if args.allowed_regime:
        config.allowed_regimes = tuple(args.allowed_regime)
    if args.research_variant:
        config.research_variant = args.research_variant

    result = run_weekly_defined_risk_research(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
