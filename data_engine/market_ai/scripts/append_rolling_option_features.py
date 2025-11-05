#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate feature_history entries from rolling option parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path

from market_ai.modules.training.feature_dataset import append_to_feature_history
from market_ai.modules.training.rolling_option_features import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append rolling option features to feature_history.csv")
    parser.add_argument("rolling_option_dir", type=Path)
    parser.add_argument("feature_history", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(args.rolling_option_dir)
    if df.empty:
        print("No rolling option features produced; nothing to append")
        return
    append_to_feature_history(df, args.feature_history)
    print(f"Appended {len(df)} rows to {args.feature_history}")


if __name__ == "__main__":
    main()
