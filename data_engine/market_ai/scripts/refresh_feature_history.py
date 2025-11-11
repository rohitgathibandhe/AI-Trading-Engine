#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refresh the feature store by combining rolling-option backfill data with
existing agent feature history and emit engineered datasets for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ENGINE_DIR = SCRIPT_PATH.parents[1]
PACKAGE_ROOT = ENGINE_DIR.parents[0]

if __package__ is None:
    import sys
    sys.path.insert(0, str(PACKAGE_ROOT))

from market_ai.modules.training.feature_dataset import (  # noqa: E402
    FEATURE_HISTORY_DEFAULT,
    engineer_features,
    load_feature_history,
    make_training_target,
    summarise_dataset,
)
from market_ai.modules.training.rolling_option_features import build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh feature history with rolling-option data.")
    parser.add_argument(
        "--rolling-root",
        type=Path,
        default=ENGINE_DIR / "state" / "rolling_option",
        help="Directory containing YYYY-MM-DD/rolling_option.parquet folders.",
    )
    parser.add_argument(
        "--feature-history",
        type=Path,
        default=FEATURE_HISTORY_DEFAULT,
        help="Feature history CSV to refresh (default state/feature_history.csv).",
    )
    parser.add_argument(
        "--engineered-out",
        type=Path,
        default=ENGINE_DIR / "state" / "feature_history_engineered.parquet",
        help="Path to store engineered feature matrix (Parquet).",
    )
    parser.add_argument(
        "--target-out",
        type=Path,
        default=ENGINE_DIR / "state" / "feature_targets.parquet",
        help="Path to store training targets (Parquet).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ENGINE_DIR / "state" / "feature_history_summary.json",
        help="Where to write dataset summary JSON.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Target horizon (rows) for provisional label generation.",
    )
    parser.add_argument(
        "--context",
        default="rollingoption",
        help="Context tag for rolling-option derived samples.",
    )
    return parser.parse_args()


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    args = parse_args()

    # 1) Build dataset from rolling-option parquet folders
    if not args.rolling_root.exists():
        raise SystemExit(f"Rolling-option dir not found: {args.rolling_root}")
    print(f"[features] scanning rolling-option root: {args.rolling_root}")
    rolling_df = build_dataset(args.rolling_root)
    if rolling_df.empty:
        raise SystemExit("No rolling-option samples available; run fetch_rolling_option first.")
    rolling_df["context"] = args.context
    rolling_df["trade_mode"] = "historical"
    rolling_df["warn_only"] = 0
    rolling_df = _normalize_timestamp(rolling_df)

    # 2) Load existing feature history (if any) and combine
    existing_df = load_feature_history(args.feature_history)
    combined = pd.concat([existing_df, rolling_df], ignore_index=True)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.dropna(subset=["timestamp"])
    subset_cols = [c for c in ["timestamp", "strategy", "context", "ce_strike", "pe_strike"] if c in combined.columns]
    if subset_cols:
        combined = combined.drop_duplicates(subset=subset_cols)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "warn_only" in combined.columns:
        combined["warn_only"] = (
            combined["warn_only"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
            .fillna(0)
            .astype(int)
        )
    combined.to_csv(args.feature_history, index=False)
    print(f"[features] refreshed feature_history.csv rows={len(combined):,}")

    # 3) Engineer features + targets for training artefacts
    engineered = engineer_features(combined)
    engineered.to_parquet(args.engineered_out, index=False)
    train_df, target = make_training_target(engineered, horizon=args.horizon)
    target_df = pd.DataFrame({"timestamp": train_df["timestamp"], "target": target})
    target_df.to_parquet(args.target_out, index=False)
    summary_input = combined.copy()
    if "timestamp" in summary_input.columns:
        summary_input["timestamp"] = pd.to_datetime(summary_input["timestamp"], errors="coerce")
    summary = summarise_dataset(summary_input)
    summary.update(
        {
            "engineered_rows": int(len(engineered)),
            "training_rows": int(len(train_df)),
            "target_horizon": args.horizon,
            "rolling_samples": int(len(rolling_df)),
            "existing_samples": int(len(existing_df)),
            "context": args.context,
        }
    )
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(f"[features] engineered dataset -> {args.engineered_out} | targets -> {args.target_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
