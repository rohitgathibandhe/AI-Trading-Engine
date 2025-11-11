#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy selector training CLI.

Loads `state/feature_history.csv` (optionally augmented with rolling-option
backfills), engineers features, fits simple per-strategy ridge regressions,
and writes both the selector model JSON and an evaluation summary.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ENGINE_DIR = SCRIPT_PATH.parents[1]
PACKAGE_ROOT = ENGINE_DIR.parents[0]

if __package__ is None:
    import sys
    sys.path.insert(0, str(PACKAGE_ROOT))

from market_ai.modules.training.feature_dataset import (
    FEATURE_HISTORY_DEFAULT,
    engineer_features,
    load_feature_history,
    make_training_target,
    summarise_dataset,
    DEFAULT_FEATURE_COLUMNS,
)
from market_ai.modules.training.rolling_option_features import build_dataset
from market_ai.modules.data_fetch.dhan_rolling_option import RollingOptionIngestor, RollingOptionConfig
from market_ai.modules.data_fetch.dhan_api import make_client

STATE_DIR = Path(__file__).resolve().parents[2] / "state"
MODEL_PATH_DEFAULT = STATE_DIR / "strategy_selector_model.json"
REPORT_PATH_DEFAULT = STATE_DIR / "strategy_selector_training_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy selector training scaffold")
    parser.add_argument(
        "--history",
        type=Path,
        default=FEATURE_HISTORY_DEFAULT,
        help="Path to feature_history.csv (defaults to state/feature_history.csv)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Shift horizon for provisional target creation (in samples)",
    )
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH_DEFAULT, help="Where to store the trained selector model")
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH_DEFAULT, help="Where to store evaluation summary JSON")
    parser.add_argument(
        "--rolling-option-dir",
        type=Path,
        help="Directory containing rolling_option parquet folders",
    )
    parser.add_argument(
        "--fetch-rolling-option",
        action="store_true",
        help="Fetch Dhan rolling option history before training",
    )
    parser.add_argument("--fetch-underlying", help="Underlying symbol for rolling option fetch (e.g., NIFTY)")
    parser.add_argument("--fetch-segment", default="NSE_FNO", help="Underlying segment for fetch (default NSE_FNO)")
    parser.add_argument("--fetch-security-id", type=int, help="Numeric securityId required for rolling option fetch")
    parser.add_argument("--fetch-start", help="Fetch start date YYYY-MM-DD")
    parser.add_argument("--fetch-end", help="Fetch end date YYYY-MM-DD")
    parser.add_argument("--fetch-expiry", help="Optional expiry filter YYYY-MM-DD")
    parser.add_argument("--fetch-interval", default="1", help="Rolling option interval (default 1)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction per strategy (default 0.2)")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum samples per strategy to train (default 50)")
    parser.add_argument("--strategies", nargs="*", help="Optional whitelist of strategies to train (defaults to all present)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rolling_dir = args.rolling_option_dir

    if args.fetch_rolling_option:
        if not all([args.fetch_underlying, args.fetch_security_id, args.fetch_start, args.fetch_end]):
            raise SystemExit("--fetch-underlying, --fetch-security-id, --fetch-start and --fetch-end are required when --fetch-rolling-option is used")
        rolling_dir = rolling_dir or Path(__file__).resolve().parents[2] / "state" / "rolling_option"
        rolling_dir.mkdir(parents=True, exist_ok=True)
        cfg = RollingOptionConfig(
            underlying=args.fetch_underlying,
            segment=args.fetch_segment,
            security_id=args.fetch_security_id,
            start=datetime.strptime(args.fetch_start, "%Y-%m-%d"),
            end=datetime.strptime(args.fetch_end, "%Y-%m-%d"),
            expiry=args.fetch_expiry,
            interval=args.fetch_interval,
        )
        ingestor = RollingOptionIngestor(client=make_client(), out_dir=rolling_dir)
        written = ingestor.fetch_range(cfg)
        print(f"Fetched {len(written)} rolling-option parquet files into {rolling_dir}")

    df_raw = load_feature_history(args.history)
    if rolling_dir:
        ro_features = build_dataset(rolling_dir)
        if not ro_features.empty:
            df_raw = pd.concat([df_raw, ro_features], ignore_index=True)
    if df_raw.empty:
        print("No feature history available yet. Run the agent in paper/live mode to collect data.")
        return

    dataset_summary = summarise_dataset(df_raw)
    print("Raw dataset summary:")
    print(json.dumps(dataset_summary, indent=2))

    df_features = engineer_features(df_raw)
    labelled_df, targets = make_training_target(df_features, horizon=args.horizon)
    if labelled_df.empty:
        print("No engineered samples to train on.")
        return

    labelled_df = labelled_df.sort_values("timestamp").reset_index(drop=True)
    targets = targets.loc[labelled_df.index].reset_index(drop=True)
    excluded = {"environment", "risk_event", None, ""}
    mask = ~labelled_df["strategy"].isin(excluded)
    labelled_df = labelled_df[mask].reset_index(drop=True)
    targets = targets[mask].reset_index(drop=True)
    if labelled_df.empty:
        print("No eligible strategy samples after filtering (environment/risk events removed).")
        return
    if len(labelled_df) < args.min_samples:
        print(f"Not enough samples (<{args.min_samples}) to train a meaningful selector yet. Aborting.")
        return

    feature_cols = DEFAULT_FEATURE_COLUMNS
    X = labelled_df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = np.nan_to_num(targets.to_numpy(dtype=float), nan=0.0)

    whitelist = set(s.lower() for s in (args.strategies or []) if s)
    trained_strategies = {}
    eval_rows = []

    for strategy in sorted(labelled_df["strategy"].dropna().unique()):
        if whitelist and strategy.lower() not in whitelist:
            continue
        mask = labelled_df["strategy"] == strategy
        Xs = X[mask.values]
        ys = y[mask.values]
        if len(Xs) < args.min_samples:
            print(f"Skipping {strategy}: insufficient samples ({len(Xs)})")
            continue
        results = _train_single_strategy(strategy, Xs, ys, val_fraction=float(args.val_fraction))
        trained_strategies[strategy] = results["model"]
        eval_rows.append(results["metrics"])
        print(
            f"[train] {strategy:35s} samples={results['metrics']['samples']} "
            f"train_rmse={results['metrics']['train_rmse']:.2f} "
            f"val_rmse={results['metrics']['val_rmse']:.2f} "
            f"val_r2={results['metrics']['val_r2']:.2f}"
        )

    if not trained_strategies:
        print("No strategy models trained.")
        return

    model_payload = {
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "feature_columns": feature_cols,
        "strategies": trained_strategies,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(model_payload, indent=2))
    print(f"\nSaved selector model to {args.model_out}")

    report = {
        "generated_at": model_payload["trained_at"],
        "samples_total": int(len(labelled_df)),
        "target_horizon": args.horizon,
        "val_fraction": float(args.val_fraction),
        "strategies_trained": len(trained_strategies),
        "dataset_summary": dataset_summary,
        "metrics": eval_rows,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2))
    print(f"Saved training summary to {args.report_out}")


def _train_single_strategy(name: str, X: np.ndarray, y: np.ndarray, val_fraction: float) -> Dict[str, Any]:
    val_fraction = min(max(val_fraction, 0.05), 0.5)
    split_idx = max(1, int(len(X) * (1 - val_fraction)))
    train_X, val_X = X[:split_idx], X[split_idx:]
    train_y, val_y = y[:split_idx], y[split_idx:]
    if len(val_X) == 0:
        val_X, val_y = train_X, train_y

    weights, intercept = _fit_linear_model(train_X, train_y)
    train_pred = _predict(weights, intercept, train_X)
    val_pred = _predict(weights, intercept, val_X)

    train_metrics = _compute_metrics(train_y, train_pred)
    val_metrics = _compute_metrics(val_y, val_pred)

    model = {
        "intercept": float(intercept),
        "weights": [float(w) for w in weights],
        "samples": int(len(X)),
        "mean_target": float(train_y.mean()),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
        },
    }
    metrics = {
        "strategy": name,
        "samples": int(len(X)),
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
    }
    return {"model": model, "metrics": metrics}


def _fit_linear_model(X: np.ndarray, y: np.ndarray, l2: float = 1e-6) -> tuple[np.ndarray, float]:
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    ridge = l2 * np.eye(X_aug.shape[1])
    beta = np.linalg.pinv(X_aug.T @ X_aug + ridge) @ X_aug.T @ y
    intercept = float(beta[0])
    weights = beta[1:]
    return weights, intercept


def _predict(weights: np.ndarray, intercept: float, X: np.ndarray) -> np.ndarray:
    return intercept + X @ weights


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2)) if len(diff) else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff))) if len(diff) else 0.0
    var = float(np.var(y_true))
    r2 = 0.0 if var == 0 else max(0.0, 1.0 - mse / var)
    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    main()
