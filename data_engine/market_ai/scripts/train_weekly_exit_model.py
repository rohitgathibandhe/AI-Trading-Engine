#!/usr/bin/env python3
"""
Train a gradient-boosted exit model for the weekly theta strategy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES: List[str] = [
    "weekday",
    "spot_open",
    "spot_close",
    "combined_premium_pct",
    "iv_rank",
    "iv_skew",
    "oi_skew",
    "volume_skew",
    "call_oi_max_strike",
    "put_oi_max_strike",
    "spot_return_1",
    "spot_return_3",
    "spot_return_5",
    "spot_trend_20",
    "spot_trend_50",
    "spot_trend_100",
    "spot_volatility",
    "spot_intraday_range_pct",
    "call_oi_change",
    "put_oi_change",
    "event_count",
    "has_event_risk",
    "days_since_entry",
    "target_days_remaining",
    "current_pnl",
    "best_pnl",
    "drawdown_from_best",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weekly exit model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to weekly_theta_learning_dataset.csv")
    parser.add_argument("--model-out", type=Path, default=Path("data_engine/market_ai/state/weekly_exit_model.pkl"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/weekly_exit_model_report.json"))
    parser.add_argument("--close-advantage", type=float, default=1000.0, help="Min INR advantage to label 'close now'.")
    parser.add_argument("--prob-threshold", type=float, default=0.6, help="Probability cutoff for inference.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction.")
    return parser.parse_args()


def load_training_frame(path: Path, advantage: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = df["has_position"] & df["final_realized"].notna() & df["current_pnl"].notna()
    df = df.loc[mask].copy()
    df["advantage"] = df["current_pnl"] - df["final_realized"]
    df["sacrifice"] = df["final_realized"] - df["current_pnl"]
    pos_mask = df["advantage"] >= advantage
    neg_mask = df["sacrifice"] >= advantage
    df = df.loc[pos_mask | neg_mask].copy()
    df["label"] = 0
    df.loc[pos_mask, "label"] = 1
    df["has_event_risk"] = df["has_event_risk"].astype(int)
    df["blocked_today"] = df["blocked_today"].astype(int)
    return df


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}")
    advantage = args.close_advantage
    train_df = load_training_frame(args.data, advantage)
    while (train_df.empty or train_df["label"].nunique() < 2) and advantage > 200:
        advantage *= 0.8
        train_df = load_training_frame(args.data, advantage)
    if train_df.empty or train_df["label"].nunique() < 2:
        raise SystemExit("Insufficient class diversity; collect more trades or lower close-advantage.")
    if advantage != args.close_advantage:
        print(f"[weekly-exit] adjusted close-advantage to {advantage:.0f}")
    X = train_df[FEATURES].fillna(0.0)
    y = train_df["label"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    proba_val = clf.predict_proba(X_val)[:, 1]
    preds = (proba_val >= args.prob_threshold).astype(int)
    report = classification_report(y_val, preds, output_dict=True)
    roc = roc_auc_score(y_val, proba_val) if len(np.unique(y_val)) > 1 else float("nan")
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "estimator": clf,
        "feature_names": FEATURES,
        "prob_threshold": args.prob_threshold,
        "close_advantage": advantage,
    }
    with args.model_out.open("wb") as fh:
        pickle.dump(payload, fh)
    metrics = {
        "samples_total": int(len(train_df)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "close_advantage": advantage,
        "prob_threshold": args.prob_threshold,
        "roc_auc": roc,
        "classification_report": report,
    }
    if hasattr(clf, "feature_importances_"):
        metrics["feature_importances"] = {
            name: float(val)
            for name, val in zip(FEATURES, clf.feature_importances_)
        }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(metrics, indent=2))
    print(f"[weekly-exit] model saved to {args.model_out}")
    print(f"[weekly-exit] report saved to {args.report_out}")


if __name__ == "__main__":
    main()
