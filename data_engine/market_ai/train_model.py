#!/usr/bin/env python3
"""
train_model.py

Robust training + prediction generator.

Produces:
 - model artifact (joblib dict) containing: {'scaler': scaler, 'clf': clf, 'features': features}
 - predictions_latest.csv with columns:
   ['date','symbol','open','high','low','close','vol_20','pred','prob_pos']

Usage examples:
  # force-train on synthetic/demo data
  python -m market_ai.train_model --force-train

  # explicitly set outputs
  python -m market_ai.train_model --out-model model_randomforest.pkl --out-preds predictions_latest.csv --force-train
"""
import argparse
import os
import tempfile
import time
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging

LOG = logging.getLogger("train_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def generate_synthetic_market_data(n_days: int = 240) -> pd.DataFrame:
    """Create a small synthetic market DataFrame for demo/testing.
    Columns: date, open, high, low, close, vol_20
    """
    rng = np.random.RandomState(42)
    base = 25000.0
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    drift = rng.normal(loc=0.0, scale=10.0, size=n_days).cumsum()
    close = base + drift + rng.normal(scale=50.0, size=n_days)
    opens = close + rng.normal(scale=10.0, size=n_days)
    highs = np.maximum(opens, close) + rng.uniform(0, 20, size=n_days)
    lows = np.minimum(opens, close) - rng.uniform(0, 20, size=n_days)
    vol_20 = np.abs(rng.normal(loc=1000000, scale=200000, size=n_days)).astype(int)
    df = pd.DataFrame({
        "date": dates.astype(str),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": close,
        "vol_20": vol_20,
    })
    LOG.info("Generated synthetic demo market data (rows=%d)", len(df))
    return df


def prepare_features_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Convert a market DataFrame into X, y for model training.

    We will:
     - create a synthetic label 'label' (next-day up/down) for demo if none exists
     - compute a few simple technical features
     - drop string columns from X if present (defensive)
    """
    d = df.copy()

    # ensure date exists and sorted
    if "date" not in d.columns:
        d["date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(d)).astype(str)
    d = d.sort_values("date").reset_index(drop=True)

    # create simple target: next-day close > today close => 1 else 0
    d["close_next"] = d["close"].shift(-1)
    d["label"] = (d["close_next"] > d["close"]).astype(int)
    d = d.dropna(subset=["label"]).reset_index(drop=True)

    # features: pct change, rolling stats
    d["ret1"] = d["close"].pct_change().fillna(0.0)
    d["ma5"] = d["close"].rolling(5, min_periods=1).mean()
    d["ma10"] = d["close"].rolling(10, min_periods=1).mean()
    d["vol_ma5"] = d["vol_20"].rolling(5, min_periods=1).mean()
    d["hl_range"] = d["high"] - d["low"]

    feature_cols = ["close", "ret1", "ma5", "ma10", "vol_ma5", "hl_range"]
    # Drop any object-type columns from X (defensive)
    X = d[feature_cols].copy()
    obj_cols = [c for c in X.columns if X[c].dtype == object]
    if obj_cols:
        LOG.warning("Dropping object dtype cols from features: %s", obj_cols)
        X = X.drop(columns=obj_cols)

    y = d["label"].astype(int)

    # ensure numeric dtype for all X
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, y, feature_cols


def train_and_save_model(df: pd.DataFrame, out_model: str, seed: int = 42) -> Dict[str, Any]:
    """Train a RandomForest on df and save artifact atomically to out_model."""
    X, y, features = prepare_features_for_training(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_val_s)
    LOG.info("Model performance (validation):\n%s", classification_report(y_val, preds))

    artifact = {"scaler": scaler, "clf": clf, "features": features}

    # write atomically
    tmp = out_model + f".tmp-{int(time.time())}"
    joblib.dump(artifact, tmp)
    os.replace(tmp, out_model)
    LOG.info("Saved model artifact to %s", out_model)
    return artifact


def generate_predictions_from_model(artifact: Dict[str, Any], market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use artifact to create prediction rows that the backtest expects.
    Returns DataFrame with columns: ['date','symbol','open','high','low','close','vol_20','pred','prob_pos']
    """
    features = artifact["features"]
    scaler = artifact["scaler"]
    clf = artifact["clf"]

    md = market_df.copy().sort_values("date").reset_index(drop=True)

    # create same features as in training (defensive)
    md["ret1"] = md["close"].pct_change().fillna(0.0)
    md["ma5"] = md["close"].rolling(5, min_periods=1).mean()
    md["ma10"] = md["close"].rolling(10, min_periods=1).mean()
    md["vol_ma5"] = md["vol_20"].rolling(5, min_periods=1).mean()
    md["hl_range"] = md["high"] - md["low"]

    X = md[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xs = scaler.transform(X)

    prob_pos = clf.predict_proba(Xs)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Xs)
    pred = (prob_pos >= 0.5).astype(int)

    out = pd.DataFrame({
        "date": md["date"].astype(str),
        "symbol": md.get("symbol", "NIFTY"),
        "open": md["open"],
        "high": md["high"],
        "low": md["low"],
        "close": md["close"],
        "vol_20": md["vol_20"],
        "pred": pred,
        "prob_pos": prob_pos
    })
    return out


def safe_load_artifact(path: str):
    """Try to load model artifact; on failure return None."""
    try:
        art = joblib.load(path)
        if not isinstance(art, dict) or "clf" not in art or "scaler" not in art or "features" not in art:
            raise ValueError("Artifact shape invalid")
        return art
    except Exception as e:
        LOG.warning("Failed to load artifact %s: %s", path, e)
        return None


def main():
    p = argparse.ArgumentParser(description="Train model & generate predictions (robust)")
    p.add_argument("--market-data", default="market_data.csv", help="CSV with market data (date,open,high,low,close,vol_20)")
    p.add_argument("--out-model", default="model_randomforest.pkl", help="output model artifact")
    p.add_argument("--out-preds", default="predictions_latest.csv", help="predictions output CSV")
    p.add_argument("--force-train", action="store_true", help="retrain even if model exists")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # load or generate market data
    if os.path.exists(args.market_data):
        try:
            md = pd.read_csv(args.market_data)
            LOG.info("Loaded market data from %s rows=%d", args.market_data, len(md))
        except Exception as e:
            LOG.warning("Failed to read %s: %s. Generating synthetic data", args.market_data, e)
            md = generate_synthetic_market_data()
    else:
        LOG.info("Market data file not found (%s) - generating synthetic demo data", args.market_data)
        md = generate_synthetic_market_data()

    artifact = None
    if not args.force_train and os.path.exists(args.out_model):
        LOG.info("Attempting to load existing model: %s", args.out_model)
        artifact = safe_load_artifact(args.out_model)
        if artifact is None:
            # move the bad file out of the way
            try:
                os.rename(args.out_model, args.out_model + ".broken")
                LOG.info("Renamed corrupt model to %s.broken", args.out_model)
            except Exception:
                pass

    if artifact is None:
        artifact = train_and_save_model(md, args.out_model, seed=args.seed)

    # generate predictions (for entire market_df)
    preds_df = generate_predictions_from_model(artifact, md)
    preds_df.to_csv(args.out_preds, index=False)
    LOG.info("Saved predictions to %s (rows=%d)", args.out_preds, len(preds_df))
    LOG.info("Done.")


if __name__ == "__main__":
    main()
