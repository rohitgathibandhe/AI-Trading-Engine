# -*- coding: utf-8 -*-
"""
Feature store helpers for strategy selection.

This module loads the feature snapshots recorded by the live agent
(`feature_history.csv`) and prepares them for downstream model
training.  The actual ML fitting is intentionally left as a stub so
we can iterate once enough data has been collected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import numpy as np

FEATURE_HISTORY_DEFAULT = Path(__file__).resolve().parents[2] / "state" / "feature_history.csv"


def load_feature_history(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load the raw feature log captured by the live agent.

    Returns an empty DataFrame when no samples are present.
    """
    history_path = Path(path or FEATURE_HISTORY_DEFAULT)
    if not history_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(history_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load feature history: {exc}") from exc

    if df.empty:
        return df

    # Normalise timestamp and numeric columns up-front
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    numeric_cols = [
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
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features ready for training.

    Currently computes simple spreads/differences; extend as the dataset grows.
    """
    if df.empty:
        return df

    engineered = df.copy()
    engineered["credit_ratio"] = (engineered["net_credit"]) / (
        engineered["ce_entry"].abs() + engineered["pe_entry"].abs() + 1e-6
    )
    engineered["delta_skew"] = engineered["ce_delta"] - engineered["pe_delta"]
    engineered["strike_gap"] = engineered["ce_strike"] - engineered["pe_strike"]
    engineered["is_warn_only"] = engineered["warn_only"].fillna(0).astype(int)
    engineered["abs_ce_delta"] = engineered["ce_delta"].abs()
    engineered["abs_pe_delta"] = engineered["pe_delta"].abs()
    engineered["credit_per_strike_gap"] = engineered["net_credit"] / (engineered["strike_gap"].abs() + 1e-6)
    return engineered


def make_training_target(df: pd.DataFrame, horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Placeholder label creation: shifts net_credit by `horizon` rows to approximate
    realised edge. Replace with true realised PnL once available.
    """
    if df.empty:
        return df, pd.Series(dtype=float)

    labelled = df.sort_values("timestamp").reset_index(drop=True)
    target = labelled["net_credit"].shift(-horizon).fillna(labelled["net_credit"])
    return labelled, target


DEFAULT_FEATURE_COLUMNS: List[str] = [
    "credit_ratio",
    "delta_skew",
    "strike_gap",
    "abs_ce_delta",
    "abs_pe_delta",
    "credit_per_strike_gap",
    "ce_ltp",
    "pe_ltp",
    "ce_entry",
    "pe_entry",
]


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, pd.Series, List[str]]:
    """
    Convert engineered dataframe to (X, y, columns) for model fitting.
    """
    if df.empty:
        return np.empty((0, 0)), pd.Series(dtype=float), []

    cols = feature_cols or DEFAULT_FEATURE_COLUMNS
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered columns required for model: {missing}")

    X = df[cols].fillna(0.0).to_numpy(dtype=float)
    y = df["net_credit"].fillna(0.0)
    return X, y, cols


def summarise_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return quick diagnostics for CLI/Notebook consumption.
    """
    if df.empty:
        return {"samples": 0}

    return {
        "samples": int(len(df)),
        "strategies": df["strategy"].value_counts().to_dict() if "strategy" in df.columns else {},
        "date_range": (
            df["timestamp"].min().isoformat() if "timestamp" in df.columns else None,
            df["timestamp"].max().isoformat() if "timestamp" in df.columns else None,
        ),
        "warn_only_rate": float(df["warn_only"].mean()) if "warn_only" in df.columns else None,
    }


def append_to_feature_history(df: pd.DataFrame, feature_log_path: Path) -> None:
    if df.empty:
        return
    existing = Path(feature_log_path)
    if existing.exists():
        df_existing = pd.read_csv(existing)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(existing, index=False)
