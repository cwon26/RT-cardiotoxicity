"""
Score new patients with a trained cardiotoxicity model.

Two entry points:

  * predict_from_csv: a CSV whose columns match the training feature table.
  * predict_from_dose: a raw per-voxel heart-dose array (Gy) — as produced by the
    MATLAB dose-on-mesh step — plus a dict of clinical covariates. DVH features
    are computed on the fly so callers don't have to replicate the feature logic.

Usage
-----
    python predict.py --csv new_patients.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import joblib

from dvh_features import extract_dvh_features

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "outputs", "model.joblib")


def load_model(path: str = DEFAULT_MODEL):
    if not os.path.exists(path):
        raise SystemExit(f"Model not found: {path}\nRun `python train.py` first.")
    return joblib.load(path)


def _align(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Add any missing training columns (as NaN -> imputed) and order them."""
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]


def predict_from_csv(csv_path: str, model_path: str = DEFAULT_MODEL) -> pd.DataFrame:
    bundle = load_model(model_path)
    pipe, feature_cols = bundle["pipeline"], bundle["features"]
    df = pd.read_csv(csv_path, index_col=0)
    X = _align(df.copy(), feature_cols)
    proba = pipe.predict_proba(X)[:, 1]
    return pd.DataFrame({"risk_probability": proba}, index=df.index)


def predict_from_dose(
    dose_gy: np.ndarray,
    clinical: dict | None = None,
    voxel_volume_cc: float | None = 0.002,
    model_path: str = DEFAULT_MODEL,
) -> float:
    """Score a single patient from their raw heart-dose array + clinical dict."""
    bundle = load_model(model_path)
    pipe, feature_cols = bundle["pipeline"], bundle["features"]
    feats = extract_dvh_features(dose_gy, voxel_volume_cc=voxel_volume_cc)
    if clinical:
        feats.update(clinical)
    X = _align(pd.DataFrame([feats]), feature_cols)
    return float(pipe.predict_proba(X)[:, 1][0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict cardiotoxicity risk.")
    ap.add_argument("--csv", required=True, help="CSV of patients to score.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="Optional output CSV path.")
    args = ap.parse_args()

    result = predict_from_csv(args.csv, args.model)
    if args.out:
        result.to_csv(args.out)
        print(f"Wrote predictions to {args.out}")
    else:
        print(result.to_string())


if __name__ == "__main__":
    main()
