"""
Score new patients with a trained cardiotoxicity model.

Handles both task types automatically based on the saved model bundle:
  * regression (default) -> predicts early subclinical markers (GLS change, CFD
    indices), one column per target.
  * classification       -> predicts the probability of the late CTRCD endpoint.

Two entry points:
  * predict_from_csv: a CSV whose columns match the training feature table.
  * predict_from_dose: a raw per-voxel heart-dose array (Gy) — as produced by the
    MATLAB dose-on-mesh step — plus a dict of clinical/baseline covariates. DVH
    features are computed on the fly so callers don't replicate the feature logic.

Usage
-----
    python predict.py --csv new_patients.csv
    python predict.py --csv new_patients.csv --model outputs/model.joblib
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import joblib

from dvh_features import extract_dvh_features

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "outputs", "model_regression.joblib")


def load_model(path: str = DEFAULT_MODEL):
    if not os.path.exists(path):
        raise SystemExit(
            f"Model not found: {path}\n"
            "Run `python train_regression.py` (regression) or `python train.py` "
            "(classification) first."
        )
    return joblib.load(path)


def _is_regression(bundle) -> bool:
    return bundle.get("task") == "regression" or "targets" in bundle


def _align(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Add any missing training columns (NaN -> imputed) and order them."""
    X = X.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]


def _predict(bundle, X: pd.DataFrame) -> pd.DataFrame:
    pipe, feats = bundle["pipeline"], bundle["features"]
    Xa = _align(X, feats)
    if _is_regression(bundle):
        Y = np.asarray(pipe.predict(Xa))
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        cols = [f"pred_{t}" for t in bundle["targets"]]
        return pd.DataFrame(Y, columns=cols, index=X.index)
    proba = pipe.predict_proba(Xa)[:, 1]
    return pd.DataFrame({"risk_probability": proba}, index=X.index)


def predict_from_csv(csv_path: str, model_path: str = DEFAULT_MODEL) -> pd.DataFrame:
    bundle = load_model(model_path)
    df = pd.read_csv(csv_path, index_col=0)
    return _predict(bundle, df)


def predict_from_dose(
    dose_gy: np.ndarray,
    clinical: dict | None = None,
    voxel_volume_cc: float | None = 0.002,
    model_path: str = DEFAULT_MODEL,
) -> pd.Series:
    """Score a single patient from their raw heart-dose array + clinical/baseline
    dict (e.g. baseline_gls, baseline_wss, age, anthracycline, ...)."""
    bundle = load_model(model_path)
    feats = extract_dvh_features(dose_gy, voxel_volume_cc=voxel_volume_cc)
    if clinical:
        feats.update(clinical)
    out = _predict(bundle, pd.DataFrame([feats], index=["patient"]))
    return out.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict cardiotoxicity markers / risk.")
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
