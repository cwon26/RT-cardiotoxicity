"""
Multi-target regression for EARLY (subclinical) cardiotoxicity markers.

Targets are continuous markers that change before LVEF declines:
  * gls_rel_change_pct  — relative change in echo global longitudinal strain (%)
  * wss_change_pa       — change in CFD endocardial wall shear stress (Pa)
  * energy_loss_change  — change in CFD intraventricular energy loss (mW)

Why regression (not classification / deep learning) at N≈300
------------------------------------------------------------
Continuous endpoints use every patient's information, so the effective sample
size is far larger than a rare binary event would give. With only a few hundred
patients, regularized *linear* models (Ridge / Lasso / ElasticNet) are the
right tool: they are stable, interpretable, and handle the many correlated DVH
predictors via shrinkage. Deep nets would overfit badly here.

Because we have baseline + follow-up, each marker's BASELINE value is used as a
predictor (baseline adjustment), and the TARGET is the change. Follow-up values
and the late LVEF endpoint are excluded from the predictors to avoid leakage.

Usage
-----
    python make_dataset.py        # writes data/cohort.csv
    python train_regression.py    # multi-target regression -> outputs/

    python train_regression.py --model mtelastic     # joint multi-task ENet
    python train_regression.py --targets gls_rel_change_pct
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning

# Coordinate-descent solvers occasionally stop just shy of the tolerance on these
# correlated DVH features; the fits are fine, so quiet the noise in this process.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    MultiTaskElasticNetCV,
    RidgeCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from make_dataset import ALL_TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

DEFAULT_TARGETS = ALL_TARGETS

# Anything that would leak the outcome: follow-up measurements, the change
# columns themselves, and the late LVEF endpoint. Baseline_* stays as predictors.
LEAKAGE_PREFIXES = ("followup_",)
LEAKAGE_SUFFIXES = ("_change", "_change_pct")
LEAKAGE_EXACT = {"lvef_decline", "cardiotoxicity", *ALL_TARGETS}


def select_features(df: pd.DataFrame, targets: list[str]) -> list[str]:
    """Predictor columns: DVH + clinical + baseline markers; exclude leakage."""
    cols = []
    for c in df.columns:
        if c in targets or c in LEAKAGE_EXACT:
            continue
        if c.startswith(LEAKAGE_PREFIXES):
            continue
        if c.endswith(LEAKAGE_SUFFIXES):
            continue
        cols.append(c)
    return cols


def build_estimator(name: str):
    """Return an unfitted (multi-output capable) regressor."""
    cv = 5
    if name == "ridge":
        return MultiOutputRegressor(RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if name == "lasso":
        return MultiOutputRegressor(LassoCV(alphas=60, cv=cv, max_iter=20000))
    if name == "elastic":
        return MultiOutputRegressor(
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], alphas=60,
                         cv=cv, max_iter=20000)
        )
    if name == "mtelastic":
        # Joint multi-task ENet: shares feature support across targets.
        return MultiTaskElasticNetCV(
            l1_ratio=[0.3, 0.5, 0.7, 0.9], alphas=60, cv=cv, max_iter=20000
        )
    if name == "rf":
        return RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    raise ValueError(f"Unknown model '{name}'. Choose ridge|lasso|elastic|mtelastic|rf.")


def make_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_tf = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat_tf = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    pre = ColumnTransformer(
        [("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop"
    )
    return Pipeline([("pre", pre), ("reg", build_estimator(model_name))])


def cross_validate_all(X, Y, seed: int) -> dict:
    """Repeated K-fold CV (stable at small N). Mean R² across targets."""
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)
    results = {}
    for name in ("ridge", "lasso", "elastic", "mtelastic", "rf"):
        pipe = make_pipeline(X, name)
        # Multi-output R² is averaged across targets by sklearn's scorer.
        r2 = cross_val_score(pipe, X, Y, cv=cv, scoring="r2", n_jobs=-1)
        results[name] = {"r2_mean": float(r2.mean()), "r2_std": float(r2.std())}
        print(f"  {name:10s}  R² {r2.mean():.3f} ± {r2.std():.3f}")
    return results


def per_target_metrics(Y_true: pd.DataFrame, Y_pred: np.ndarray, targets) -> dict:
    out = {}
    for j, t in enumerate(targets):
        yt, yp = Y_true.iloc[:, j].values, Y_pred[:, j]
        out[t] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "r2": float(r2_score(yt, yp)),
        }
    return out


def plot_pred_vs_actual(Y_true, Y_pred, targets, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    axes = np.atleast_1d(axes)
    for j, t in enumerate(targets):
        yt, yp = Y_true.iloc[:, j].values, Y_pred[:, j]
        ax = axes[j]
        ax.scatter(yt, yp, s=14, alpha=0.6)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{t}\nR² = {r2_score(yt, yp):.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def feature_importance_per_target(pipe, X_test, Y_test, targets, path) -> pd.DataFrame:
    """Permutation importance (R²) computed separately for each target."""
    frames = []
    for j, t in enumerate(targets):
        view = _SingleTargetView(pipe=pipe, j=j)
        r = permutation_importance(
            view, X_test, Y_test.iloc[:, j].values,
            scoring="r2", n_repeats=12, random_state=0, n_jobs=-1,
        )
        frames.append(
            pd.DataFrame({"target": t, "feature": X_test.columns,
                          "importance": r.importances_mean})
        )
    imp = pd.concat(frames, ignore_index=True)
    imp = imp.sort_values(["target", "importance"], ascending=[True, False])
    imp.to_csv(path, index=False)
    return imp


class _SingleTargetView(RegressorMixin, BaseEstimator):
    """Expose one column of a fitted multi-output regressor as a single-output
    estimator, so per-target permutation importance can be scored with R²."""

    def __init__(self, pipe=None, j=0):
        self.pipe = pipe
        self.j = j

    def fit(self, X, y=None):  # already fitted upstream; no-op for the API
        return self

    def predict(self, X):
        return self.pipe.predict(X)[:, self.j]


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-target regression for early cardiotoxicity markers.")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "cohort.csv"))
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--model", default="auto",
                    help="ridge|lasso|elastic|mtelastic|rf|auto (auto = best CV R²).")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\nRun `python make_dataset.py` first.")

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(args.data, index_col=0)
    missing = [t for t in args.targets if t not in df.columns]
    if missing:
        raise SystemExit(f"Targets not in data: {missing}\nAvailable: {list(df.columns)}")

    feature_cols = select_features(df, args.targets)
    X = df[feature_cols].copy()
    Y = df[args.targets].copy()

    print(f"Loaded {len(df)} patients, {X.shape[1]} predictors, {len(args.targets)} targets.")
    print(f"Targets: {args.targets}\n")

    print("Cross-validation (repeated 5-fold x5, mean R² across targets):")
    cv_results = cross_validate_all(X, Y, args.seed)

    if args.model == "auto":
        best = max(cv_results, key=lambda k: cv_results[k]["r2_mean"])
        print(f"\nAuto-selected best model: {best}")
    else:
        best = args.model

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed
    )
    pipe = make_pipeline(X, best)
    pipe.fit(X_tr, Y_tr)
    Y_pred = pipe.predict(X_te)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    test_metrics = per_target_metrics(Y_te, Y_pred, args.targets)
    print("\nHeld-out test performance (per target):")
    for t, m in test_metrics.items():
        print(f"  {t:20s}  R² {m['r2']:.3f}   MAE {m['mae']:.3f}   RMSE {m['rmse']:.3f}")

    # Artifacts.
    joblib.dump(
        {"pipeline": pipe, "features": feature_cols, "targets": args.targets, "task": "regression"},
        os.path.join(OUT_DIR, "model_regression.joblib"),
    )
    plot_pred_vs_actual(Y_te, Y_pred, args.targets, os.path.join(OUT_DIR, "pred_vs_actual.png"))
    imp = feature_importance_per_target(
        pipe, X_te, Y_te, args.targets, os.path.join(OUT_DIR, "feature_importance_regression.csv")
    )

    with open(os.path.join(OUT_DIR, "metrics_regression.json"), "w") as f:
        json.dump({"model": best, "n_train": len(X_tr), "n_test": len(X_te),
                   "targets": args.targets, "cv": cv_results, "test": test_metrics}, f, indent=2)

    print("\nTop predictors per target (permutation importance):")
    for t in args.targets:
        top = imp[imp.target == t].nlargest(4, "importance")
        feats = ", ".join(f"{r.feature}({r.importance:.3f})" for r in top.itertuples())
        print(f"  {t:20s} -> {feats}")

    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  model_regression.joblib, metrics_regression.json,")
    print("  pred_vs_actual.png, feature_importance_regression.csv")


if __name__ == "__main__":
    main()
