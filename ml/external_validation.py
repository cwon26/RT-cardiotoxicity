"""
External validation of the cardiotoxicity risk model on an independent cohort.

Internal cross-validation overstates how a model travels to a new hospital, whose
case-mix and treatment differ. Here we:

  1. train a CTRCD risk model on the development cohort;
  2. apply it, frozen, to an *external* cohort simulated with distribution shift
     (more right-sided cases, less anthracycline, lower delivered dose);
  3. report discrimination (ROC-AUC) AND calibration (calibration slope and
     intercept, Brier score) — calibration is what usually breaks under shift;
  4. show that a simple logistic recalibration restores agreement.

The external cohort is generated with the shift knobs in `make_dataset`.

Usage
-----
    python external_validation.py
    python external_validation.py --target cardiotoxicity --n-external 400
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from make_dataset import generate_cohort
from train import _predictor_columns   # reuse the leakage-safe predictor selection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")


def build_model(X: pd.DataFrame) -> Pipeline:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def calibration_slope_intercept(y, p):
    """Fit logit(y) ~ a + b·logit(p). b=1, a=0 means perfect calibration."""
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def evaluate(y, p, label):
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    print(f"  {label}")
    print(f"    ROC-AUC          = {auc:.3f}")
    print(f"    Brier score      = {brier:.3f}")
    print(f"    calibration slope= {slope:.3f}   (ideal 1.0)")
    print(f"    calibration int. = {intercept:.3f}   (ideal 0.0)")
    return {"roc_auc": float(auc), "brier": float(brier),
            "calib_slope": slope, "calib_intercept": intercept}


def main():
    ap = argparse.ArgumentParser(description="External validation under distribution shift.")
    ap.add_argument("--target", default="cardiotoxicity")
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-external", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Development cohort (primary distribution).
    dev = generate_cohort(args.n_dev, seed=args.seed)
    # External cohort: different site — more right-sided, less anthracycline,
    # lower delivered dose, separate seed.
    ext = generate_cohort(args.n_external, seed=args.seed + 101,
                          left_frac=0.35, anthracycline_p=0.25,
                          trastuzumab_p=0.20, dose_scale=0.8)

    feat_cols = _predictor_columns(dev, args.target)
    Xdev, ydev = dev[feat_cols], dev[args.target].astype(int)
    Xext, yext = ext[feat_cols], ext[args.target].astype(int)

    print("External validation of a CTRCD risk model")
    print(f"  development : n={len(dev)}, event rate {ydev.mean():.1%}, "
          f"left-sided {(dev['laterality']=='left').mean():.0%}")
    print(f"  external    : n={len(ext)}, event rate {yext.mean():.1%}, "
          f"left-sided {(ext['laterality']=='left').mean():.0%}\n")

    model = build_model(Xdev).fit(Xdev, ydev)

    # Apparent (development) performance.
    p_dev = model.predict_proba(Xdev)[:, 1]
    print("Development (apparent) performance:")
    dev_metrics = evaluate(ydev.to_numpy(), p_dev, "in-sample")

    # External performance (frozen model).
    p_ext = model.predict_proba(Xext)[:, 1]
    print("\nExternal performance (frozen model):")
    ext_metrics = evaluate(yext.to_numpy(), p_ext, "external")

    # Logistic recalibration on the external cohort (slope+intercept update).
    eps = 1e-6
    lp_ext = np.log(np.clip(p_ext, eps, 1 - eps) / (1 - np.clip(p_ext, eps, 1 - eps)))
    recal = LogisticRegression(C=1e6, max_iter=1000).fit(lp_ext.reshape(-1, 1), yext)
    p_recal = recal.predict_proba(lp_ext.reshape(-1, 1))[:, 1]
    print("\nExternal after logistic recalibration:")
    recal_metrics = evaluate(yext.to_numpy(), p_recal, "external+recal")

    # Figure: calibration curves before/after.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def calib_points(y, p, bins=8):
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p <= hi)
            if m.sum() > 0:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        return np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ydev_a, yext_a = ydev.to_numpy(), yext.to_numpy()
    for p, y_a, lbl, col in [(p_dev, ydev_a, "development", "#08519c"),
                             (p_ext, yext_a, "external (frozen)", "#c0504d"),
                             (p_recal, yext_a, "external (recalibrated)", "#2ca25f")]:
        xs, ys = calib_points(y_a, p)
        ax.plot(xs, ys, "o-", color=col, label=lbl, ms=5)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration under distribution shift"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "external_calibration.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "external_validation_results.json"), "w") as f:
        json.dump({"development": dev_metrics, "external": ext_metrics,
                   "external_recalibrated": recal_metrics,
                   "dev_event_rate": float(ydev.mean()),
                   "ext_event_rate": float(yext.mean())}, f, indent=2)
    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  external_calibration.png, external_validation_results.json")


if __name__ == "__main__":
    main()
