"""
Train and evaluate cardiotoxicity-prediction models on the DVH + clinical table.

Pipeline
--------
1. Load the cohort CSV (DVH features + clinical covariates + label).
2. Build a preprocessing + model pipeline (impute, scale, one-hot encode).
3. Compare several classifiers with stratified cross-validation (ROC-AUC, PR-AUC).
4. Refit the best model on a train split, evaluate on a held-out test set, and
   save: the fitted model, a metrics JSON, a ROC curve, and feature importances.

Usage
-----
    python make_dataset.py            # creates ml/data/cohort.csv
    python train.py                   # trains on it, writes ml/outputs/

    python train.py --data my.csv --target cardiotoxicity --model gboost
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

# Columns that are outcomes/identifiers, never predictors.
LEAKAGE_COLUMNS = {"cardiotoxicity", "lvef_decline"}


def build_model(name: str):
    """Return an unfitted classifier by short name."""
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1
        )
    if name == "gboost":
        return GradientBoostingClassifier(random_state=0)
    raise ValueError(f"Unknown model '{name}'. Choose logreg|rf|gboost.")


def make_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    """Preprocessing + classifier pipeline that adapts to column dtypes."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    num_tf = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat_tf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        [("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", build_model(model_name))])


def cross_validate_all(X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    """Stratified 5-fold CV ROC-AUC / PR-AUC for each candidate model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = {}
    for name in ("logreg", "rf", "gboost"):
        pipe = make_pipeline(X, name)
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        ap = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
        results[name] = {
            "roc_auc_mean": float(auc.mean()),
            "roc_auc_std": float(auc.std()),
            "pr_auc_mean": float(ap.mean()),
            "pr_auc_std": float(ap.std()),
        }
        print(
            f"  {name:7s}  ROC-AUC {auc.mean():.3f}±{auc.std():.3f}   "
            f"PR-AUC {ap.mean():.3f}±{ap.std():.3f}"
        )
    return results


def plot_roc(y_true, y_score, auc, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Cardiotoxicity prediction — held-out ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def feature_importances(pipe: Pipeline, X_test, y_test, path: str) -> pd.DataFrame:
    """Permutation importance on the held-out set (model-agnostic)."""
    r = permutation_importance(
        pipe, X_test, y_test, scoring="roc_auc", n_repeats=15, random_state=0, n_jobs=-1
    )
    imp = (
        pd.DataFrame(
            {"feature": X_test.columns, "importance": r.importances_mean, "std": r.importances_std}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(path, index=False)
    return imp


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RT-cardiotoxicity classifier.")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "cohort.csv"))
    ap.add_argument("--target", default="cardiotoxicity")
    ap.add_argument(
        "--model",
        default="auto",
        help="logreg|rf|gboost|auto (auto = best CV ROC-AUC).",
    )
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Run `python make_dataset.py` first to create a synthetic cohort."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(args.data, index_col=0)
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not in columns: {list(df.columns)}")

    drop_cols = LEAKAGE_COLUMNS | {args.target}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[args.target].astype(int)

    print(f"Loaded {len(df)} patients, {X.shape[1]} features.")
    print(f"Event rate: {y.mean():.1%}\n")

    print("Cross-validation (stratified 5-fold):")
    cv_results = cross_validate_all(X, y, args.seed)

    if args.model == "auto":
        best = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
        print(f"\nAuto-selected best model: {best}")
    else:
        best = args.model

    # Train / test split + refit.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    pipe = make_pipeline(X, best)
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_te, proba)
    ap_score = average_precision_score(y_te, proba)
    brier = brier_score_loss(y_te, proba)

    print("\nHeld-out test performance:")
    print(f"  ROC-AUC : {auc:.3f}")
    print(f"  PR-AUC  : {ap_score:.3f}")
    print(f"  Brier   : {brier:.3f}")
    print("\n" + classification_report(y_te, pred, digits=3))

    # Persist artifacts.
    model_path = os.path.join(OUT_DIR, "model.joblib")
    joblib.dump({"pipeline": pipe, "features": feature_cols, "target": args.target}, model_path)
    plot_roc(y_te, proba, auc, os.path.join(OUT_DIR, "roc_curve.png"))
    imp = feature_importances(pipe, X_te, y_te, os.path.join(OUT_DIR, "feature_importance.csv"))

    metrics = {
        "model": best,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "event_rate": float(y.mean()),
        "cv": cv_results,
        "test": {
            "roc_auc": float(auc),
            "pr_auc": float(ap_score),
            "brier": float(brier),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        },
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTop predictive features (permutation importance):")
    print(imp.head(8).to_string(index=False))
    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  model.joblib, metrics.json, roc_curve.png, feature_importance.csv")


if __name__ == "__main__":
    main()
