"""
Partial Least Squares (PLS) regression for substructure dose -> early markers.

Why PLS here
------------
The predictor block (DVH features for heart / LV / LAD / RCA / LCX) is wide and
highly collinear, and the response block (myocardial markers GLS/LV-CFD plus
coronary markers LAD-CFD) is itself correlated. PLS is built exactly for this:
it finds a few latent components that jointly explain covariance between the
dose block and the marker block, staying stable at N~300 where ordinary
multivariate regression would be unstable.

It also yields interpretable diagnostics:
  * the number of latent components (model complexity), chosen by CV;
  * VIP (Variable Importance in Projection) scores — VIP > 1 flags predictors
    that matter, e.g. whether LAD dose drives the coronary markers;
  * per-target R² / MAE / RMSE on a held-out set.

Usage
-----
    python make_dataset.py
    python train_pls.py
    python train_pls.py --max-components 8
    python train_pls.py --targets gls_rel_change_pct lv_wss_change
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from make_dataset import ALL_TARGETS, CORONARY_TARGETS, MYOCARDIAL_TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

# Post-baseline / outcome columns that must never be predictors.
def select_features(df: pd.DataFrame, targets: list[str]) -> list[str]:
    leak_exact = {"lvef_decline", "cardiotoxicity", *ALL_TARGETS}
    cols = []
    for c in df.columns:
        if c in targets or c in leak_exact:
            continue
        if c.startswith("followup_") or c.endswith("_change") or c.endswith("_change_pct"):
            continue
        cols.append(c)
    return cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_tf = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat_tf = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
                             remainder="drop")


def make_pipeline(X: pd.DataFrame, n_components: int) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(X)),
                     ("pls", PLSRegression(n_components=n_components, scale=False))])


def choose_n_components(X, Y, max_components: int, seed: int) -> tuple[int, dict]:
    """Pick the number of latent components by repeated K-fold CV (mean R²)."""
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)
    scores = {}
    print("Selecting number of PLS components (repeated 5-fold x5, mean R²):")
    for k in range(1, max_components + 1):
        pipe = make_pipeline(X, k)
        r2 = cross_val_score(pipe, X, Y, cv=cv, scoring="r2", n_jobs=-1)
        scores[k] = {"r2_mean": float(r2.mean()), "r2_std": float(r2.std())}
        print(f"  components={k:2d}   R² {r2.mean():.3f} ± {r2.std():.3f}")
    best_k = max(scores, key=lambda k: scores[k]["r2_mean"])
    print(f"Selected {best_k} components.")
    return best_k, scores


def vip_scores(pls: PLSRegression) -> np.ndarray:
    """Variable Importance in Projection for a fitted PLSRegression.

    VIP_j = sqrt( p * Σ_a (w_aj² · SSY_a) / Σ_a SSY_a ), where SSY_a is the
    response variance captured by component a. VIP > 1 marks influential
    predictors.
    """
    t = pls.x_scores_         # (n, A)
    w = pls.x_weights_        # (p, A)
    q = pls.y_loadings_       # (m, A)
    p, A = w.shape
    ssy = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)   # (A,)
    total = np.sum(ssy)
    w_norm2 = (w / np.linalg.norm(w, axis=0)) ** 2          # normalize each comp
    return np.sqrt(p * (w_norm2 @ ssy) / total)


def per_target_metrics(Y_true, Y_pred, targets) -> dict:
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
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for j, t in enumerate(targets):
        yt, yp = Y_true.iloc[:, j].values, Y_pred[:, j]
        ax = axes[j]
        ax.scatter(yt, yp, s=14, alpha=0.6)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{t}\nR² = {r2_score(yt, yp):.2f}", fontsize=9)
    for j in range(len(targets), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="PLS regression: substructure dose -> early markers.")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "cohort.csv"))
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS,
                    help="Response columns (default: all myocardial + coronary markers).")
    ap.add_argument("--max-components", type=int, default=8)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\nRun `python make_dataset.py` first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(args.data, index_col=0)
    missing = [t for t in args.targets if t not in df.columns]
    if missing:
        raise SystemExit(f"Targets not in data: {missing}")

    feature_cols = select_features(df, args.targets)
    X = df[feature_cols].copy()
    Y = df[args.targets].copy()
    print(f"Loaded {len(df)} patients, {X.shape[1]} predictors, {len(args.targets)} targets.")
    myo = [t for t in args.targets if t in MYOCARDIAL_TARGETS]
    cor = [t for t in args.targets if t in CORONARY_TARGETS]
    print(f"  myocardial: {myo}")
    print(f"  coronary  : {cor}\n")

    best_k, cv_scores = choose_n_components(X, Y, args.max_components, args.seed)

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=args.test_size, random_state=args.seed)
    pipe = make_pipeline(X, best_k)
    pipe.fit(X_tr, Y_tr)
    Y_pred = np.asarray(pipe.predict(X_te))
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    test_metrics = per_target_metrics(Y_te, Y_pred, args.targets)
    print("\nHeld-out test performance (per target):")
    for t, m in test_metrics.items():
        print(f"  {t:22s} R² {m['r2']:6.3f}   MAE {m['mae']:.4f}   RMSE {m['rmse']:.4f}")

    # VIP scores over the preprocessed predictors.
    feat_names = pipe.named_steps["pre"].get_feature_names_out()
    feat_names = [n.split("__", 1)[-1] for n in feat_names]  # drop ColumnTransformer prefix
    vips = vip_scores(pipe.named_steps["pls"])
    vip_df = (pd.DataFrame({"feature": feat_names, "vip": vips})
              .sort_values("vip", ascending=False).reset_index(drop=True))
    vip_df.to_csv(os.path.join(OUT_DIR, "vip_scores.csv"), index=False)

    plot_pred_vs_actual(Y_te, Y_pred, args.targets, os.path.join(OUT_DIR, "pls_pred_vs_actual.png"))
    joblib.dump(
        {"pipeline": pipe, "features": feature_cols, "targets": args.targets,
         "task": "regression", "n_components": best_k},
        os.path.join(OUT_DIR, "model_pls.joblib"),
    )
    with open(os.path.join(OUT_DIR, "metrics_pls.json"), "w") as f:
        json.dump({"n_components": best_k, "n_train": len(X_tr), "n_test": len(X_te),
                   "targets": args.targets, "cv_components": cv_scores,
                   "test": test_metrics}, f, indent=2)

    print(f"\nTop predictors by VIP (>1 = influential), {len(vip_df)} total:")
    print(vip_df.head(12).to_string(index=False))
    n_relevant = int((vip_df["vip"] > 1).sum())
    print(f"  ({n_relevant} predictors with VIP > 1)")

    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  model_pls.joblib, metrics_pls.json, pls_pred_vs_actual.png, vip_scores.csv")


if __name__ == "__main__":
    main()
