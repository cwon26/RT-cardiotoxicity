"""
Functional DVH analysis via fPCA.

A dose-volume histogram is a *curve*, not a handful of Vx points. Choosing fixed
thresholds (V5, V25, ...) throws away shape information and the thresholds are
arbitrary. Functional PCA treats each patient's cumulative DVH as one functional
observation and extracts a few principal-component "shape modes" that capture
almost all between-patient variation. Those scores are then compact, decorrelated
predictors.

This script, for a chosen substructure:
  1. builds finely-sampled cumulative DVH curves for every patient;
  2. runs PCA over the curves (fPCA), reports variance explained;
  3. plots the mean DVH and the first modes (mean ± component);
  4. checks how well the first K fPCA scores predict an outcome, vs the scalar
     Vx features — i.e. does the whole-curve representation help?

Usage
-----
    python fpca_dvh.py
    python fpca_dvh.py --structure LAD --outcome lad_osi_change --n-components 4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_dataset import generate_cohort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

# Sensible default outcome per substructure.
DEFAULT_OUTCOME = {"LV": "gls_rel_change_pct", "LAD": "lad_osi_change",
                   "heart": "lvef_decline", "RCA": "lad_wss_change",
                   "LCX": "lad_wss_change"}


def cv_r2(X, y, seed):
    """Repeated 5-fold CV R² of Ridge on the given feature block."""
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    return cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean()


def main():
    ap = argparse.ArgumentParser(description="Functional DVH analysis (fPCA).")
    ap.add_argument("--structure", default="LV", choices=list(DEFAULT_OUTCOME))
    ap.add_argument("--outcome", default=None, help="Outcome column (defaults per structure).")
    ap.add_argument("--n-components", type=int, default=4)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dose-max", type=float, default=50.0)
    ap.add_argument("--dose-step", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    outcome = args.outcome or DEFAULT_OUTCOME[args.structure]
    grid = np.arange(0.0, args.dose_max, args.dose_step)

    df, curves = generate_cohort(args.n, args.seed, dvh_grid=grid, return_curves=True)
    Y = df[outcome].to_numpy(float)
    C = curves[args.structure]                      # (n_patients, len(grid))

    # fPCA = PCA on the curve matrix (center each dose-bin).
    pca = PCA(n_components=args.n_components)
    scores = pca.fit_transform(C)                   # (n, K)
    evr = pca.explained_variance_ratio_
    print(f"Functional DVH (fPCA) — structure '{args.structure}', n={args.n}")
    print(f"Curve grid: 0–{args.dose_max} Gy step {args.dose_step}  ({len(grid)} points)\n")
    print("Variance explained by component:")
    for k, e in enumerate(evr, 1):
        print(f"  PC{k}: {e:6.1%}   (cumulative {np.cumsum(evr)[k-1]:.1%})")

    # Predict the outcome: fPCA scores vs scalar Vx features.
    vx_cols = [c for c in df.columns if c.startswith(f"{args.structure}__V") and c.endswith("Gy_cc")]
    r2_fpca = cv_r2(scores, Y, args.seed)
    r2_vx = cv_r2(df[vx_cols].to_numpy(float), Y, args.seed)
    print(f"\nPredicting '{outcome}' (Ridge, repeated 5-fold CV R²):")
    print(f"  {args.n_components} fPCA scores : R² {r2_fpca:.3f}")
    print(f"  {len(vx_cols)} scalar Vx feats : R² {r2_vx:.3f}")

    # Figure: variance, mode shapes, score scatter.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(range(1, len(evr) + 1), evr * 100, color="#08519c")
    axes[0].set_xlabel("Component"); axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("fPCA scree")

    mean_curve = C.mean(axis=0)
    axes[1].plot(grid, mean_curve, "k", lw=2, label="mean DVH")
    sd = np.sqrt(pca.explained_variance_)
    for k in range(min(2, args.n_components)):
        comp = pca.components_[k]
        axes[1].plot(grid, mean_curve + sd[k] * comp, lw=1.2, label=f"mean +PC{k+1}")
        axes[1].plot(grid, mean_curve - sd[k] * comp, lw=1.2, ls="--", label=f"mean -PC{k+1}")
    axes[1].set_xlabel("Dose (Gy)"); axes[1].set_ylabel("Volume ≥ dose (%)")
    axes[1].set_title(f"{args.structure} DVH shape modes"); axes[1].legend(fontsize=7)

    sc = axes[2].scatter(scores[:, 0], scores[:, 1] if args.n_components > 1 else Y * 0,
                         c=Y, cmap="viridis", s=14)
    axes[2].set_xlabel("PC1 score"); axes[2].set_ylabel("PC2 score")
    axes[2].set_title(f"Scores colored by\n{outcome}")
    fig.colorbar(sc, ax=axes[2])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fpca_dvh.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "fpca_results.json"), "w") as f:
        json.dump({"structure": args.structure, "outcome": outcome, "n": args.n,
                   "explained_variance_ratio": evr.tolist(),
                   "cumulative_variance": np.cumsum(evr).tolist(),
                   "cv_r2_fpca": float(r2_fpca), "cv_r2_vx": float(r2_vx),
                   "n_vx_features": len(vx_cols)}, f, indent=2)
    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  fpca_dvh.png, fpca_results.json")


if __name__ == "__main__":
    main()
