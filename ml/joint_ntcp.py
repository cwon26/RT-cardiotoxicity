"""
Joint multi-substructure NTCP model.

Single-structure NTCP (`ntcp.py`) fits one dose-response curve at a time, but the
substructure doses are correlated (a left-sided plan raises LV *and* LAD dose),
so a single-structure D50 mixes in the effect of the others. A joint multivariable
logistic NTCP puts several substructure mean doses in one model and reports each
one's **independent** (mutually adjusted) contribution:

    logit P(event) = b0 + Σ_s b_s · D_s

Outputs:
  * adjusted odds ratios per substructure (per Gy), with bootstrap CIs;
  * a likelihood-ratio test and AIC vs the single-structure models, showing
    whether the joint model is justified;
  * a 2-D risk surface over the two most influential substructures.

Usage
-----
    python make_dataset.py
    python joint_ntcp.py
    python joint_ntcp.py --structures heart LV LAD RCA LCX --endpoint cardiotoxicity
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ntcp import derive_endpoints

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

DEFAULT_STRUCTURES = ["heart", "LV", "LAD"]
DEFAULT_ENDPOINT = "cardiotoxicity"


def _loglik(clf, X, y):
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logit(X, y):
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(X, y)


def bootstrap_or(X, y, n_boot, seed):
    """Bootstrap CIs for per-coefficient odds ratios (per raw unit = per Gy)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        try:
            c = fit_logit(X[idx], yb).coef_[0]
        except Exception:
            continue
        coefs.append(c)
    coefs = np.asarray(coefs)
    lo = np.exp(np.percentile(coefs, 2.5, axis=0))
    hi = np.exp(np.percentile(coefs, 97.5, axis=0))
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description="Joint multi-substructure NTCP.")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "cohort.csv"))
    ap.add_argument("--structures", nargs="+", default=DEFAULT_STRUCTURES)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\nRun `python make_dataset.py` first.")
    os.makedirs(OUT_DIR, exist_ok=True)
    df = derive_endpoints(pd.read_csv(args.data, index_col=0))

    dose_cols = [f"{s}__mean_dose_gy" for s in args.structures]
    for c in (*dose_cols, args.endpoint):
        if c not in df.columns:
            raise SystemExit(f"Column missing: {c}")
    X = df[dose_cols].to_numpy(float)
    y = df[args.endpoint].to_numpy(int)
    k = len(args.structures)

    # Joint model.
    joint = fit_logit(X, y)
    ll_joint = _loglik(joint, X, y)
    aic_joint = 2 * (k + 1) - 2 * ll_joint
    or_joint = np.exp(joint.coef_[0])
    or_lo, or_hi = bootstrap_or(X, y, args.n_boot, args.seed)

    print(f"Joint multi-substructure NTCP  (n={len(df)}, endpoint='{args.endpoint}', "
          f"event rate {y.mean():.1%})\n")
    print("Mutually-adjusted odds ratios (per Gy of mean dose):")
    print(f"  {'structure':8s} {'OR/Gy':>7s}  {'95% CI':>16s}   {'single-OR/Gy':>12s}")
    single_lls = []
    rows = []
    for j, s in enumerate(args.structures):
        # single-structure model for comparison
        xs = X[:, [j]]
        single = fit_logit(xs, y)
        single_lls.append(_loglik(single, xs, y))
        or_single = float(np.exp(single.coef_[0, 0]))
        adj = "*" if (or_lo[j] > 1) == (or_hi[j] > 1) else " "
        print(f"  {s:8s} {or_joint[j]:7.3f}  [{or_lo[j]:5.3f}, {or_hi[j]:5.3f}]   "
              f"{or_single:12.3f} {adj}")
        rows.append({"structure": s, "adjusted_or_per_gy": float(or_joint[j]),
                     "ci95": [float(or_lo[j]), float(or_hi[j])],
                     "single_or_per_gy": or_single})

    # Best single-structure model (highest log-lik) vs joint: LR test + AIC.
    best_j = int(np.argmax(single_lls))
    ll_best_single = single_lls[best_j]
    aic_best_single = 2 * 2 - 2 * ll_best_single
    lr_stat = 2 * (ll_joint - ll_best_single)
    dof = k - 1
    from math import erf, sqrt
    # chi-square survival via series is overkill; approximate p with a Wilson-Hilferty
    # transform to normal for a quick, dependency-free p-value.
    if dof > 0 and lr_stat > 0:
        x = lr_stat / dof
        z = (x ** (1 / 3) - (1 - 2 / (9 * dof))) / sqrt(2 / (9 * dof))
        p_lr = 1 - 0.5 * (1 + erf(z / sqrt(2)))
    else:
        p_lr = float("nan")

    print(f"\nModel comparison (joint vs best single = {args.structures[best_j]}):")
    print(f"  log-lik joint        = {ll_joint:.2f}   AIC = {aic_joint:.1f}")
    print(f"  log-lik best single  = {ll_best_single:.2f}   AIC = {aic_best_single:.1f}")
    print(f"  LR test  chi2({dof}) = {lr_stat:.2f},  p ≈ {p_lr:.4f}")
    better = "joint model justified" if (aic_joint < aic_best_single) else "single suffices"
    print(f"  -> {better} (lower AIC)")

    # 2-D risk surface over the two most influential structures (by |coef|).
    top2 = np.argsort(-np.abs(joint.coef_[0]))[:2]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = int(top2[0]), int(top2[1])
    ga = np.linspace(X[:, a].min(), np.percentile(X[:, a], 98), 60)
    gb = np.linspace(X[:, b].min(), np.percentile(X[:, b], 98), 60)
    GA, GB = np.meshgrid(ga, gb)
    grid = np.tile(np.median(X, axis=0), (GA.size, 1))
    grid[:, a] = GA.ravel(); grid[:, b] = GB.ravel()
    P = joint.predict_proba(grid)[:, 1].reshape(GA.shape)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cs = ax.contourf(GA, GB, P * 100, levels=12, cmap="RdYlBu_r")
    ax.scatter(X[:, a], X[:, b], c=y, cmap="binary", edgecolor="k", s=18, alpha=0.5)
    fig.colorbar(cs, label=f"P({args.endpoint}) %")
    ax.set_xlabel(f"{args.structures[a]} mean dose (Gy)")
    ax.set_ylabel(f"{args.structures[b]} mean dose (Gy)")
    ax.set_title("Joint NTCP risk surface")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "joint_ntcp_surface.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "joint_ntcp_results.json"), "w") as f:
        json.dump({"endpoint": args.endpoint, "event_rate": float(y.mean()),
                   "structures": args.structures, "odds_ratios": rows,
                   "loglik_joint": ll_joint, "aic_joint": aic_joint,
                   "best_single": args.structures[best_j],
                   "loglik_best_single": ll_best_single, "aic_best_single": aic_best_single,
                   "lr_stat": float(lr_stat), "lr_dof": dof, "lr_p": float(p_lr)}, f, indent=2)
    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  joint_ntcp_surface.png, joint_ntcp_results.json")


if __name__ == "__main__":
    main()
