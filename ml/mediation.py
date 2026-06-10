"""
Mediation analysis: do early markers carry the dose -> LVEF-decline effect?

Hypothesis
----------
Radiation dose to the left ventricle injures the myocardium; that injury shows up
*first* as subclinical change in early markers (echo GLS, LV-CFD wall shear
stress), and the overt LVEF decline follows. If so, the effect of LV dose on LVEF
decline should be (partly) **mediated** by the early markers.

Model (Baron-Kenny with bootstrap, adjusted for confounders)
------------------------------------------------------------
Exposure  X : LV mean dose (Gy)
Mediators M : early myocardial markers (default: GLS change, LV-CFD WSS change)
Outcome   Y : LVEF decline (percentage points)
Covariates C: cardiotoxic chemo + clinical risk factors (confound M–Y and X–Y)

  a_k : effect of X on mediator k          (M_k ~ X + C)
  b_k : effect of mediator k on Y          (Y ~ X + ΣM + C), holding X, other M
  c'  : direct effect of X on Y            (same model, coefficient on X)
  c   : total effect of X on Y             (Y ~ X + C, no mediators)

  indirect_k        = a_k · b_k            (effect through marker k)
  total indirect    = Σ a_k · b_k
  proportion mediated = total indirect / c

Confidence intervals are percentile bootstrap (resampling patients), which is the
recommended approach for the indirect effect (its sampling distribution is skewed).

Usage
-----
    python make_dataset.py
    python mediation.py
    python mediation.py --exposure LV__mean_dose_gy --outcome lvef_decline \
        --mediators gls_rel_change_pct lv_wss_change --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from make_dataset import MYOCARDIAL_TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

DEFAULT_EXPOSURE = "LV__mean_dose_gy"
DEFAULT_OUTCOME = "lvef_decline"
# GLS and LV-CFD WSS are the myocardial markers on the LVEF causal path. (Energy
# loss is excluded by default as it is weakly identified, but can be added.)
DEFAULT_MEDIATORS = ["gls_rel_change_pct", "lv_wss_change"]
DEFAULT_COVARIATES = [
    "anthracycline", "trastuzumab", "hypertension", "diabetes", "smoker",
    "age", "baseline_lvef",
]


def _ols_coef(X: np.ndarray, y: np.ndarray, target_col: int) -> float:
    """Fit OLS y ~ X (X already includes all regressors) and return one coef."""
    model = LinearRegression().fit(X, y)
    return float(model.coef_[target_col])


def estimate_effects(
    df: pd.DataFrame,
    exposure: str,
    outcome: str,
    mediators: list[str],
    covariates: list[str],
) -> dict:
    """Single point estimate of all mediation quantities."""
    cov = df[covariates].to_numpy(dtype=float)
    x = df[exposure].to_numpy(dtype=float).reshape(-1, 1)
    y = df[outcome].to_numpy(dtype=float)
    M = df[mediators].to_numpy(dtype=float)

    # a_k: exposure -> mediator k, adjusting for covariates. X = [exposure, cov].
    Xa = np.hstack([x, cov])
    a = {m: _ols_coef(Xa, M[:, k], target_col=0) for k, m in enumerate(mediators)}

    # Outcome model Y ~ exposure + mediators + covariates -> c' and b_k.
    Xb = np.hstack([x, M, cov])
    cprime = _ols_coef(Xb, y, target_col=0)
    b = {m: _ols_coef(Xb, y, target_col=1 + k) for k, m in enumerate(mediators)}

    # Total effect Y ~ exposure + covariates.
    c_total = _ols_coef(Xa, y, target_col=0)

    indirect = {m: a[m] * b[m] for m in mediators}
    total_indirect = float(sum(indirect.values()))
    prop_mediated = float(total_indirect / c_total) if c_total != 0 else float("nan")

    return {
        "a": a, "b": b,
        "direct_effect": cprime,
        "total_effect": c_total,
        "indirect_per_mediator": indirect,
        "total_indirect_effect": total_indirect,
        "proportion_mediated": prop_mediated,
    }


def bootstrap_ci(
    df: pd.DataFrame, exposure, outcome, mediators, covariates,
    n_boot: int, seed: int,
) -> dict:
    """Percentile bootstrap CIs for the mediation quantities."""
    rng = np.random.default_rng(seed)
    n = len(df)
    keys_indirect = mediators
    boot = {
        "total_indirect_effect": [],
        "direct_effect": [],
        "proportion_mediated": [],
        **{f"indirect::{m}": [] for m in keys_indirect},
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx]
        try:
            est = estimate_effects(sample, exposure, outcome, mediators, covariates)
        except np.linalg.LinAlgError:
            continue
        boot["total_indirect_effect"].append(est["total_indirect_effect"])
        boot["direct_effect"].append(est["direct_effect"])
        boot["proportion_mediated"].append(est["proportion_mediated"])
        for m in keys_indirect:
            boot[f"indirect::{m}"].append(est["indirect_per_mediator"][m])

    def ci(vals):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {k: ci(v) for k, v in boot.items()}


def plot_path_diagram(est: dict, cis: dict, exposure, outcome, mediators, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(cx, cy, text):
        ax.add_patch(plt.Rectangle((cx - 1.3, cy - 0.6), 2.6, 1.2,
                                   fill=True, fc="#eef3fb", ec="#33558b", lw=1.5))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=9)

    box(1.6, 5, exposure.replace("__", "\n"))
    box(8.4, 5, outcome)
    ys = np.linspace(8.2, 1.8, len(mediators))
    for m, my in zip(mediators, ys):
        box(5.0, my, m)
        ax.annotate("", xy=(3.7, my), xytext=(2.9, 5),
                    arrowprops=dict(arrowstyle="->", color="#888"))
        ax.annotate("", xy=(7.1, 5), xytext=(6.3, my),
                    arrowprops=dict(arrowstyle="->", color="#888"))
        ind = est["indirect_per_mediator"][m]
        lo, hi = cis[f"indirect::{m}"]
        ax.text(5.0, my - 0.95, f"indirect={ind:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]",
                ha="center", va="top", fontsize=7.5, color="#33558b")

    # Direct path arrow.
    ax.annotate("", xy=(7.1, 4.2), xytext=(2.9, 4.2),
                arrowprops=dict(arrowstyle="->", color="#c0504d", lw=1.5))
    dlo, dhi = cis["direct_effect"]
    ax.text(5.0, 3.5, f"direct c'={est['direct_effect']:.3f}  "
            f"[{dlo:.3f}, {dhi:.3f}]", ha="center", fontsize=8, color="#c0504d")

    pm = est["proportion_mediated"] * 100
    plo, phi = [v * 100 for v in cis["proportion_mediated"]]
    ax.set_title(f"Mediation of {exposure} → {outcome}\n"
                 f"proportion mediated = {pm:.0f}%  (95% CI {plo:.0f}–{phi:.0f}%)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mediation analysis of dose -> early markers -> LVEF decline.")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "cohort.csv"))
    ap.add_argument("--exposure", default=DEFAULT_EXPOSURE)
    ap.add_argument("--outcome", default=DEFAULT_OUTCOME)
    ap.add_argument("--mediators", nargs="+", default=DEFAULT_MEDIATORS)
    ap.add_argument("--covariates", nargs="+", default=DEFAULT_COVARIATES)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\nRun `python make_dataset.py` first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(args.data, index_col=0)
    needed = [args.exposure, args.outcome, *args.mediators, *args.covariates]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise SystemExit(f"Columns not in data: {miss}")
    df = df[needed].dropna()

    print(f"Mediation analysis  (n={len(df)})")
    print(f"  exposure  X : {args.exposure}")
    print(f"  outcome   Y : {args.outcome}")
    print(f"  mediators M : {args.mediators}")
    print(f"  covariates  : {args.covariates}\n")

    est = estimate_effects(df, args.exposure, args.outcome, args.mediators, args.covariates)
    cis = bootstrap_ci(df, args.exposure, args.outcome, args.mediators,
                       args.covariates, args.n_boot, args.seed)

    print("Path coefficients (per mediator):")
    for m in args.mediators:
        lo, hi = cis[f"indirect::{m}"]
        sig = "*" if (lo > 0) == (hi > 0) else " "  # CI excludes 0
        print(f"  {m:22s}  a={est['a'][m]:+.4f}  b={est['b'][m]:+.4f}  "
              f"indirect(a·b)={est['indirect_per_mediator'][m]:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")

    print("\nDecomposition of the LV-dose effect on LVEF decline:")
    tlo, thi = cis["total_indirect_effect"]
    dlo, dhi = cis["direct_effect"]
    plo, phi = cis["proportion_mediated"]
    print(f"  total effect c      = {est['total_effect']:+.4f}")
    print(f"  direct effect c'    = {est['direct_effect']:+.4f}   95% CI [{dlo:+.4f}, {dhi:+.4f}]")
    print(f"  total indirect a·b  = {est['total_indirect_effect']:+.4f}   95% CI [{tlo:+.4f}, {thi:+.4f}]")
    print(f"  proportion mediated = {est['proportion_mediated']*100:.1f}%   "
          f"95% CI [{plo*100:.1f}%, {phi*100:.1f}%]")
    mediated_sig = "YES" if (tlo > 0) == (thi > 0) else "NO"
    print(f"\n  -> Significant mediation (indirect CI excludes 0): {mediated_sig}")

    plot_path_diagram(est, cis, args.exposure, args.outcome, args.mediators,
                      os.path.join(OUT_DIR, "mediation_path.png"))
    with open(os.path.join(OUT_DIR, "mediation_results.json"), "w") as f:
        json.dump({"config": {"exposure": args.exposure, "outcome": args.outcome,
                              "mediators": args.mediators, "covariates": args.covariates,
                              "n": int(len(df)), "n_boot": args.n_boot},
                   "estimates": est, "ci95": cis}, f, indent=2)
    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  mediation_results.json, mediation_path.png")


if __name__ == "__main__":
    main()
