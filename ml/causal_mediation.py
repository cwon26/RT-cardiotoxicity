"""
Causal mediation (potential-outcomes) — robustness check on the Baron-Kenny result.

The Baron-Kenny product-of-coefficients in `mediation.py` assumes no exposure–
mediator interaction. The potential-outcomes framework relaxes that: it defines
natural direct and indirect effects through counterfactual mediator values and
allows an X·M interaction in the outcome model.

For a single mediator M, exposure contrast x0 -> x1, covariates C:

    mediator model : E[M | x, C] = α0 + αx·x + αc·C
    outcome model  : E[Y | x, m, C] = β0 + βx·x + βm·m + βxm·(x·m) + βc·C

    NDE (natural direct)   = E[ Y(x1, M(x0)) − Y(x0, M(x0)) ]
    NIE (natural indirect) = E[ Y(x1, M(x1)) − Y(x1, M(x0)) ]
    TE  = NDE + NIE,   proportion mediated = NIE / TE

For linear mediator and outcome models these expectations are exact using
E[M | x, C] (no Monte-Carlo needed), averaged over the observed covariate
distribution. CIs are percentile bootstrap (models refit per resample).

Usage
-----
    python causal_mediation.py
    python causal_mediation.py --mediator lv_wss_change --x-lo 5 --x-hi 25
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

DEFAULT_EXPOSURE = "LV__mean_dose_gy"
DEFAULT_MEDIATOR = "gls_rel_change_pct"
DEFAULT_OUTCOME = "lvef_decline"
DEFAULT_COVARIATES = ["anthracycline", "trastuzumab", "hypertension", "diabetes",
                      "smoker", "age", "baseline_lvef"]


def _fit_effects(df, exposure, mediator, outcome, covariates, x0, x1):
    """Point estimate of NDE / NIE / TE / proportion mediated and the X·M coef."""
    C = df[covariates].to_numpy(float)
    x = df[exposure].to_numpy(float)
    m = df[mediator].to_numpy(float)
    y = df[outcome].to_numpy(float)

    # Mediator model: m ~ x + C
    Xm = np.column_stack([x, C])
    med = LinearRegression().fit(Xm, m)

    def mu_m(xval):
        # E[M | x=xval, C] for every patient's covariates
        return med.intercept_ + med.coef_[0] * xval + C @ med.coef_[1:]

    # Outcome model: y ~ x + m + x*m + C
    Xy = np.column_stack([x, m, x * m, C])
    out = LinearRegression().fit(Xy, y)
    b_x, b_m, b_xm = out.coef_[0], out.coef_[1], out.coef_[2]
    b_c = out.coef_[3:]
    b0 = out.intercept_

    def y_pred(xval, mval):
        return b0 + b_x * xval + b_m * mval + b_xm * (xval * mval) + C @ b_c

    m_x0, m_x1 = mu_m(x0), mu_m(x1)
    nde = float(np.mean(y_pred(x1, m_x0) - y_pred(x0, m_x0)))
    nie = float(np.mean(y_pred(x1, m_x1) - y_pred(x1, m_x0)))
    te = nde + nie
    prop = float(nie / te) if te != 0 else float("nan")
    return {"NDE": nde, "NIE": nie, "TE": te, "proportion_mediated": prop,
            "interaction_coef": float(b_xm)}


def bootstrap(df, exposure, mediator, outcome, covariates, x0, x1, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(df)
    keys = ["NDE", "NIE", "TE", "proportion_mediated", "interaction_coef"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = df.iloc[rng.integers(0, n, size=n)]
        try:
            est = _fit_effects(sample, exposure, mediator, outcome, covariates, x0, x1)
        except np.linalg.LinAlgError:
            continue
        for k in keys:
            acc[k].append(est[k])

    def ci(v):
        a = np.asarray(v, float); a = a[np.isfinite(a)]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {k: ci(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description="Potential-outcomes causal mediation.")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "cohort.csv"))
    ap.add_argument("--exposure", default=DEFAULT_EXPOSURE)
    ap.add_argument("--mediator", default=DEFAULT_MEDIATOR)
    ap.add_argument("--outcome", default=DEFAULT_OUTCOME)
    ap.add_argument("--covariates", nargs="+", default=DEFAULT_COVARIATES)
    ap.add_argument("--x-lo", type=float, default=None, help="Low exposure (default: 25th pctile).")
    ap.add_argument("--x-hi", type=float, default=None, help="High exposure (default: 75th pctile).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\nRun `python make_dataset.py` first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(args.data, index_col=0)
    needed = [args.exposure, args.mediator, args.outcome, *args.covariates]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise SystemExit(f"Columns not in data: {miss}")
    df = df[needed].dropna()

    x = df[args.exposure].to_numpy(float)
    x0 = args.x_lo if args.x_lo is not None else float(np.percentile(x, 25))
    x1 = args.x_hi if args.x_hi is not None else float(np.percentile(x, 75))

    print(f"Causal mediation (potential outcomes)  n={len(df)}")
    print(f"  exposure  : {args.exposure}   contrast {x0:.2f} -> {x1:.2f} Gy")
    print(f"  mediator  : {args.mediator}")
    print(f"  outcome   : {args.outcome}")
    print(f"  covariates: {args.covariates}\n")

    est = _fit_effects(df, args.exposure, args.mediator, args.outcome, args.covariates, x0, x1)
    cis = bootstrap(df, args.exposure, args.mediator, args.outcome, args.covariates,
                    x0, x1, args.n_boot, args.seed)

    def line(name, key, unit=""):
        lo, hi = cis[key]
        sig = "*" if (lo > 0) == (hi > 0) else " "
        print(f"  {name:26s} {est[key]:+.4f}{unit}   95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")

    print("Natural effects of the exposure contrast on LVEF decline:")
    line("NDE (direct)", "NDE")
    line("NIE (indirect, mediated)", "NIE")
    line("TE (total)", "TE")
    pm_lo, pm_hi = cis["proportion_mediated"]
    print(f"  {'proportion mediated':26s} {est['proportion_mediated']*100:+.1f}%"
          f"     95% CI [{pm_lo*100:.1f}%, {pm_hi*100:.1f}%]")
    ix_lo, ix_hi = cis["interaction_coef"]
    ix_sig = "present" if (ix_lo > 0) == (ix_hi > 0) else "negligible"
    print(f"\n  X·M interaction coef = {est['interaction_coef']:+.4f}  "
          f"95% CI [{ix_lo:+.4f}, {ix_hi:+.4f}]  -> {ix_sig}")
    nie_lo, nie_hi = cis["NIE"]
    print(f"  Significant mediation (NIE CI excludes 0): "
          f"{'YES' if (nie_lo > 0) == (nie_hi > 0) else 'NO'}")

    with open(os.path.join(OUT_DIR, "causal_mediation_results.json"), "w") as f:
        json.dump({"config": {"exposure": args.exposure, "mediator": args.mediator,
                              "outcome": args.outcome, "covariates": args.covariates,
                              "x0": x0, "x1": x1, "n": int(len(df)), "n_boot": args.n_boot},
                   "estimates": est, "ci95": cis}, f, indent=2)
    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  causal_mediation_results.json")


if __name__ == "__main__":
    main()
