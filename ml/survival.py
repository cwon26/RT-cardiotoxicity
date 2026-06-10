"""
Competing-risks survival analysis for time to cardiac events.

A patient can have a cardiac event, die of a non-cardiac cause first (a competing
risk that *prevents* the cardiac event), or be censored. Ignoring competition and
using naive Kaplan-Meier overstates cardiac risk, so we use:

  * cause-specific Cox proportional-hazards model for the cardiac hazard
    (hazard ratios for dose / clinical predictors, with Wald CIs);
  * Aalen-Johansen cumulative incidence function (CIF) of cardiac events,
    by LV-dose group, which properly accounts for the competing death;
  * Harrell's C-index for discrimination.

Everything is implemented with numpy (no compiled survival package needed).

Usage
-----
    python make_dataset.py
    python survival.py
    python survival.py --predictors LV__mean_dose_gy anthracycline age
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

DEFAULT_PREDICTORS = ["LV__mean_dose_gy", "LAD__mean_dose_gy", "anthracycline",
                      "trastuzumab", "age", "baseline_lvef"]


# --------------------------------------------------------------------------- #
# Cox proportional hazards (Breslow ties), Newton-Raphson
# --------------------------------------------------------------------------- #
def cox_fit(X: np.ndarray, time: np.ndarray, event: np.ndarray, max_iter=50, tol=1e-8):
    """Cause-specific Cox PH. event=1 for the event of interest, 0 otherwise.

    Returns beta, standard errors, and the per-iteration convergence flag.
    """
    n, p = X.shape
    order = np.argsort(-time)               # descending time -> cumulative risk sets
    Xo, to, eo = X[order], time[order], event[order]
    beta = np.zeros(p)

    for _ in range(max_iter):
        eta = Xo @ beta
        w = np.exp(eta)
        # cumulative sums over descending time = risk-set sums (ties handled Breslow)
        S0 = np.cumsum(w)                                  # (n,)
        S1 = np.cumsum(w[:, None] * Xo, axis=0)            # (n,p)
        # S2 per-row outer products accumulated
        WX = w[:, None] * Xo
        S2 = np.cumsum(np.einsum("ni,nj->nij", WX, Xo), axis=0)  # (n,p,p)

        grad = np.zeros(p)
        hess = np.zeros((p, p))
        ev_idx = np.where(eo == 1)[0]
        for i in ev_idx:
            ratio = S1[i] / S0[i]
            grad += Xo[i] - ratio
            hess -= S2[i] / S0[i] - np.outer(ratio, ratio)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # covariance = inverse observed information (= -Hessian of log-lik)
    try:
        cov = np.linalg.inv(-hess)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return beta, se


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C for the event of interest (higher risk -> earlier event)."""
    n = len(time)
    conc = perm = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[j] > time[i]:           # j is comparable (still at risk after i's event)
                perm += 1
                if risk[i] > risk[j]:
                    conc += 1
                elif risk[i] == risk[j]:
                    conc += 0.5
    return conc / perm if perm > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Aalen-Johansen cumulative incidence (competing risks)
# --------------------------------------------------------------------------- #
def cif_aalen_johansen(time, event_type, cause=1):
    """CIF for `cause` accounting for all competing events. Returns (t, cif)."""
    times = np.sort(np.unique(time[event_type != 0]))
    n = len(time)
    surv_prev = 1.0
    cif = 0.0
    out_t, out_c = [0.0], [0.0]
    at_risk = n
    for t in times:
        d_cause = int(np.sum((time == t) & (event_type == cause)))
        d_any = int(np.sum((time == t) & (event_type != 0)))
        n_risk = int(np.sum(time >= t))
        if n_risk == 0:
            break
        cif += surv_prev * (d_cause / n_risk)
        surv_prev *= (1 - d_any / n_risk)
        out_t.append(float(t)); out_c.append(float(cif))
    return np.array(out_t), np.array(out_c)


def main():
    ap = argparse.ArgumentParser(description="Competing-risks survival analysis.")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "cohort.csv"))
    ap.add_argument("--predictors", nargs="+", default=DEFAULT_PREDICTORS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\nRun `python make_dataset.py` first.")
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(args.data, index_col=0)
    for c in ("event_time", "event_type", "cardiac_event", *args.predictors):
        if c not in df.columns:
            raise SystemExit(f"Column missing: {c}")

    time = df["event_time"].to_numpy(float)
    etype = df["event_type"].to_numpy(int)
    cardiac = (etype == 1).astype(int)

    # Standardize predictors for interpretable, stable HRs (per 1 SD).
    Xraw = df[args.predictors].to_numpy(float)
    mu, sd = Xraw.mean(0), Xraw.std(0)
    sd[sd == 0] = 1.0
    X = (Xraw - mu) / sd

    beta, se = cox_fit(X, time, cardiac)
    z = beta / se
    hr = np.exp(beta)
    hr_lo, hr_hi = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)

    n_card = int(cardiac.sum()); n_death = int((etype == 2).sum())
    print(f"Competing-risks survival  (n={len(df)})")
    print(f"  cardiac events = {n_card} ({cardiac.mean():.1%}),  "
          f"competing deaths = {n_death} ({(etype==2).mean():.1%}),  "
          f"censored = {(etype==0).mean():.1%}")
    print(f"  median follow-up = {np.median(time):.2f} y\n")

    print("Cause-specific Cox model for cardiac events (HR per 1 SD):")
    print(f"  {'predictor':22s} {'HR':>6s}  {'95% CI':>16s}   {'p':>7s}")
    from math import erf, sqrt
    rows = []
    for k, name in enumerate(args.predictors):
        p_two = 2 * (1 - 0.5 * (1 + erf(abs(z[k]) / sqrt(2))))
        sig = "*" if p_two < 0.05 else " "
        print(f"  {name:22s} {hr[k]:6.2f}  [{hr_lo[k]:5.2f}, {hr_hi[k]:5.2f}]   {p_two:7.4f} {sig}")
        rows.append({"predictor": name, "HR": float(hr[k]),
                     "ci95": [float(hr_lo[k]), float(hr_hi[k])], "p": float(p_two)})

    risk = X @ beta
    cidx = concordance_index(risk, time, cardiac)
    print(f"\n  Harrell's C-index (cardiac) = {cidx:.3f}")

    # CIF by LV-dose group (median split), with competing risk handled.
    lv = df["LV__mean_dose_gy"].to_numpy(float)
    hi = lv >= np.median(lv)
    t_hi, c_hi = cif_aalen_johansen(time[hi], etype[hi], cause=1)
    t_lo, c_lo = cif_aalen_johansen(time[~hi], etype[~hi], cause=1)
    print(f"\n10-y cumulative incidence of cardiac events (Aalen-Johansen):")
    print(f"  high LV dose (≥ median): {np.interp(10, t_hi, c_hi):.1%}")
    print(f"  low  LV dose (< median): {np.interp(10, t_lo, c_lo):.1%}")

    # Figures.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].step(t_hi, c_hi * 100, where="post", color="#c0504d", lw=2, label="LV dose ≥ median")
    axes[0].step(t_lo, c_lo * 100, where="post", color="#08519c", lw=2, label="LV dose < median")
    axes[0].set_xlabel("Years"); axes[0].set_ylabel("Cumulative incidence (%)")
    axes[0].set_title("Cardiac-event CIF by LV dose\n(competing risk: non-cardiac death)")
    axes[0].legend(fontsize=8); axes[0].set_xlim(0, 10)

    ypos = np.arange(len(args.predictors))[::-1]
    axes[1].errorbar(hr, ypos, xerr=[hr - hr_lo, hr_hi - hr], fmt="o", color="#08519c", capsize=3)
    axes[1].axvline(1.0, color="k", ls="--", lw=1)
    axes[1].set_yticks(ypos); axes[1].set_yticklabels(args.predictors, fontsize=8)
    axes[1].set_xlabel("Hazard ratio (per 1 SD)"); axes[1].set_title("Cause-specific Cox HRs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "survival_cif.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "survival_results.json"), "w") as f:
        json.dump({"n": len(df), "n_cardiac": n_card, "n_competing_death": n_death,
                   "c_index": float(cidx), "cox_hr": rows,
                   "cif10y_high_lv": float(np.interp(10, t_hi, c_hi)),
                   "cif10y_low_lv": float(np.interp(10, t_lo, c_lo))}, f, indent=2)
    print(f"\nArtifacts written to {OUT_DIR}/")
    print("  survival_cif.png, survival_results.json")


if __name__ == "__main__":
    main()
