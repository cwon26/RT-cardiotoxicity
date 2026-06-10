"""
Substructure NTCP (normal-tissue complication probability) dose-response curves.

For each cardiac substructure we fit a logistic dose-response model relating its
mean dose to the probability of a complication:

    P(complication | D) = 1 / (1 + exp(-(b0 + b1 · D)))

and report the standard radiobiology summaries:

    D50      = -b0 / b1            dose giving 50% complication probability (Gy)
    gamma50  = b1 · D50 / 4        normalized dose-response steepness at D50

Curves come with percentile-bootstrap confidence bands. This is the form the
radiation-oncology literature uses to communicate cardiac dose constraints, and
it pairs each endpoint with the substructure that drives it:

    heart mean dose -> CTRCD (clinical LVEF-defined cardiotoxicity)
    LV    mean dose -> CTRCD
    LAD   mean dose -> coronary subclinical injury (large drop in coronary WSS)

Usage
-----
    python make_dataset.py
    python ntcp.py
    python ntcp.py --pairs LV:cardiotoxicity LAD:coronary_event
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

# structure : (mean-dose column, endpoint column)
DEFAULT_PAIRS = ["heart:cardiotoxicity", "LV:cardiotoxicity", "LAD:coronary_event"]


def derive_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary endpoints derived from the continuous markers (documented
    thresholds), so dose-response can be fit for the coronary / strain domains."""
    df = df.copy()
    if "coronary_event" not in df:
        # >0.15 Pa reduction in coronary wall shear stress = subclinical injury.
        df["coronary_event"] = (df["lad_wss_change"] <= -0.15).astype(int)
    if "strain_abnormal" not in df:
        # ESC cardio-oncology: >=15% relative GLS reduction.
        df["strain_abnormal"] = (df["gls_rel_change_pct"] >= 15.0).astype(int)
    return df


def fit_logistic_dose_response(dose: np.ndarray, y: np.ndarray) -> dict:
    """MLE logistic fit P(y=1|dose). Returns b0, b1, D50, gamma50."""
    # C=inf -> effectively unpenalized MLE (penalty=None is deprecated in sklearn 1.8+).
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    clf.fit(dose.reshape(-1, 1), y)
    b1 = float(clf.coef_[0, 0])
    b0 = float(clf.intercept_[0])
    d50 = -b0 / b1 if b1 != 0 else float("nan")
    gamma50 = b1 * d50 / 4.0 if np.isfinite(d50) else float("nan")
    return {"b0": b0, "b1": b1, "D50": d50, "gamma50": gamma50}


def bootstrap_curve(dose, y, grid, n_boot, seed):
    """Bootstrap NTCP curve band and D50/gamma50 CIs."""
    rng = np.random.default_rng(seed)
    n = len(dose)
    curves, d50s, g50s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():       # need both classes to fit
            continue
        try:
            fit = fit_logistic_dose_response(dose[idx], yb)
        except Exception:
            continue
        p = 1.0 / (1.0 + np.exp(-(fit["b0"] + fit["b1"] * grid)))
        curves.append(p)
        d50s.append(fit["D50"]); g50s.append(fit["gamma50"])
    curves = np.asarray(curves)
    band = (np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0))
    def ci(v):
        a = np.asarray(v); a = a[np.isfinite(a)]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return band, ci(d50s), ci(g50s)


def analyze_pair(df, structure, endpoint, n_boot, seed):
    dose_col = f"{structure}__mean_dose_gy"
    if dose_col not in df or endpoint not in df:
        raise SystemExit(f"Missing columns: {dose_col} / {endpoint}")
    d = df[dose_col].to_numpy(float)
    y = df[endpoint].to_numpy(int)
    fit = fit_logistic_dose_response(d, y)
    grid = np.linspace(0, float(np.percentile(d, 99)) + 2, 200)
    p_hat = 1.0 / (1.0 + np.exp(-(fit["b0"] + fit["b1"] * grid)))
    band, d50_ci, g50_ci = bootstrap_curve(d, y, grid, n_boot, seed)
    return {
        "structure": structure, "endpoint": endpoint, "event_rate": float(y.mean()),
        "fit": fit, "D50_ci": d50_ci, "gamma50_ci": g50_ci,
        "grid": grid, "p_hat": p_hat, "band": band, "dose": d, "y": y,
    }


def plot_panels(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        grid, p, (lo, hi) = r["grid"], r["p_hat"], r["band"]
        ax.fill_between(grid, lo, hi, color="#9ecae1", alpha=0.5, label="95% CI")
        ax.plot(grid, p, color="#08519c", lw=2, label="NTCP fit")
        # jittered observed points
        rngj = np.random.default_rng(0)
        jit = rngj.uniform(-0.025, 0.025, size=len(r["y"]))
        ax.scatter(r["dose"], r["y"] + jit, s=10, alpha=0.3, color="#444")
        d50 = r["fit"]["D50"]
        if np.isfinite(d50):
            ax.axvline(d50, color="#c0504d", ls="--", lw=1)
            ax.text(d50, 0.55, f" D50={d50:.1f} Gy", color="#c0504d", fontsize=8)
        ax.set_title(f"{r['structure']} dose → {r['endpoint']}\n"
                     f"γ50={r['fit']['gamma50']:.2f}, event rate={r['event_rate']:.0%}",
                     fontsize=9)
        ax.set_xlabel("Mean dose (Gy)"); ax.set_ylabel("Complication probability")
        ax.set_ylim(-0.08, 1.08); ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Substructure NTCP dose-response curves.")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "cohort.csv"))
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                    help="structure:endpoint pairs, e.g. LV:cardiotoxicity LAD:coronary_event")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\nRun `python make_dataset.py` first.")
    os.makedirs(OUT_DIR, exist_ok=True)
    df = derive_endpoints(pd.read_csv(args.data, index_col=0))

    results, summary = [], []
    print(f"Substructure NTCP dose-response (n={len(df)})\n")
    for pair in args.pairs:
        structure, endpoint = pair.split(":")
        r = analyze_pair(df, structure, endpoint, args.n_boot, args.seed)
        results.append(r)
        f = r["fit"]
        print(f"{structure:6s} mean dose -> {endpoint}")
        print(f"   event rate = {r['event_rate']:.1%}")
        print(f"   D50     = {f['D50']:6.2f} Gy   95% CI [{r['D50_ci'][0]:.1f}, {r['D50_ci'][1]:.1f}]")
        print(f"   gamma50 = {f['gamma50']:6.2f}      95% CI [{r['gamma50_ci'][0]:.2f}, {r['gamma50_ci'][1]:.2f}]")
        print()
        summary.append({"structure": structure, "endpoint": endpoint,
                        "event_rate": r["event_rate"], "fit": f,
                        "D50_ci": r["D50_ci"], "gamma50_ci": r["gamma50_ci"]})

    plot_panels(results, os.path.join(OUT_DIR, "ntcp_curves.png"))
    with open(os.path.join(OUT_DIR, "ntcp_results.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Artifacts written to {OUT_DIR}/")
    print("  ntcp_curves.png, ntcp_results.json")


if __name__ == "__main__":
    main()
