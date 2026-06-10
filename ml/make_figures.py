"""
Build the figure set for the RT-cardiotoxicity analyses.

Reads the saved result JSONs / CSVs in outputs/ and renders:

  1. causal_mediation_effects.png  — NDE / NIE / TE decomposition with CIs
  2. pls_vip.png                   — top PLS VIP predictors (substructure-colored)
  3. summary_dashboard.png         — one-page overview of every analysis

The per-analysis figures (NTCP curves, survival CIF, calibration, fPCA, PLS
predicted-vs-actual, mediation path, joint NTCP surface) are produced by their own
scripts; run those first (or `python run_all.py`) so the JSONs exist.

Usage
-----
    python make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

BLUE, RED, GREEN, GREY = "#08519c", "#c0504d", "#2ca25f", "#888888"
SUB_COLORS = {"heart": "#6baed6", "LV": "#fd8d3c", "LAD": "#e6550d",
              "RCA": "#74c476", "LCX": "#9e9ac8"}


def _load(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
def fig_causal_mediation():
    cm = _load("causal_mediation_results.json")
    bk = _load("mediation_results.json")
    if cm is None:
        return
    est, ci = cm["estimates"], cm["ci95"]
    names = ["NDE\n(direct)", "NIE\n(indirect)", "TE\n(total)"]
    keys = ["NDE", "NIE", "TE"]
    vals = [est[k] for k in keys]
    los = [est[k] - ci[k][0] for k in keys]
    his = [ci[k][1] - est[k] for k in keys]
    colors = [RED, GREEN, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.bar(names, vals, yerr=[los, his], color=colors, capsize=5, alpha=0.9)
    ax.axhline(0, color="k", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + (his[i] + 0.05), f"{v:+.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Effect on LVEF decline (points)")
    ax.set_title("Causal mediation of LV dose → LVEF decline\n(potential-outcomes natural effects, 95% CI)")

    # proportion mediated: causal vs Baron-Kenny
    ax2 = axes[1]
    pm_c = est["proportion_mediated"] * 100
    pm_c_ci = [c * 100 for c in ci["proportion_mediated"]]
    bars = [("Causal\n(GLS)", pm_c, pm_c_ci, GREEN)]
    if bk is not None:
        pm_b = bk["estimates"]["proportion_mediated"] * 100
        pm_b_ci = [c * 100 for c in bk["ci95"]["proportion_mediated"]]
        bars.append(("Baron-Kenny\n(GLS+WSS)", pm_b, pm_b_ci, BLUE))
    xs = range(len(bars))
    ax2.bar([b[0] for b in bars], [b[1] for b in bars],
            yerr=[[b[1] - b[2][0] for b in bars], [b[2][1] - b[1] for b in bars]],
            color=[b[3] for b in bars], capsize=5, alpha=0.9)
    for i, b in enumerate(bars):
        ax2.text(i, b[1] + 3, f"{b[1]:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 100); ax2.set_ylabel("Proportion mediated (%)")
    ax2.set_title("Mediated share")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "causal_mediation_effects.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_pls_vip(top_n=15):
    path = os.path.join(OUT_DIR, "vip_scores.csv")
    if not os.path.exists(path):
        return
    vip = pd.read_csv(path).head(top_n).iloc[::-1]
    def color(feat):
        for s, c in SUB_COLORS.items():
            if feat.startswith(f"{s}__"):
                return c
        return GREY
    colors = [color(f) for f in vip["feature"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(vip["feature"], vip["vip"], color=colors)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.text(1.02, 0.3, "VIP = 1", rotation=90, fontsize=8, color="k")
    ax.set_xlabel("VIP score (>1 = influential)")
    ax.set_title("PLS variable importance (top predictors)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SUB_COLORS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=GREY))
    ax.legend(handles, [*SUB_COLORS.keys(), "clinical/baseline"], fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pls_vip.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_dashboard():
    pls = _load("metrics_pls.json")
    med = _load("mediation_results.json")
    cm = _load("causal_mediation_results.json")
    ntcp = _load("ntcp_results.json")
    surv = _load("survival_results.json")
    ext = _load("external_validation_results.json")
    joint = _load("joint_ntcp_results.json")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (1) PLS test R2 per target
    ax = axes[0, 0]
    if pls:
        tg = list(pls["test"].keys())
        r2 = [pls["test"][t]["r2"] for t in tg]
        cols = [RED if t.startswith("lad") else BLUE for t in tg]
        ax.barh([t.replace("_change", "").replace("_rel", "") for t in tg], r2, color=cols)
        ax.set_xlabel("Held-out R²"); ax.set_xlim(0, max(0.5, max(r2) + 0.1))
    ax.set_title("① PLS: early-marker prediction\n(blue=myocardial, red=coronary)", fontsize=10)

    # (2) Mediation proportion
    ax = axes[0, 1]
    labels, vals, errs, cols = [], [], [], []
    if med:
        pm = med["estimates"]["proportion_mediated"] * 100
        ci = [c * 100 for c in med["ci95"]["proportion_mediated"]]
        labels.append("Baron-Kenny"); vals.append(pm); errs.append([pm - ci[0], ci[1] - pm]); cols.append(BLUE)
    if cm:
        pm = cm["estimates"]["proportion_mediated"] * 100
        ci = [c * 100 for c in cm["ci95"]["proportion_mediated"]]
        labels.append("Causal"); vals.append(pm); errs.append([pm - ci[0], ci[1] - pm]); cols.append(GREEN)
    if vals:
        errs = np.array(errs).T
        ax.bar(labels, vals, yerr=errs, color=cols, capsize=5)
        for i, v in enumerate(vals):
            ax.text(i, v + 4, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 100); ax.set_ylabel("% mediated")
    ax.set_title("② Mediation: dose→marker→LVEF", fontsize=10)

    # (3) NTCP D50 by structure/endpoint
    ax = axes[0, 2]
    if ntcp:
        labs = [f"{r['structure']}→\n{r['endpoint'][:8]}" for r in ntcp]
        d50 = [r["fit"]["D50"] for r in ntcp]
        d50ci = [r["D50_ci"] for r in ntcp]
        lo = [max(0, d - c[0]) for d, c in zip(d50, d50ci)]
        hi = [max(0, c[1] - d) for d, c in zip(d50, d50ci)]
        ax.bar(labs, d50, yerr=[lo, hi], color=BLUE, capsize=4)
        ax.set_ylabel("D50 (Gy)"); ax.set_ylim(0, min(60, max(d50) * 1.5))
    ax.set_title("③ NTCP dose-response (D50)", fontsize=10)

    # (4) Survival CIF 10y high vs low LV dose
    ax = axes[1, 0]
    if surv:
        hi_v = surv["cif10y_high_lv"] * 100
        lo_v = surv["cif10y_low_lv"] * 100
        ax.bar(["LV dose\n≥ median", "LV dose\n< median"], [hi_v, lo_v], color=[RED, BLUE])
        for i, v in enumerate([hi_v, lo_v]):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
        ax.set_ylabel("10-y cardiac incidence (%)")
        ax.text(0.5, 0.92, f"C-index = {surv['c_index']:.2f}", transform=ax.transAxes,
                ha="center", fontsize=9, color=GREY)
    ax.set_title("④ Competing-risks survival (CIF)", fontsize=10)

    # (5) External validation: AUC + calibration slope
    ax = axes[1, 1]
    if ext:
        stages = [("dev", "development"), ("ext", "external"), ("recal", "external_recalibrated")]
        keys = ["development", "external", "external_recalibrated"]
        auc = [ext[k]["roc_auc"] for k in keys]
        slope = [ext[k]["calib_slope"] for k in keys]
        x = np.arange(3); w = 0.38
        ax.bar(x - w / 2, auc, w, label="ROC-AUC", color=BLUE)
        ax.bar(x + w / 2, slope, w, label="calib. slope", color=RED)
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(["dev", "external", "ext+recal"], fontsize=8)
        ax.legend(fontsize=7); ax.set_ylim(0, 2.0)
    ax.set_title("⑤ External validation\n(calib. slope ideal=1)", fontsize=10)

    # (6) Joint NTCP coronary: adjusted vs single OR
    ax = axes[1, 2]
    if joint:
        ors = joint["odds_ratios"]
        labs = [o["structure"] for o in ors]
        adj = [o["adjusted_or_per_gy"] for o in ors]
        single = [o["single_or_per_gy"] for o in ors]
        x = np.arange(len(labs)); w = 0.38
        ax.bar(x - w / 2, single, w, label="single-structure", color=GREY)
        ax.bar(x + w / 2, adj, w, label="mutually adjusted", color=RED)
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labs)
        ax.set_ylabel("OR per Gy"); ax.legend(fontsize=7)
        ax.set_title(f"⑥ Joint NTCP: {joint['endpoint']}\n(only LAD survives adjustment)", fontsize=10)

    fig.suptitle("RT cardiotoxicity — analysis summary (synthetic cohort, N=300)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_DIR, "summary_dashboard.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_causal_mediation()
    fig_pls_vip()
    fig_dashboard()
    print("Wrote figures to outputs/:")
    print("  causal_mediation_effects.png, pls_vip.png, summary_dashboard.png")


if __name__ == "__main__":
    main()
