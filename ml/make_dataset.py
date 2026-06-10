"""
Synthetic RT-cardiotoxicity cohort generator (substructure-resolved).

Real patient data (CT, RTSTRUCT, RTDOSE, echo, CFD) cannot live in this repo, so
this script fabricates a plausible cohort and serves as the schema reference.

Key design choice: dose is resolved **per cardiac substructure** (whole heart,
left ventricle, and the three main coronary arteries LAD / RCA / LCX), because
the relevant endpoints are driven by the dose to *that* structure, not by
whole-heart mean dose. Left-sided breast RT in particular delivers focal dose to
the LAD, which runs in the anterior interventricular groove right under the
tangent fields.

Two latent injury processes are simulated, each feeding its own early markers:

  * Myocardial injury  (driven by LV dose + cardiotoxic chemo + clinical risk)
      -> gls_rel_change_pct        (echo global longitudinal strain, % change)
      -> lv_wss_change             (LV CFD endocardial wall shear stress, Pa)
      -> lv_energy_loss_change     (LV CFD intraventricular energy loss, mW)
  * Coronary injury    (driven by LAD dose + atherosclerotic risk factors)
      -> lad_wss_change            (coronary CFD wall shear stress, Pa)
      -> lad_osi_change            (coronary CFD oscillatory shear index, unitless)

The late LVEF decline is a downstream consequence of myocardial injury and is
kept only as a reference endpoint for the secondary classifier.

Output: ml/data/cohort.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from dvh_features import extract_substructure_features

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

SUBSTRUCTURES = ("heart", "LV", "LAD", "RCA", "LCX")

# Per-substructure dose model. For each structure: voxel-count range, the bath
# mean dose (Gy) for left / right laterality, the bath spread, and the focal
# hot-spot fraction + mean for left-sided plans. Values are illustrative but
# ordered as in breast RT: LAD takes the highest focal dose on the left.
_DOSE_MODEL = {
    #            n_lo,  n_hi,  bath_L, bath_R, sd,  hot_frac_L, hot_mean_L
    "heart": (18000, 25000, 4.5, 1.6, 2.2, 0.10, 30.0),
    "LV":    (6000,  9000,  5.5, 1.8, 2.4, 0.12, 28.0),
    "LAD":   (300,   700,   14.0, 3.5, 5.0, 0.30, 36.0),
    "RCA":   (250,   600,   3.0, 1.2, 2.0, 0.05, 18.0),
    "LCX":   (250,   600,   4.0, 1.5, 2.2, 0.08, 22.0),
}


def _simulate_substructure_dose(rng, struct: str, left: bool) -> np.ndarray:
    """Per-voxel dose array (Gy) for one cardiac substructure."""
    n_lo, n_hi, bath_L, bath_R, sd, hot_frac_L, hot_mean_L = _DOSE_MODEL[struct]
    n_vox = int(rng.integers(n_lo, n_hi))
    bath_mean = bath_L if left else bath_R
    # add patient-to-patient variation around the structure's typical level
    bath_mean = max(0.1, bath_mean * rng.uniform(0.7, 1.3))
    dose = rng.normal(bath_mean, sd, size=n_vox)

    hot_frac = hot_frac_L * rng.uniform(0.5, 1.3) if left else hot_frac_L * 0.2 * rng.random()
    n_hot = int(hot_frac * n_vox)
    if n_hot > 0:
        dose[:n_hot] = rng.normal(hot_mean_L, 5.0, size=n_hot)
    dose = np.clip(dose, 0, 55)
    rng.shuffle(dose)
    return dose


def generate_cohort(n_patients: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for i in range(n_patients):
        pid = f"PT{i:04d}"
        laterality = rng.choice(["left", "right"], p=[0.55, 0.45])
        left = laterality == "left"

        per_sub_dose = {s: _simulate_substructure_dose(rng, s, left) for s in SUBSTRUCTURES}
        feats = extract_substructure_features(per_sub_dose, voxel_volume_cc=0.002)

        # Clinical covariates.
        age = float(np.clip(rng.normal(58, 11), 28, 88))
        anthracycline = int(rng.random() < 0.45)
        trastuzumab = int(rng.random() < 0.30)
        hypertension = int(rng.random() < 0.35)
        diabetes = int(rng.random() < 0.18)
        smoker = int(rng.random() < 0.22)
        baseline_lvef = float(np.clip(rng.normal(62, 4.5), 50, 72))

        # ---- Latent injury 1: myocardial (LV dose-driven) --------------------
        lv_mean = feats["LV__mean_dose_gy"]
        lv_v25 = feats.get("LV__V25Gy_cc", 0.0)
        myo_injury = (
            0.45 * lv_mean
            + 0.020 * lv_v25
            + 3.2 * anthracycline
            + 2.4 * trastuzumab
            + 1.0 * hypertension
            + 0.8 * diabetes
            + 0.6 * smoker
            + 0.10 * max(0.0, age - 55)
        )
        myo_z = myo_injury / 10.0

        # ---- Latent injury 2: coronary (LAD dose + atherosclerotic risk) -----
        lad_mean = feats["LAD__mean_dose_gy"]
        lad_v15 = feats.get("LAD__V15Gy_cc", 0.0)
        cor_injury = (
            0.40 * lad_mean
            + 0.030 * lad_v15
            + 1.8 * hypertension
            + 1.6 * diabetes
            + 1.7 * smoker
            + 0.9 * anthracycline
            + 0.12 * max(0.0, age - 55)
        )
        cor_z = cor_injury / 10.0

        # ---- Myocardial early markers ----------------------------------------
        # GLS (more negative = better); injury makes it less negative.
        baseline_gls = float(np.clip(rng.normal(-20.5, 1.4), -24, -16))
        followup_gls = float(baseline_gls + 1.6 * myo_z + rng.normal(0, 0.6))
        gls_rel_change_pct = float((followup_gls - baseline_gls) / abs(baseline_gls) * 100.0)

        # LV CFD endocardial wall shear stress (Pa); falls with injury.
        baseline_lv_wss = float(np.clip(rng.normal(1.25, 0.18), 0.7, 1.9))
        followup_lv_wss = float(np.clip(baseline_lv_wss - 0.13 * myo_z + rng.normal(0, 0.05), 0.3, 2.2))
        lv_wss_change = float(followup_lv_wss - baseline_lv_wss)

        # LV CFD intraventricular energy loss (mW); rises with injury.
        baseline_lv_eloss = float(np.clip(rng.normal(0.45, 0.08), 0.2, 0.8))
        followup_lv_eloss = float(np.clip(baseline_lv_eloss + 0.06 * myo_z + rng.normal(0, 0.025), 0.1, 1.2))
        lv_energy_loss_change = float(followup_lv_eloss - baseline_lv_eloss)

        # ---- Coronary early markers (LAD CFD) --------------------------------
        # Coronary wall shear stress (Pa); endothelial injury lowers it.
        baseline_lad_wss = float(np.clip(rng.normal(1.6, 0.25), 0.9, 2.4))
        followup_lad_wss = float(np.clip(baseline_lad_wss - 0.18 * cor_z + rng.normal(0, 0.07), 0.4, 2.8))
        lad_wss_change = float(followup_lad_wss - baseline_lad_wss)

        # Oscillatory shear index (0..0.5); rises with atherogenic remodeling.
        baseline_lad_osi = float(np.clip(rng.normal(0.12, 0.03), 0.02, 0.30))
        followup_lad_osi = float(np.clip(baseline_lad_osi + 0.020 * cor_z + rng.normal(0, 0.010), 0.0, 0.5))
        lad_osi_change = float(followup_lad_osi - baseline_lad_osi)

        # ---- Late endpoint with a genuine mediation structure ----------------
        # The overt LVEF drop arises partly *through* the early subclinical
        # markers (strain/CFD changes precede the EF fall) and partly via a
        # residual direct injury path. This builds a real dose -> marker -> LVEF
        # chain, so the mediation analysis recovers a non-trivial mediated share.
        lvef_decline = float(np.clip(
            0.28 * myo_injury                 # direct path (injury not via markers)
            + 0.70 * gls_rel_change_pct       # indirect via strain
            + 11.0 * (-lv_wss_change)         # indirect via LV CFD wall shear stress
            + rng.normal(0, 2.6),
            0, 40,
        ))
        cardiotoxicity = int(lvef_decline >= 10.0)

        rec = {"patient_id": pid, "laterality": laterality, **feats}
        rec.update(
            age=age,
            anthracycline=anthracycline,
            trastuzumab=trastuzumab,
            hypertension=hypertension,
            diabetes=diabetes,
            smoker=smoker,
            baseline_lvef=baseline_lvef,
            # --- myocardial markers: baselines (predictors) + targets ---
            baseline_gls=baseline_gls,
            followup_gls=followup_gls,
            gls_rel_change_pct=gls_rel_change_pct,
            baseline_lv_wss=baseline_lv_wss,
            followup_lv_wss=followup_lv_wss,
            lv_wss_change=lv_wss_change,
            baseline_lv_eloss=baseline_lv_eloss,
            followup_lv_eloss=followup_lv_eloss,
            lv_energy_loss_change=lv_energy_loss_change,
            # --- coronary markers: baselines (predictors) + targets ---
            baseline_lad_wss=baseline_lad_wss,
            followup_lad_wss=followup_lad_wss,
            lad_wss_change=lad_wss_change,
            baseline_lad_osi=baseline_lad_osi,
            followup_lad_osi=followup_lad_osi,
            lad_osi_change=lad_osi_change,
            # --- late endpoint, reference comparison only ---
            lvef_decline=lvef_decline,
            cardiotoxicity=cardiotoxicity,
        )
        records.append(rec)

    return pd.DataFrame.from_records(records).set_index("patient_id")


# Target columns grouped by physiological domain (consumed by train_pls.py).
MYOCARDIAL_TARGETS = ["gls_rel_change_pct", "lv_wss_change", "lv_energy_loss_change"]
CORONARY_TARGETS = ["lad_wss_change", "lad_osi_change"]
ALL_TARGETS = MYOCARDIAL_TARGETS + CORONARY_TARGETS


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic RT-cardiotoxicity cohort.")
    ap.add_argument("--n", type=int, default=300, help="Number of patients.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "cohort.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate_cohort(args.n, args.seed)
    df.to_csv(args.out)
    print(f"Wrote {len(df)} patients to {args.out}")
    print(f"Substructures: {', '.join(SUBSTRUCTURES)}  (per-structure DVH features)")
    print("Early-marker targets (mean ± sd):")
    for t in ALL_TARGETS:
        print(f"  {t:22s} {df[t].mean():8.4f} ± {df[t].std():.4f}")
    print(f"Late reference endpoint (CTRCD) rate: {df['cardiotoxicity'].mean():.1%}")
    print(f"Total columns: {df.shape[1]}")


if __name__ == "__main__":
    main()
