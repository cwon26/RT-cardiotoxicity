"""
Synthetic RT-cardiotoxicity cohort generator.

Real patient data (CT, RTSTRUCT, RTDOSE, echo) cannot live in this repo, so this
script fabricates a plausible cohort that lets the modeling pipeline run
end-to-end and serves as a schema reference. The generative model encodes
clinically motivated relationships:

  * Higher mean heart dose and high-dose volumes raise cardiotoxicity risk.
  * Anthracycline chemotherapy and trastuzumab compound radiation risk.
  * Older age, hypertension, diabetes, smoking and low baseline LVEF add risk.

The outcome is a binary "cardiotoxicity" label (e.g. a >=10% absolute drop in
LVEF to below 50%, or a significant decline in echo global longitudinal strain
within follow-up), plus the continuous LVEF decline it is derived from.

Output: ml/data/cohort.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from dvh_features import extract_dvh_features

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def _simulate_heart_dose(rng: np.random.Generator, laterality: str) -> np.ndarray:
    """Fabricate a per-voxel heart dose array (Gy) for one patient.

    Left-sided breast RT deposits more dose in the heart than right-sided. We
    model the heart dose as a mixture: most voxels get a low-dose bath, a tail
    gets higher dose where the heart abuts the chest wall / tangent fields.
    """
    n_vox = rng.integers(15000, 25000)
    if laterality == "left":
        bath_mean, bath_sd = rng.uniform(2.5, 6.0), 2.0
        hot_frac = rng.uniform(0.04, 0.18)
        hot_mean = rng.uniform(20, 42)
    else:  # right
        bath_mean, bath_sd = rng.uniform(0.5, 2.5), 1.0
        hot_frac = rng.uniform(0.0, 0.04)
        hot_mean = rng.uniform(8, 20)

    bath = rng.normal(bath_mean, bath_sd, size=n_vox)
    n_hot = int(hot_frac * n_vox)
    if n_hot > 0:
        hot = rng.normal(hot_mean, 5.0, size=n_hot)
        bath[:n_hot] = hot
    dose = np.clip(bath, 0, 50)
    rng.shuffle(dose)
    return dose


def generate_cohort(n_patients: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for i in range(n_patients):
        pid = f"PT{i:04d}"
        laterality = rng.choice(["left", "right"], p=[0.55, 0.45])
        dose = _simulate_heart_dose(rng, laterality)
        feats = extract_dvh_features(dose, voxel_volume_cc=0.002)

        # Clinical covariates.
        age = float(np.clip(rng.normal(58, 11), 28, 88))
        anthracycline = int(rng.random() < 0.45)
        trastuzumab = int(rng.random() < 0.30)
        hypertension = int(rng.random() < 0.35)
        diabetes = int(rng.random() < 0.18)
        smoker = int(rng.random() < 0.22)
        baseline_lvef = float(np.clip(rng.normal(62, 4.5), 50, 72))

        # Latent radiation cardiac injury — the common upstream driver. Early
        # subclinical markers (GLS, CFD indices) respond first; the LVEF decline
        # is a later, downstream consequence of the same injury.
        mhd = feats["mean_heart_dose_gy"]
        v25 = feats.get("V25Gy_cc", 0.0)
        injury = (
            0.42 * mhd
            + 0.015 * v25
            + 3.2 * anthracycline
            + 2.4 * trastuzumab
            + 1.3 * hypertension
            + 1.1 * diabetes
            + 0.9 * smoker
            + 0.10 * max(0.0, age - 55)
            + 0.18 * max(0.0, 60 - baseline_lvef)
        )
        injury_z = injury / 10.0  # convenient unit scaling

        # --- Early marker 1: GLS (speckle-tracking strain; more negative = better).
        # Injury makes GLS less negative (worsens). Predict the relative change,
        # the endpoint used in the ESC cardio-oncology guideline (>15% = abnormal).
        baseline_gls = float(np.clip(rng.normal(-20.5, 1.4), -24, -16))
        followup_gls = float(baseline_gls + 1.6 * injury_z + rng.normal(0, 0.6))
        gls_rel_change_pct = float((followup_gls - baseline_gls) / abs(baseline_gls) * 100.0)

        # --- Early marker 2: CFD endocardial wall shear stress (Pa); falls with injury.
        baseline_wss = float(np.clip(rng.normal(1.25, 0.18), 0.7, 1.9))
        followup_wss = float(np.clip(baseline_wss - 0.13 * injury_z + rng.normal(0, 0.05), 0.3, 2.2))
        wss_change_pa = float(followup_wss - baseline_wss)

        # --- Early marker 3: CFD intraventricular energy loss (mW); rises with injury.
        baseline_eloss = float(np.clip(rng.normal(0.45, 0.08), 0.2, 0.8))
        followup_eloss = float(np.clip(baseline_eloss + 0.06 * injury_z + rng.normal(0, 0.025), 0.1, 1.2))
        energy_loss_change = float(followup_eloss - baseline_eloss)

        # --- Late endpoint (downstream of early injury), kept for reference only.
        lvef_decline = float(np.clip(0.75 * injury + rng.normal(0, 3.5), 0, 40))
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
            # --- early subclinical markers: baselines (predictors) + targets ---
            baseline_gls=baseline_gls,
            followup_gls=followup_gls,
            gls_rel_change_pct=gls_rel_change_pct,
            baseline_wss=baseline_wss,
            followup_wss=followup_wss,
            wss_change_pa=wss_change_pa,
            baseline_eloss=baseline_eloss,
            followup_eloss=followup_eloss,
            energy_loss_change=energy_loss_change,
            # --- late endpoint, reference comparison only ---
            lvef_decline=lvef_decline,
            cardiotoxicity=cardiotoxicity,
        )
        records.append(rec)

    df = pd.DataFrame.from_records(records).set_index("patient_id")
    return df


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
    print("Early-marker targets (mean ± sd):")
    for t in ("gls_rel_change_pct", "wss_change_pa", "energy_loss_change"):
        print(f"  {t:20s} {df[t].mean():7.3f} ± {df[t].std():.3f}")
    print(f"Late reference endpoint (CTRCD) rate: {df['cardiotoxicity'].mean():.1%}")
    print(f"Columns ({df.shape[1]}): {list(df.columns)}")


if __name__ == "__main__":
    main()
