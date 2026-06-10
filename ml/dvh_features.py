"""
Dose-Volume Histogram (DVH) feature extraction for RT cardiotoxicity modeling.

The MATLAB pipeline (RTdoseoverlay.m) maps an RT dose distribution onto a heart
mesh and produces a per-vertex (or per-voxel) dose array in Gy. This module turns
that raw dose array into the dosimetric features that the cardio-oncology
literature has linked to radiation-induced cardiac toxicity, e.g. mean heart
dose (MHD), Vx (volume receiving >= x Gy) and Dx (minimum dose to the hottest
x% of the structure).

The features are deliberately model-agnostic: feed them either a per-voxel dose
volume (preferred, since each voxel carries equal volume) or a per-vertex array.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Dose thresholds (Gy) used for Vx metrics. These cover the low-dose bath
# (V5) through the high-dose region relevant for breast/thoracic RT.
DEFAULT_V_THRESHOLDS_GY = (5, 10, 15, 20, 25, 30, 35, 40)

# Volume percentiles (%) used for Dx metrics (Dx = min dose to hottest x%).
DEFAULT_D_PERCENTILES = (2, 5, 10, 50, 95, 98)


def extract_dvh_features(
    dose_gy: np.ndarray,
    voxel_volume_cc: float | None = None,
    v_thresholds_gy: tuple = DEFAULT_V_THRESHOLDS_GY,
    d_percentiles: tuple = DEFAULT_D_PERCENTILES,
) -> dict[str, float]:
    """Compute DVH summary features for a single structure (the heart).

    Parameters
    ----------
    dose_gy
        1-D array of dose values in Gy. Each element should represent an equal
        unit of volume (one voxel). If you pass per-vertex mesh dose, the Vx
        metrics approximate surface area fractions rather than true volume.
    voxel_volume_cc
        Volume of a single voxel in cubic centimetres. If provided, Vx metrics
        are reported as absolute volumes (cc); otherwise as percent of structure.
    v_thresholds_gy
        Dose thresholds for Vx (volume receiving at least x Gy).
    d_percentiles
        Percentiles for Dx (dose covering the hottest x% of volume).

    Returns
    -------
    dict
        Feature name -> value. Vx are percent (or cc), Dx and summary stats Gy.
    """
    dose = np.asarray(dose_gy, dtype=float).ravel()
    dose = dose[np.isfinite(dose)]
    if dose.size == 0:
        raise ValueError("dose_gy is empty after removing non-finite values.")

    n = dose.size
    feats: dict[str, float] = {}

    # Basic summary statistics.
    feats["mean_heart_dose_gy"] = float(np.mean(dose))
    feats["max_heart_dose_gy"] = float(np.max(dose))
    feats["min_heart_dose_gy"] = float(np.min(dose))
    feats["median_heart_dose_gy"] = float(np.median(dose))
    feats["std_heart_dose_gy"] = float(np.std(dose))
    # Integral dose (Gy*cc) if voxel volume known, else Gy*voxels.
    total_volume = n * voxel_volume_cc if voxel_volume_cc else float(n)
    feats["integral_dose"] = float(np.sum(dose) * (voxel_volume_cc or 1.0))

    # Vx: fraction of volume receiving at least x Gy.
    for x in v_thresholds_gy:
        frac = float(np.mean(dose >= x))  # 0..1
        if voxel_volume_cc:
            feats[f"V{x}Gy_cc"] = frac * total_volume
        else:
            feats[f"V{x}Gy_pct"] = frac * 100.0

    # Dx: min dose to the hottest x% of volume (Dx% = (100-x) percentile).
    for x in d_percentiles:
        feats[f"D{x}pct_gy"] = float(np.percentile(dose, 100 - x))

    return feats


def build_feature_table(
    per_patient_dose: dict[str, np.ndarray],
    voxel_volume_cc: float | None = None,
    clinical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble a tidy feature table from many patients' dose arrays.

    Parameters
    ----------
    per_patient_dose
        Mapping of patient_id -> 1-D dose array (Gy).
    voxel_volume_cc
        Passed through to :func:`extract_dvh_features`.
    clinical
        Optional clinical feature table indexed by patient_id (age, anthracycline
        exposure, baseline LVEF, etc.). Joined onto the DVH features.

    Returns
    -------
    pandas.DataFrame indexed by patient_id.
    """
    rows = {}
    for pid, dose in per_patient_dose.items():
        rows[pid] = extract_dvh_features(dose, voxel_volume_cc=voxel_volume_cc)
    dvh = pd.DataFrame.from_dict(rows, orient="index")
    dvh.index.name = "patient_id"

    if clinical is not None:
        dvh = dvh.join(clinical, how="left")
    return dvh


if __name__ == "__main__":
    # Tiny demonstration on a synthetic dose array.
    rng = np.random.default_rng(0)
    demo_dose = np.clip(rng.normal(8, 6, size=20000), 0, 50)
    feats = extract_dvh_features(demo_dose)
    for k, v in feats.items():
        print(f"{k:24s} {v:8.3f}")
