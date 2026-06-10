# RT Cardiotoxicity — Machine Learning Pipeline

A reproducible pipeline that predicts **radiation-induced cardiotoxicity**
(cancer-therapy-related cardiac dysfunction, CTRCD) from heart **dose-volume
histogram (DVH) features** plus clinical covariates.

This complements the MATLAB code in the repository root, which extracts a heart
mesh from CT/RTSTRUCT and maps the RT dose distribution onto it
(`RTdoseoverlay.m`). That step produces a per-voxel/per-vertex heart dose array
in Gy — exactly the input this pipeline turns into predictive features.

## What's here

| File | Purpose |
|------|---------|
| `dvh_features.py` | Turn a raw heart-dose array (Gy) into DVH features: mean heart dose, Vx (volume ≥ x Gy), Dx (dose to hottest x%), and summary statistics. |
| `make_dataset.py` | Generate a **synthetic** cohort (DVH features + clinical covariates + CTRCD label) so the pipeline runs without protected patient data. Encodes clinically motivated risk relationships. |
| `train.py` | Cross-validate logistic regression / random forest / gradient boosting, auto-select the best, evaluate on a held-out test set, and save model + metrics + ROC plot + feature importances. |
| `predict.py` | Score new patients from a feature CSV, or directly from a raw heart-dose array + clinical dict. |
| `requirements.txt` | Python dependencies. |

## Quick start

```bash
pip install -r requirements.txt

python make_dataset.py        # writes data/cohort.csv (synthetic)
python train.py               # trains, writes outputs/
python predict.py --csv data/cohort.csv --out outputs/predictions.csv
```

Outputs land in `outputs/`: `model.joblib`, `metrics.json`, `roc_curve.png`,
`feature_importance.csv`.

## Features

**Dosimetric (from the heart dose array):**
mean / max / median / std heart dose, integral dose,
V5–V40 Gy (volume receiving ≥ threshold), and
D2/D5/D10/D50/D95/D98 % (dose to the hottest x% of the heart).

**Clinical:**
age, laterality (left vs right breast), anthracycline and trastuzumab exposure,
hypertension, diabetes, smoking, baseline LVEF.

**Label:** `cardiotoxicity` — absolute LVEF drop ≥ 10 percentage points
(continuous `lvef_decline` is also provided for regression experiments).

## Using your own data

Replace the synthetic cohort with a real one. Either:

1. Provide a CSV with the same feature columns and a `cardiotoxicity` column,
   then run `train.py --data your.csv`; or
2. Compute features from real dose arrays with
   `dvh_features.build_feature_table(per_patient_dose, clinical=...)`, where
   `per_patient_dose` maps patient id → the Gy array exported from the MATLAB
   `dose_on_mesh` step (`writematrix(dose_on_mesh, 'PTxxxx_dose.csv')`).

> ⚠️ The shipped dataset is **synthetic** and for pipeline demonstration only.
> Reported metrics reflect the simulated data-generating process, not clinical
> validity. Validate on real, IRB-approved data before drawing conclusions.
