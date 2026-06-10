# RT Cardiotoxicity — Machine Learning Pipeline

Predict **early (subclinical) cardiotoxicity markers** from heart **dose-volume
histogram (DVH) features** plus clinical covariates, for breast/thoracic
radiation therapy.

The primary task is **multi-target regression** of markers that change *before*
LVEF declines:

| Target | Source | Meaning |
|--------|--------|---------|
| `gls_rel_change_pct` | Echo speckle-tracking | Relative change in global longitudinal strain (%). ESC cardio-oncology endpoint; >15% = abnormal. |
| `wss_change_pa` | CFD on the heart/LV mesh | Change in endocardial wall shear stress (Pa). |
| `energy_loss_change` | CFD on the heart/LV mesh | Change in intraventricular energy loss (mW). |

These complement the MATLAB code in the repo root, which extracts the heart/LV
mesh from CT/RTSTRUCT and maps the RT dose onto it (`RTdoseoverlay.m`, `LV.m`).
That step produces both the per-voxel **heart dose array (Gy)** that feeds the
DVH features and the **mesh** used for the CFD indices.

## Why regression (not classification or deep learning) at N≈300

This was a deliberate modeling decision for a ~300-patient cohort:

- **Continuous early markers use every patient's data.** A rare binary endpoint
  (clinical CTRCD, ~8% of patients) leaves only ~25 positives — too few to train
  stably. On this synthetic cohort the binary classifier reaches only
  CV ROC-AUC ≈ 0.68, while the regression targets are predicted stably.
- **Regularized linear models fit small, wide, collinear data well.** Ridge /
  Lasso / ElasticNet shrink the many correlated DVH predictors; a multi-task
  ElasticNet shares feature support across the related markers.
- **Deep learning would overfit.** With a few hundred patients and tens of
  features, nets have far more parameters than the data can constrain. Deep
  learning becomes worth considering only with thousands of patients or when
  feeding the **raw 3-D dose volume** to a CNN instead of DVH summaries.

Because the cohort has **baseline + follow-up** measurements, each marker's
*baseline* value is a predictor (baseline adjustment) and the *change* is the
target. Follow-up values and the late LVEF endpoint are excluded from predictors
to avoid leakage.

## What's here

| File | Purpose |
|------|---------|
| `dvh_features.py` | Heart dose array (Gy) → DVH features: mean heart dose, Vx, Dx, summary stats. |
| `make_dataset.py` | Synthetic cohort (N=300 default). A latent radiation-injury signal drives the early markers first and the late LVEF decline downstream. |
| `train_regression.py` | **Primary.** Multi-target regression; repeated CV over Ridge/Lasso/ElasticNet/multi-task ENet/RF; per-target R²/MAE/RMSE; predicted-vs-actual plots; per-target permutation importance. |
| `train.py` | Secondary. Binary classifier for the *late* CTRCD endpoint (kept for comparison; weaker at this N, by design). |
| `predict.py` | Score new patients (regression or classification, auto-detected) from a feature CSV or a raw dose array + clinical/baseline dict. |
| `requirements.txt` | Dependencies. |

## Quick start

```bash
pip install -r requirements.txt

python make_dataset.py                 # writes data/cohort.csv (synthetic, N=300)
python train_regression.py             # multi-target regression -> outputs/
python predict.py --csv data/cohort.csv --out outputs/predictions.csv

# options
python train_regression.py --model mtelastic            # force a model
python train_regression.py --targets gls_rel_change_pct # single target
python train.py                                         # late CTRCD classifier
```

Regression artifacts land in `outputs/`: `model_regression.joblib`,
`metrics_regression.json`, `pred_vs_actual.png`,
`feature_importance_regression.csv`.

## Predictors

**Dosimetric (from the heart dose array):** mean / max / median / std heart
dose, integral dose, V5–V40 Gy, D2/D5/D10/D50/D95/D98 %.

**Clinical & baseline:** age, laterality, anthracycline and trastuzumab
exposure, hypertension, diabetes, smoking, baseline LVEF, and each marker's
baseline value (`baseline_gls`, `baseline_wss`, `baseline_eloss`).

## Using your own data

Replace the synthetic cohort with a real one — same column names, with the
target change columns and per-marker baselines. Then:

```bash
python train_regression.py --data your_cohort.csv \
    --targets gls_rel_change_pct wss_change_pa energy_loss_change
```

To compute DVH features from real dose arrays, export `dose_on_mesh` from MATLAB
(`writematrix(dose_on_mesh, 'PTxxxx_dose.csv')`) and use
`dvh_features.build_feature_table(per_patient_dose, clinical=...)`. The CFD
indices come from your CFD post-processing of the LV mesh (the repo already
builds the mesh and a Fluent valve-segmentation workflow).

> ⚠️ The shipped dataset is **synthetic** and for pipeline demonstration only.
> Reported metrics reflect the simulated data-generating process, not clinical
> validity. Validate on real, IRB-approved data before drawing conclusions.
