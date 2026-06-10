# RT Cardiotoxicity — Machine Learning Pipeline

Predict **early (subclinical) cardiotoxicity markers** from **substructure-resolved
heart dose** plus clinical covariates, for breast/thoracic radiation therapy.

The markers change *before* LVEF declines and span two physiological domains:

| Domain | Target | Source | Driven by |
|--------|--------|--------|-----------|
| Myocardial | `gls_rel_change_pct` | Echo speckle-tracking | LV dose + chemo |
| Myocardial | `lv_wss_change` | LV CFD | LV dose + chemo |
| Myocardial | `lv_energy_loss_change` | LV CFD | LV dose + chemo |
| Coronary | `lad_wss_change` | Coronary CFD | **LAD dose** + atherosclerotic risk |
| Coronary | `lad_osi_change` | Coronary CFD | **LAD dose** + atherosclerotic risk |

These complement the MATLAB code in the repo root, which builds the heart/LV mesh
from CT/RTSTRUCT and maps RT dose onto it (`RTdoseoverlay.m`, `LV.m`) — the source
of both the per-voxel dose arrays (DVH features) and the meshes (CFD indices).

## Two methodological choices that matter

**1. Dose is resolved per cardiac substructure, not whole-heart.**
Coronary injury is driven by the dose to the **LAD** (which sits under the tangent
fields in left-sided breast RT), and myocardial injury by **LV** dose — whole-heart
mean dose is too crude. DVH features are therefore computed separately for
`heart`, `LV`, `LAD`, `RCA`, `LCX`, with columns named `"{structure}__{feature}"`
(e.g. `LAD__mean_dose_gy`). In the demo, restricting the model to the coronary
markers makes the top VIP predictors all LAD-dose metrics — the intended signal.

**2. PLS regression is the primary model (not deep learning) at N≈300.**
The predictor block (DVH across 5 structures) is wide and collinear, and the
response block (5 correlated markers) is multivariate. Partial Least Squares finds
a few latent components that jointly explain dose↔marker covariance — stable at a
few hundred patients, where ordinary multivariate regression or a neural net would
overfit. PLS also gives **VIP scores** (>1 = influential predictor) for
interpretation. Because the cohort has **baseline + follow-up**, each marker's
baseline value is a predictor and the *change* is the target; follow-up values and
the late LVEF endpoint are excluded to avoid leakage.

## What's here

| File | Purpose |
|------|---------|
| `dvh_features.py` | Dose array (Gy) → DVH features; `extract_substructure_features` prefixes them per structure. |
| `make_dataset.py` | Synthetic cohort (N=300). Two latent injuries — myocardial (LV-dose driven) and coronary (LAD-dose driven) — feed their respective markers; late LVEF decline is downstream. |
| `train_pls.py` | **Primary.** PLS regression: substructure dose → multi-domain markers. CV-selected components, per-target R²/MAE/RMSE, VIP scores, predicted-vs-actual plots. |
| `train_regression.py` | Alternative. Penalized regression (Ridge/Lasso/ElasticNet/multi-task ENet/RF) on the same targets. |
| `train.py` | Secondary. Binary classifier for the *late* CTRCD endpoint (weaker at this N, by design). |
| `predict.py` | Score new patients (auto-detects the saved model) from a feature CSV or a raw dose dict. |
| `requirements.txt` | Dependencies (PLS ships with scikit-learn — no extra deps). |

## Quick start

```bash
pip install -r requirements.txt

python make_dataset.py                  # data/cohort.csv (synthetic, N=300)
python train_pls.py                     # PLS, all 5 markers -> outputs/
python predict.py --csv data/cohort.csv --out outputs/predictions.csv

# domain-specific runs
python train_pls.py --targets lad_wss_change lad_osi_change      # coronary only
python train_pls.py --targets gls_rel_change_pct lv_wss_change   # myocardial only
python train_pls.py --max-components 8
```

PLS artifacts in `outputs/`: `model_pls.joblib`, `metrics_pls.json`,
`pls_pred_vs_actual.png`, `vip_scores.csv`.

## Predictors

**Dosimetric, per substructure** (`heart`, `LV`, `LAD`, `RCA`, `LCX`): mean / max /
median / std dose, integral dose, V5–V40 Gy, D2/D5/D10/D50/D95/D98 %.

**Clinical & baseline:** age, laterality, anthracycline / trastuzumab, hypertension,
diabetes, smoking, baseline LVEF, and each marker's baseline value.

## Using your own data

Provide a CSV with the same column scheme — per-substructure DVH (`{structure}__…`),
clinical covariates, per-marker baselines, and the change targets — then:

```bash
python train_pls.py --data your_cohort.csv \
    --targets gls_rel_change_pct lv_wss_change lv_energy_loss_change \
              lad_wss_change lad_osi_change
```

To build the DVH block from real plans, export each substructure's `dose_on_mesh`
from MATLAB and use `dvh_features.extract_substructure_features({'LAD': dose_lad,
'LV': dose_lv, ...})`. The CFD markers come from your LV / coronary CFD
post-processing (the repo already builds the LV mesh and a Fluent valve workflow).

> ⚠️ The shipped dataset is **synthetic** and for pipeline demonstration only.
> Reported metrics reflect the simulated data-generating process, not clinical
> validity. Validate on real, IRB-approved data before drawing conclusions.

## Roadmap

Natural follow-ups discussed for this cohort design:
- **Mediation analysis** — formally test that early markers mediate the dose→LVEF
  effect (the next analysis after this substructure+PLS step).
- **Substructure NTCP curves** — LAD/LV dose-response for clinical reporting.
- **Functional DVH (fPCA)** — use whole DVH curves instead of fixed Vx thresholds.
