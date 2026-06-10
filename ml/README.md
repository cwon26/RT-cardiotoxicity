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
| `mediation.py` | **Mediation analysis** — tests that early markers carry the LV-dose → LVEF-decline effect (Baron-Kenny + bootstrap CIs, confounder-adjusted). |
| `causal_mediation.py` | Potential-outcomes natural direct/indirect effects with an X·M interaction — robustness check on `mediation.py`. |
| `ntcp.py` | Substructure **NTCP dose-response curves** (logistic, D50 / γ50, bootstrap bands) — LAD→coronary, LV/heart→CTRCD. |
| `fpca_dvh.py` | **Functional DVH (fPCA)** — reduce whole DVH curves to a few shape modes and compare them to scalar Vx features as predictors. |
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

## Mediation analysis

`mediation.py` tests the core scientific claim — that the early markers lie on the
causal path from dose to overt dysfunction:

```
LV dose ──a──> early marker ──b──> LVEF decline      (indirect = a·b)
   └──────────────c'──────────────> LVEF decline      (direct)
```

It estimates, adjusting for chemo and clinical confounders, the total effect of LV
dose on LVEF decline and decomposes it into a **direct** path and an **indirect**
path through the markers, with percentile-bootstrap 95% CIs and a path diagram
(`outputs/mediation_path.png`). On the synthetic cohort it recovers a large,
significant mediated share (≈70%, CI excludes 0) — i.e. most of the dose effect on
LVEF acts *through* the early markers, which is the rationale for monitoring them.

```bash
python mediation.py
python mediation.py --exposure LV__mean_dose_gy --outcome lvef_decline \
    --mediators gls_rel_change_pct lv_wss_change --n-boot 2000
```

## Further analyses

```bash
python causal_mediation.py                 # potential-outcomes natural effects
python ntcp.py                             # substructure NTCP dose-response curves
python ntcp.py --pairs LV:cardiotoxicity LAD:coronary_event
python fpca_dvh.py --structure LAD --outcome lad_osi_change   # functional DVH
```

Outputs: `causal_mediation_results.json`; `ntcp_curves.png` / `ntcp_results.json`;
`fpca_dvh.png` / `fpca_results.json`.

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

All implemented:
- ✅ **Mediation analysis** (`mediation.py`) — ~70% of the LV-dose effect on LVEF
  is mediated by the early markers (CI excludes 0).
- ✅ **Causal mediation** (`causal_mediation.py`) — potential-outcomes natural
  effects corroborate it (~58% via GLS alone); X·M interaction is negligible, so
  the Baron-Kenny no-interaction assumption holds.
- ✅ **Substructure NTCP curves** (`ntcp.py`) — e.g. LAD mean-dose → coronary
  injury with D50 ≈ 14 Gy; clinical dose-response form with bootstrap bands.
- ✅ **Functional DVH / fPCA** (`fpca_dvh.py`) — ~97% of DVH-shape variance in PC1;
  4 fPCA scores match or beat the 8 scalar Vx features as predictors.

Possible further work: external validation on real cohorts, competing-risks /
time-to-event models for cardiac events, and joint multi-substructure NTCP.
