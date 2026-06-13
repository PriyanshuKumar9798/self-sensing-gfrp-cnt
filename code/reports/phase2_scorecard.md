# Phase 2: Classical LOSO baseline scorecard

Seed 42. Features: 14 causal. Models: Linear (floor, 8 features) and LightGBM (inner-LOSO tuned). Validation: leave-one-specimen-out (4 folds).

Initial alarm operating point: τ_alarm=2.0 mm, N=6 consecutive samples.


## LINEAR: per-fold metrics

| metric | S1 | S2 | S3 | S4 | mean |
|---|---|---|---|---|---|
| DTF_MAE | 0.883 | 0.848 | 2.464 | 2.865 | 1.765 |
| DTF_MedAE | 0.963 | 0.935 | 2.371 | 2.671 | 1.735 |
| TTF_MAE_s | 52.422 | 49.764 | 74.590 | 86.026 | 65.700 |
| load_RMSE | 0.044 | 0.043 | 0.116 | 0.127 | 0.082 |
| load_RMSE_I | 0.028 | 0.048 | 0.039 | 0.130 | 0.061 |
| load_RMSE_III | 0.064 | 0.039 | 0.157 | 0.132 | 0.098 |
| defl_RMSE | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stage_bal_acc | 0.489 | 0.870 | 0.733 | 0.879 | 0.743 |


**Alarm @ default op point:** precision=0.33, recall=0.50, F1=0.40, median lead=60.6 s (TP=1, FP=2, FN=1).


## LGBM: per-fold metrics

| metric | S1 | S2 | S3 | S4 | mean |
|---|---|---|---|---|---|
| DTF_MAE | 1.269 | 1.049 | 2.101 | 1.605 | 1.506 |
| DTF_MedAE | 1.267 | 1.192 | 2.212 | 1.699 | 1.593 |
| TTF_MAE_s | 75.202 | 61.633 | 62.865 | 48.641 | 62.085 |
| load_RMSE | 0.028 | 0.033 | 0.084 | 0.055 | 0.050 |
| load_RMSE_I | 0.007 | 0.005 | 0.047 | 0.046 | 0.026 |
| load_RMSE_III | 0.044 | 0.042 | 0.104 | 0.053 | 0.061 |
| defl_RMSE | 0.312 | 0.325 | 0.554 | 0.167 | 0.340 |
| stage_bal_acc | 0.497 | 0.722 | 0.672 | 0.632 | 0.631 |


**Alarm @ default op point:** precision=0.50, recall=1.00, F1=0.67, median lead=53.0 s (TP=2, FP=2, FN=0).


## LightGBM stage confusion (rows=true, cols=pred)

| true\pred | early | elastic-rising | pre-failure | post-failure |
|---|---|---|---|---|
| early | 2001 | 206 | 0 | 0 |
| elastic-rising | 100 | 12761 | 160 | 232 |
| pre-failure | 0 | 1666 | 13 | 2111 |
| post-failure | 0 | 499 | 412 | 3394 |

## Alarm operating curve (LightGBM)

| tau | N | precision | recall | f1 | median_lead_s | tp | fp | fn |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 3 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 0.5 | 6 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 0.5 | 12 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 1 | 3 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 1 | 6 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 1 | 12 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 1.5 | 3 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 1.5 | 6 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 1.5 | 12 | 0.00 | 0.00 | 0.00 | nan | 0 | 2 | 2 |
| 2 | 3 | 0.50 | 1.00 | 0.67 | 53.29 | 2 | 2 | 0 |
| 2 | 6 | 0.50 | 1.00 | 0.67 | 53.04 | 2 | 2 | 0 |
| 2 | 12 | 0.50 | 1.00 | 0.67 | 52.54 | 2 | 2 | 0 |
| 3 | 3 | 0.50 | 1.00 | 0.67 | 73.54 | 2 | 2 | 0 |
| 3 | 6 | 0.50 | 1.00 | 0.67 | 73.29 | 2 | 2 | 0 |
| 3 | 12 | 0.50 | 1.00 | 0.67 | 72.79 | 2 | 2 | 0 |
| 4 | 3 | 0.50 | 1.00 | 0.67 | 107.79 | 2 | 2 | 0 |
| 4 | 6 | 0.50 | 1.00 | 0.67 | 102.17 | 2 | 2 | 0 |
| 4 | 12 | 0.50 | 1.00 | 0.67 | 101.67 | 2 | 2 | 0 |
| 5 | 3 | 0.50 | 1.00 | 0.67 | 172.21 | 2 | 2 | 0 |
| 5 | 6 | 0.50 | 1.00 | 0.67 | 171.96 | 2 | 2 | 0 |
| 5 | 12 | 0.50 | 1.00 | 0.67 | 171.46 | 2 | 2 | 0 |

## Figures

- `figures/phase2/dtf_pred_vs_true.png`
- `figures/phase2/alarm_operating_curve.png`
