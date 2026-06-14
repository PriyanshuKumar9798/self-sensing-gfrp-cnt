# Phase 3: Sequence-model scorecard

Seed 42. CPU. Inner-LOSO ensemble (3 models/fold). Multi-task (DTF Huber + load/defl MSE + stage CE). Validation: leave-one-specimen-out.

**Selected model: `lstm_aug_w2`** (20935 params), chosen as the lowest LOSO DTF MAE across the ablation.

**Headline:** selected mean DTF MAE = 1.489 mm vs Phase 2 LightGBM 1.506 mm (**+1.1%**; it beats the baseline, target was ≥15% reduction). Alarm @ τ=2.0,N=6: F1=0.67, recall=0.67, precision=0.67, median lead=69.2 s.

> **Finding.** No sequence model reaches the +15% target. The TCN underperforms LightGBM and augmentation does not help it here; only the LSTM with augmentation marginally edges the baseline. With n=4 and an 8× baseline spread, the gradient-boosted trees of Phase 2 remain very competitive (as anticipated in §3 D6).


## Ablation: mean DTF MAE and alarm F1

| config | S1 | S2 | S3 | S4 | mean | vs P2 | alarm F1 |
|---|---|---|---|---|---|---|---|
| tcn_aug_w2 | 2.43 | 1.19 | 2.43 | 1.39 | 1.857 | -23.3% | 0.67 |
| tcn_noaug_w2 | 2.34 | 1.15 | 2.31 | 1.26 | 1.767 | -17.3% | 0.67 |
| lstm_aug_w2 | 0.92 | 1.01 | 2.46 | 1.57 | 1.489 | +1.1% | 0.67 |
| tcn_aug_w1 | 2.28 | 1.15 | 2.37 | 1.42 | 1.807 | -20.0% | 0.67 |
| tcn_aug_w5 | 2.32 | 1.14 | 2.56 | 1.45 | 1.868 | -24.1% | 0.40 |
| _Phase 2 LightGBM_ | 1.27 | 1.05 | 2.10 | 1.60 | 1.506 | - | - |

## Selected model (lstm_aug_w2): full per-fold metrics

| metric | S1 | S2 | S3 | S4 | mean |
|---|---|---|---|---|---|
| DTF_MAE | 0.918 | 1.005 | 2.458 | 1.574 | 1.489 |
| DTF_MedAE | 1.049 | 1.136 | 2.103 | 1.787 | 1.518 |
| TTF_MAE_s | 54.554 | 59.058 | 73.657 | 47.534 | 58.701 |
| load_RMSE | 0.045 | 0.008 | 0.017 | 0.072 | 0.035 |
| load_RMSE_I | 0.009 | 0.011 | 0.017 | 0.080 | 0.029 |
| load_RMSE_III | 0.067 | 0.009 | 0.025 | 0.064 | 0.041 |
| defl_RMSE | 1.659 | 1.282 | 2.419 | 1.636 | 1.749 |
| stage_bal_acc | 0.485 | 0.464 | 0.391 | 0.480 | 0.455 |

## Selected model stage confusion (rows=true, cols=pred)

| true\pred | early | elastic-rising | pre-failure | post-failure |
|---|---|---|---|---|
| early | 0 | 2207 | 0 | 0 |
| elastic-rising | 0 | 10942 | 2096 | 215 |
| pre-failure | 0 | 0 | 2748 | 1042 |
| post-failure | 0 | 0 | 3907 | 398 |

## Figures

- `figures/phase3/dtf_pred_vs_true.png`
- `figures/phase3/ablation_dtf_mae.png`
