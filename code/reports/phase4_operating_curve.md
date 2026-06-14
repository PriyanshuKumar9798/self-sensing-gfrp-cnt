# Phase 4: Calibration & alarm operating curve

Model: LightGBM quantile heads (q=0.1/0.5/0.9, pinball loss) for DTF, LOSO. Alarm signal: the **conformally-calibrated** q=0.9 upper bound. Bootstrap ensemble (10 L1 models) as a second uncertainty estimate.

> **Calibration finding.** The raw quantile heads collapse under LOSO: DTF is nearly deterministic within a specimen, so the dominant uncertainty is the cross-specimen D_peak, which pooled quantile regression cannot express (raw q90 covers only ~0.6 vs nominal 0.9). A cross-conformal correction (§4.1.5) inflates the upper bound to restore coverage.

## Calibration (empirical coverage vs nominal)

| estimator | nominal | empirical P(true ≤ pred) |
|---|---|---|
| raw q10 | 0.10 | 0.469 |
| raw q50 | 0.50 | 0.572 |
| raw q90 | 0.90 | 0.606 |
| **conformal q90** | 0.90 | **0.876** |
| bootstrap upper | 0.90 | 0.537 |

Quantiles sorted per-sample to prevent crossing.


## Alarm operating curve (signal = q90)

| tau | N | precision | recall | f1 | median_lead_s | false_alarms_per_test | tp | fp | fn |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | nan | 0.00 | 0.00 | nan | 0.00 | 0 | 0 | 4 |
| 1 | 6 | nan | 0.00 | 0.00 | nan | 0.00 | 0 | 0 | 4 |
| 1 | 12 | nan | 0.00 | 0.00 | nan | 0.00 | 0 | 0 | 4 |
| 1.5 | 3 | nan | 0.00 | 0.00 | nan | 0.00 | 0 | 0 | 4 |
| 1.5 | 6 | nan | 0.00 | 0.00 | nan | 0.00 | 0 | 0 | 4 |
| 1.5 | 12 | nan | 0.00 | 0.00 | nan | 0.00 | 0 | 0 | 4 |
| 2 | 3 | 1.00 | 0.25 | 0.40 | 78.67 | 0.00 | 1 | 0 | 3 |
| 2 | 6 | 1.00 | 0.25 | 0.40 | 78.42 | 0.00 | 1 | 0 | 3 |
| 2 | 12 | 1.00 | 0.25 | 0.40 | 77.92 | 0.00 | 1 | 0 | 3 |
| 2.5 | 3 | 1.00 | 0.25 | 0.40 | 84.00 | 0.00 | 1 | 0 | 3 |
| 2.5 | 6 | 1.00 | 0.25 | 0.40 | 83.75 | 0.00 | 1 | 0 | 3 |
| 2.5 | 12 | 1.00 | 0.25 | 0.40 | 83.25 | 0.00 | 1 | 0 | 3 |
| 3 | 3 | 1.00 | 0.25 | 0.40 | 122.50 | 0.00 | 1 | 0 | 3 |
| 3 | 6 | 1.00 | 0.25 | 0.40 | 122.25 | 0.00 | 1 | 0 | 3 |
| 3 | 12 | 1.00 | 0.25 | 0.40 | 121.75 | 0.00 | 1 | 0 | 3 |
| 4 | 3 | 1.00 | 0.50 | 0.67 | 137.75 | 0.00 | 2 | 0 | 2 |
| 4 | 6 | 1.00 | 0.50 | 0.67 | 137.50 | 0.00 | 2 | 0 | 2 |
| 4 | 12 | 1.00 | 0.50 | 0.67 | 137.00 | 0.00 | 2 | 0 | 2 |
| 5 | 3 | 1.00 | 1.00 | 1.00 | 73.92 | 0.00 | 4 | 0 | 0 |
| 5 | 6 | 1.00 | 1.00 | 1.00 | 73.67 | 0.00 | 4 | 0 | 0 |
| 5 | 12 | 1.00 | 1.00 | 1.00 | 73.17 | 0.00 | 4 | 0 | 0 |

## Chosen operating point (max median lead s.t. recall ≥ 0.75, precision ≥ 0.8)

- τ_alarm = **5 mm**, N = **3** samples
- precision = 1.00, recall = 1.00, F1 = 1.00
- median lead time = **73.9 s** (~1.85 mm of deflection at the median rate)
- false alarms per test = 0.00


## Per-specimen alarm timing at the chosen point

| specimen | alarm t [s] | failure t [s] | lead [s] | outcome |
|---|---|---|---|---|
| S1 | 383 | 440 | 56 | TP |
| S2 | 272 | 551 | 279 | TP |
| S3 | 261 | 353 | 92 | TP |
| S4 | 209 | 260 | 51 | TP |

## Figures

- `figures/phase4/reliability.png`
- `figures/phase4/operating_curve.png`
- `figures/phase4/per_specimen_alarm.png`
