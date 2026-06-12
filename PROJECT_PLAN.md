# Design and Methodology Summary

**Project:** Self-Sensing GFRP / CNT Composites under Flexural Loading: Failure Forecasting from Electrical Resistance
**Course:** AS4545, Undergraduate Research Project
**Author:** Priyanshu Kumar (AE22B048)
**Department of Aerospace Engineering, Indian Institute of Technology Madras**

This document records the design of the project: the problem and the data, the key modelling decisions and their rationale, the methods used to handle a small dataset, the plan of work, and the metrics against which the models are judged.

---

## 1. Project at a glance

We have flexural test data for four nominally identical bars of a glass-fibre composite that has carbon nanotubes (CNTs) dispersed in the polymer matrix. Each bar was bent to failure in a three-point bending rig. The Universal Testing Machine (UTM) recorded load and cross-head deflection at about 12 Hz, and a Keithley source-meter recorded the electrical resistance of the embedded CNT network at about 12 Hz. The two recordings can be aligned in time.

The CNT network is electrically conductive while the glass fibre is not, so the resistance signal reflects almost entirely the state of the polymer matrix where damage initiates. The goal of this project is to use the resistance signal to **detect damage and give early warning of failure**, that is, to estimate how close the bar is to breaking while it is still being loaded, and to do so accurately enough to be useful.

The final deliverables are:

- a clean Python repository with the data pipeline, features, models, and evaluation code;
- a set of figures and tables documenting the performance under leave-one-specimen-out cross-validation;
- the final report presenting the modelling results.

---

## 2. Background and rationale

### 2.1 The material

A GFRP composite has glass fibres embedded in a polymer matrix. Glass does not conduct electricity, so a plain GFRP is an electrical insulator. By dispersing CNTs into the polymer, a conductive network is formed above a critical filler concentration (the percolation threshold). Conduction is partly through direct contact between CNTs and partly through electron tunnelling across small inter-tube gaps. Because tunnelling resistance grows exponentially with gap distance, the resistance of the composite is extremely sensitive to small strains and to the early stages of matrix damage. As loading proceeds, micro-cracks in the matrix progressively sever conducting paths and push the resistance up further.

### 2.2 The test

Each specimen is loaded in three-point bending: it rests on two supports a span L apart, and a central nose presses down at a constant cross-head speed. The standard relations apply (ASTM D790): outer-fibre stress σ = 3PL/(2bh²), outer-fibre strain ε = 6Dh/L², flexural modulus E_f = L³m/(4bh³). Test is displacement-controlled; load is recorded until the bar fails (load reaches a peak and then drops).

### 2.3 What the data shows

A first-pass EDA on the four specimens shows three distinct regimes in the fractional resistance change ΔR/R₀:

- **I, steep early rise:** ΔR/R₀ reaches roughly 80% of its pre-peak value in the first 10 to 20% of the test, consistent with the tunnelling network being most sensitive to the first strains and the earliest micro-damage.
- **II, saturation:** ΔR/R₀ then plateaus through the middle of the test while the load is still climbing. The network is in a relatively steady configuration in this regime.
- **III, features around failure:** distinct steps or jumps appear in ΔR/R₀ near the load peak, often continuing to rise even after the load peak as the crack opens further at increasing deflection.

The simple linear correlation between resistance and load varies across specimens from 0.16 to 0.89, confirming that no single fixed formula maps resistance to mechanical state. The baseline resistance varies from about 7 to 59 MΩ across the four bars, which is process scatter since the bars are nominally identical.

Loading rate also differs across the four tests (S2's test took roughly twice as long as S4's), so any feature or target that uses physical time directly will be confounded by rate.

---

## 3. Key design decisions

Each decision below is referenced in the implementation.

| ID | Decision | Rationale | Trigger |
|----|----------|-----------|---------|
| D1 | Frame the headline task as **regression**, not classification | Continuous targets give much richer gradients than discrete stage labels, which matters with only four specimens. A binary alarm can always be derived by thresholding a regression output. | EDA + small-data constraint |
| D2 | Primary target is **deflection-to-failure (DTF)**, not time-to-failure (TTF) | The guide confirmed loading rates differ across tests, so TTF is not directly comparable across specimens. DTF is rate-invariant. We convert DTF to seconds at inference time using the known cross-head rate. | Guide's note that loading rates differed |
| D3 | All resistance work in **ΔR/R₀** (fractional change), never raw R | Baseline R varies 7 to 59 MΩ between bars meant to be identical. The shape of the signal is what carries information, not the absolute level. ΔR/R₀ removes the baseline scatter. | Observation of baseline spread |
| D4 | **Multi-task** model: DTF (primary) + load + deflection + damage stage | Shared encoder gives auxiliary heads a regularisation effect. With n = 4 we want every bit of signal. The auxiliaries are also useful in their own right (virtual sensing, stage tagging). | Small-data constraint |
| D5 | **Leave-one-specimen-out** (LOSO) cross-validation | The unit of generalisation is "a new bar". Random splits within a specimen are not informative. With 4 specimens this gives 4 folds and 4 test points. | The standard for small specimen counts in SHM |
| D6 | **Simple models first**, then sequence models | Gradient-boosted trees on hand-engineered features are well known to be hard to beat on small structured data, and they give us a fair baseline to argue against later. Going straight to deep models is unjustifiable with n = 4. | Standard small-data discipline |
| D7 | **Heavy data augmentation** (see §4.2 for full list) | The effective dataset is tiny. Augmentation expands the effective sample count by orders of magnitude when done correctly. | Small-data constraint |
| D8 | **Causal features only**; no future information leaks into the model at any time t | The model is meant to be deployable in real time on a beam being loaded. Features at time t must use only data at or before t. | Real-world deployment intent |
| D9 | Loading rate included as an **input feature**, and time-rate features are normalised by it | Different tests have different rates. The model must see the rate so it can convert internal representations correctly. | Guide's note on rate variability |
| D10 | Headline metric is **alarm lead-time distribution** plus precision and recall of the alarm, evaluated under LOSO | This is what the project is for: how many seconds of warning the model gives before the load peak, on a bar the model has not seen. | Project goal |

---

## 4. Mitigation strategies, in depth

This section is the heart of what the guide asked for. For every challenge in the project, the mitigation is named, explained, justified, and tied to the implementation.

### 4.1 Small dataset (n = 4)

Four specimens is a small number for any model that has a non-trivial number of parameters. The mitigations are layered:

**4.1.1 Leave-one-specimen-out cross-validation (LOSO).**
We rotate which specimen is held out for testing while the other three form the training pool. Every model in every phase is evaluated this way. With 4 specimens we obtain 4 test rolls. Within the training pool we use a further small held-out segment of one of the three training specimens for early stopping and hyper-parameter selection (no random k-fold inside a specimen, because adjacent samples in a time series are highly correlated and would leak information). This is the only honest measurement of "how does the model perform on a bar it has not seen".

**4.1.2 Simple models before complex ones.**
The first model in Phase 2 is linear regression on a few hand-engineered features, purely as a floor. The second model is a gradient-boosted tree ensemble (LightGBM or XGBoost). These models have very few effective degrees of freedom relative to neural networks; with adequate regularisation they fit a small dataset without memorising it. They also give an honest benchmark; any later sequence model has to beat them by a meaningful margin to be worth the added complexity.

**4.1.3 Multi-task learning as implicit regularisation.**
The sequence model has one shared backbone and four small heads (DTF regression, load regression, deflection regression, stage classification). Each head's loss is averaged into a total loss with weights that we sweep. The auxiliary heads are forced to produce sensible values too, which constrains the backbone away from over-fitting to the primary head.

**4.1.4 Heavy augmentation (see §4.2).**
The single largest lever for a small dataset.

**4.1.5 Uncertainty estimation.**
For the headline regression target, we estimate prediction intervals, either by training a small set of bootstrapped models (5 to 10 bags over the training pool with replacement) and using their disagreement, or by training a quantile head that directly predicts the 10th, 50th and 90th percentile of DTF. The alarm rule then uses the predicted upper-bound on DTF rather than the median, which makes the alarm more conservative without sacrificing average lead time.

### 4.2 Data augmentation, in depth

Each augmentation is described as: what it does, why it helps, what range of parameters to use, and what it must not break.

**4.2.1 Random window crops.**
The model sees a window of length W of features ending at the current time t. From each specimen we can sample many such windows by sliding t along the time axis. We use a stride of 1 sample (about 0.083 s) during training and a stride of one window during evaluation. This is not strictly "augmentation": it is how a sequence model consumes a time series, but it expands the effective training set from 4 sequences to tens of thousands of windows. No data is invented.

**4.2.2 Mild time-warping.**
A training window is resampled onto a slightly stretched or compressed time axis (factor uniformly sampled in [0.9, 1.1], i.e. ±10%). Linear interpolation is used. The label is the new DTF at the new last sample. The point is to teach the model that the *shape* of the response is more important than its absolute timing, which is exactly the right inductive bias given that loading rates differ between specimens. We do not go beyond ±10% so that the warped curve still looks physically plausible.

**4.2.3 Additive noise on ΔR/R₀.**
A zero-mean Gaussian noise term with standard deviation in the range 0.1% to 0.3% (absolute, in ΔR/R₀ units) is added to the input features during training. The level is calibrated from the source-meter's measurement noise visible in the flat early portion of the signal, so the augmentation simulates the sensor noise that a deployed system would actually face. We do not add noise to the labels.

**4.2.4 Baseline-R perturbation.**
For each training window we recompute ΔR/R₀ using a perturbed baseline R₀' = R₀ · (1 + δ), with δ uniformly sampled in [-0.2, 0.2]. Because the four real bars already span an 8× spread in baseline resistance, this augmentation tells the model that the *shape* of the ΔR/R₀ curve is meaningful while a 20% shift in the assumed baseline is not. It directly addresses the generalisation-across-baselines question.

**4.2.5 Specimen mixup (used cautiously).**
For two random windows from different training specimens, we form a convex mixture: x' = λ x_a + (1 − λ) x_b, y' = λ y_a + (1 − λ) y_b, with λ ~ Beta(0.2, 0.2) so values cluster near 0 and 1. This is enabled in late training only, with a small weight, and only for the regression heads. We do not mix across very different stages (we only mix windows whose stage label matches) so we don't create physically nonsensical samples.

**4.2.6 What we will not do, and why.**
- We will **not** time-reverse windows for DTF prediction. Direction matters; a reversed window is post-failure, not pre-failure, and the labels would be wrong.
- We will **not** apply random gain (scaling ΔR/R₀ by a random multiplicative constant), because the shape of the curve is the signal, and arbitrary gain would corrupt the gauge-factor information.
- We will **not** use augmentations that change the load or deflection labels (e.g., adding noise to deflection). The target is a physical quantity, not a perceptual one.

### 4.3 Variable loading rate

Different specimens were loaded at different cross-head speeds, so the time axis is not comparable between specimens. The mitigations are:

- The **primary target is DTF**, in millimetres of remaining deflection until the load peak. This quantity is invariant to rate.
- The **loading rate is included as a per-specimen input feature**. The model can condition on it.
- Any feature that involves time directly (e.g., dΔR/dt) is **converted into a deflection-rate feature** (dΔR/d(deflection)) where possible, or normalised by the loading rate. Rolling-window statistics use a window expressed in millimetres of deflection rather than seconds.
- At inference time, if the user wants the prediction in seconds (early warning lead time in seconds), DTF is divided by the current cross-head speed to give TTF.

This means our test reports will quote both DTF (the model's native output) and TTF at the actual specimen's rate.

### 4.4 Baseline resistance spread (7 to 59 MΩ)

The four bars are meant to be identical but their R₀ values span almost a decade. This is process scatter in CNT dispersion. The mitigations are:

- Always work in **ΔR/R₀** rather than raw R, so the baseline cancels out.
- Optionally include **log(R₀) as a static input feature**, so the model can adjust if the *shape* of the response also depends slightly on the baseline level (we will test whether this helps in Phase 2 ablation).
- Use **baseline-R perturbation augmentation** (§4.2.4) so the model sees a wide effective range of R₀ during training.

### 4.5 Causality (no future leakage)

A deployable early-warning system can only use data up to the current instant. We enforce this strictly:

- All rolling-window features at time t are computed over the window [t − W, t]. No centred window. No future smoothing.
- Labels at time t use the future (DTF(t) = D_peak − D(t)), but only the labels, never the inputs.
- In the LOSO evaluation, predictions at every time t are made by feeding the model only the past of the test specimen.
- The "time since the first significant resistance jump" feature uses a causal detector: the threshold is crossed in real time, never set retrospectively.

This is verified in the test suite (a unit test confirms that randomising future samples leaves predictions unchanged).

### 4.6 Saturation in the middle of the test

ΔR/R₀ is flat through the middle of the test while the load and the deflection keep changing. This means **virtual sensing of load from resistance will have higher error in the saturation regime** than in regimes I or III. The mitigations are:

- We **report regression error by stage** (load RMSE in stage I, II, III separately), so a poor middle-regime fit does not hide behind a good early-regime fit.
- We **do not promise** virtual sensing as a primary deliverable; it is reported as auxiliary information.
- The **headline metric is failure forecasting**, where the signal is most informative (regime III).
- A possible Phase 3 extension is to add a "I am in regime II and resistance is uninformative" output, which the model can use to widen its uncertainty bounds in that regime.

### 4.7 Two-wire contact resistance

The source-meter records the resistance using a two-wire measurement, which means the recorded R includes the lead and contact resistance. For a deployable sensor this is realistic. The mitigations are:

- We do not attempt to remove the contact component within this project; it is part of the recorded signal.
- We note it as a limitation in the final report.
- If a future round of experiments uses a four-wire measurement, the entire pipeline still applies.

---

## 5. Phased plan, with sub-steps

Each phase is sized to be about a week of work for one person and produces a concrete artifact that can be reviewed by the guide before moving to the next phase.

### Phase 0: Data pipeline (reproducible loader)

**Objective.** Produce a single tidy parquet file containing all four specimens on a common time grid, with consistent units and pre-computed labels.

**Steps.**

1. Parse each UTM `.xls` (`xlrd` for the legacy format). Extract the columns *Time sec, Load kN, Rel. Stroke mm*. Apply the sign convention so that compressive load is positive and downward deflection is positive (currently both are negative in the raw data).
2. Parse each Sourcemeter `.csv`. Skip the metadata block. Extract the *Reading* column (resistance in ohms) and the *Relative Time* column (seconds from the start of the record).
3. Resample both signals onto a common 12 Hz grid using linear interpolation. Estimate any time offset between the two records (both start near t = 0; we check by aligning the first significant deflection event in UTM with the first significant resistance rise in the source-meter).
4. Compute ΔR/R₀ per specimen. R₀ is the median of the resistance over the first 1 s of the test (when no load has been applied).
5. Compute the load peak in time: `t_peak = argmax(load)`. Compute the corresponding deflection D_peak.
6. Compute the labels at every sample: TTF = t_peak − t (clipped at zero for post-peak), DTF = D_peak − D (clipped likewise).
7. Compute the loading rate per specimen by linear fitting D versus t on the first half of the test (before saturation effects).
8. Save to `data/processed/all_specimens.parquet` with the columns: `specimen, t, load_kN, deflection_mm, R_ohm, FCR_pct, TTF_s, DTF_mm, loading_rate_mm_per_s`.
9. QA: produce alignment plots for each specimen, and a single combined plot showing all four ΔR/R₀ vs. deflection curves.

**Deliverables.**
- `src/data/pipeline.py`, the loader and resampler, with a CLI entry point.
- `data/processed/all_specimens.parquet`.
- `notebooks/00_pipeline_qa.ipynb` with the alignment and sanity plots.

**Success criteria.** The notebook plots show that, for every specimen, the load reaches a peak then drops, the deflection grows monotonically, the resistance is monotonic until the post-peak regime, and the UTM and source-meter traces are visibly synchronised.

---

### Phase 1: EDA, feature engineering, and labels

**Objective.** Define the inputs and outputs that every model in Phases 2 and 3 will use, and document the choice of damage-stage labels.

**Steps.**

1. EDA plots: ΔR/R₀ vs. deflection per specimen; ΔR/R₀ vs. normalised progress (D / D_peak); dΔR/d(deflection) vs. deflection. Inspect the saturation regime.
2. **Feature set (causal, computed at every t):**
   - Instantaneous ΔR/R₀(t).
   - First derivative dΔR/d(deflection) (use a Savitzky-Golay filter with window of ~21 samples).
   - Second derivative d²ΔR/d(deflection)².
   - Rolling mean and standard deviation of ΔR/R₀ over the last 1 mm, 3 mm and 5 mm of deflection.
   - Cumulative change in ΔR/R₀ since the start.
   - Time since the **first significant jump** event, where a jump is defined as dΔR/d(deflection) crossing a per-specimen threshold (one robust standard deviation above its running median). The detection is causal.
   - Number of detected step events in the last 3 mm of deflection.
   - Static features (constant per specimen): loading rate, log(R₀).
3. **Label set:**
   - `DTF` (primary, regression): millimetres of remaining deflection.
   - `TTF` (secondary, regression): derived from DTF and the loading rate, reported for interpretability.
   - `load` (auxiliary, regression).
   - `deflection` (auxiliary, regression).
   - `stage` (auxiliary, classification): a four-class label, *early* (D < 0.1·D_peak), *elastic-rising* (0.1·D_peak ≤ D ≤ 0.8·D_peak), *pre-failure* (0.8·D_peak ≤ D ≤ D_peak), *post-failure* (D > D_peak). Note the thresholds are deflection-based, not time-based, so they are rate-invariant.
4. Save the full feature/label table to `data/processed/features.parquet`.

**Deliverables.**
- `src/features/engineering.py`, feature computation, with unit tests for causality (a test that randomises samples after t and checks features at t are unchanged).
- `src/features/labels.py`, label computation.
- `data/processed/features.parquet`.
- `notebooks/01_features_eda.ipynb`.

**Success criteria.** Every feature is computed using only past data (verified by the causality unit test). Each feature has a clear physical interpretation (no opaque transformations). The label table is correctly aligned and free of NaN where it shouldn't be.

---

### Phase 2: Classical baselines under LOSO

**Objective.** Establish a fair benchmark for every metric using simple, well-understood models.

**Steps.**

1. Implement a **LOSO harness** in `src/evaluation/loso.py`. The harness takes a model class, a feature table, a list of specimens, and returns a per-fold scorecard. For each fold it trains on three specimens (with a small held-out segment of one of them for early stopping) and tests on the fourth.
2. Implement **two baseline regressors**:
   - A linear regression on a hand-picked subset of features (floor).
   - A gradient-boosted tree ensemble (LightGBM, multi-output via per-target models since LightGBM is single-output). Search over a small grid (number of leaves, learning rate, regularisation) within LOSO inner CV.
3. Implement a **stage classifier**: gradient-boosted multi-class classifier on the same features.
4. **Compute metrics per fold and aggregate:**
   - DTF MAE and median absolute error.
   - TTF MAE (derived).
   - Load RMSE and per-stage RMSE.
   - Deflection RMSE.
   - Stage accuracy, balanced accuracy, confusion matrix.
   - Alarm precision, recall, F1 across a sweep of alarm thresholds (predicted DTF below τ_alarm for N consecutive samples).
   - Lead-time distribution of true-positive alarms.
5. Produce a markdown scorecard `reports/phase2_scorecard.md` with a table and the per-specimen pred-vs-true overlays.

**Deliverables.**
- `src/models/baselines.py`
- `src/evaluation/loso.py`
- `src/evaluation/metrics.py`
- `reports/phase2_scorecard.md`
- `reports/figures/phase2/*.png`

**Success criteria.** The harness runs end-to-end. The numbers are reproducible (fixed random seed, recorded in the scorecard). The baselines give a non-trivial DTF MAE (we target below 30% of the average D_peak as a rough sanity floor).

---

### Phase 3: Sequence models with augmentation

**Objective.** Build a small but capable sequence model that beats the Phase 2 baselines, and characterise where it does and does not help.

**Steps.**

1. **Inputs.** The model consumes a causal window of length W (in samples) ending at the current time. We start with W corresponding to about 2 mm of deflection at the median loading rate. The window contains the features computed in Phase 1, plus the static features broadcast across the window.
2. **Architecture, in order of preference.**
   - A small **1-D temporal convolutional network** (TCN-lite): three dilated convolutional blocks with kernel size 3, dilations 1/2/4, 32 channels, causal padding. Residual connections. About 30 k parameters.
   - A **small LSTM**: one or two layers, hidden size 32 to 64, last-output head. Causal by construction.
   - A **tiny Transformer encoder** is kept as a stretch experiment, with the expectation that with n = 4 it will likely under-perform the simpler models.
3. **Heads.** A shared backbone feeds four small linear heads:
   - DTF regression (Huber loss).
   - Load regression (MSE).
   - Deflection regression (MSE).
   - Stage classification (cross-entropy).
4. **Loss.** Weighted sum, weights start at 1.0 / 0.3 / 0.3 / 0.5 and are tuned within the inner CV. The DTF head's weight dominates.
5. **Optimiser.** AdamW, learning rate 3e-4, weight decay 1e-4, cosine schedule, early stopping on the held-out validation segment.
6. **Augmentation pipeline.** Applied at sample time during training: window crop, then with probability 0.5 a time warp, then additive noise, then with probability 0.3 a baseline-R perturbation. Mixup is enabled only for the last few epochs.
7. **Inference at evaluation time.** No augmentation; the model receives the actual causal window from the held-out specimen.
8. **Ablations.** With and without augmentation; with and without each auxiliary head; with TCN vs. LSTM; with W = 1 mm vs. 2 mm vs. 5 mm.

**Deliverables.**
- `src/models/sequence.py`, model definitions.
- `src/training/train_sequence.py`, training loop.
- `src/training/augmentations.py`, augmentation operators.
- `reports/phase3_scorecard.md`, comparison to Phase 2.
- `reports/figures/phase3/*.png`.

**Success criteria.** The best sequence model improves on the Phase 2 LightGBM by a meaningful margin (we target ≥ 15% reduction in DTF MAE) without losing alarm precision.

---

### Phase 4: Calibration and alarm policy

**Objective.** Turn the best regression model into a well-calibrated alarm and report its operating curve.

**Steps.**

1. **Quantile heads.** Replace the DTF Huber head with three pinball-loss heads predicting the 10th, 50th and 90th percentile of DTF. Train under LOSO as before. Confirm the 90th percentile is conservatively above the median on the held-out specimens.
2. **Bootstrap alternative.** Train ten copies of the best model on bootstrap resamples of the training pool. Use their disagreement as a second uncertainty estimate. Compare to the quantile-based interval.
3. **Alarm rule.** Alarm when the **upper bound** on DTF drops below τ_alarm for at least N consecutive samples.
4. **Operating curve.** Sweep τ_alarm × N. For each combination, compute the median lead time (in seconds, using the specimen's loading rate), the precision (fraction of alarms that are true), and the recall (fraction of true failures that get a timely alarm). Plot.
5. **Pick an operating point.** Choose the (τ_alarm, N) that maximises lead time subject to recall ≥ 0.75 and precision ≥ 0.8, as our default. We can change the constraint if the guide gives an operating-point preference later.
6. **Per-specimen narrative.** Re-plot pred-vs-true for each held-out specimen with the alarm event marked.

**Deliverables.**
- `src/evaluation/calibration.py`
- `src/evaluation/alarm.py`
- `reports/phase4_operating_curve.md`
- `reports/figures/phase4/*.png`

**Success criteria.** The chosen operating point gives a median lead time of at least 5 seconds at the medium loading rate, with recall ≥ 0.75 and precision ≥ 0.8 on LOSO.

---

### Phase 5: Final report, artifacts, and clean repo

**Objective.** Hand in a substantive, defensible final deliverable.

**Steps.**

1. Per-specimen pred-vs-true plots, time-series and operating curves into the LaTeX report (already drafted, in `final_report/`).
2. A new section in the report: *Results* (Phase 2 vs. Phase 3 vs. Phase 4) with the scorecards.
3. *Limitations* section, candid: n = 4, LOSO uncertainty bands wide, no claim of cross-material generalisation, 2-wire contact effect un-modelled.
4. *Reproducibility* appendix: how to install dependencies, run the pipeline, and reproduce every number.
5. Clean the repo: ensure `make all` (or an equivalent script) reproduces the parquets, the scorecards, and the report figures from the raw data.

**Deliverables.**
- Final PDF of the report.
- Clean, reproducible repository.

---

## 6. Repository structure

```
ugrp/
├── PROJECT_PLAN.md            design and methodology summary
├── README.md                  setup and reproduction notes
├── requirements.txt           Python dependencies
├── UTM/                       raw UTM .xls files
├── Sourcemeter/               raw source-meter .csv files
├── code/
│   ├── Makefile               reproducibility entry points
│   ├── src/
│   │   ├── data/              pipeline and QA figures
│   │   ├── features/          causal feature engineering and labels
│   │   ├── models/            baselines and sequence models
│   │   ├── training/          augmentation and the sequence-model trainer
│   │   └── evaluation/        LOSO harness, metrics, calibration, alarm, prediction
│   ├── tests/                 causality unit test
│   ├── notebooks/             exploratory QA and EDA notebooks
│   ├── data/processed/        generated parquet tables
│   └── reports/               scorecards and figures
└── final_report/              the report and presentation
```

---

## 7. Metrics and success criteria

| Metric | What it measures | Where reported | Target (initial) |
|--------|-------------------|----------------|------------------|
| DTF MAE | How accurately the model predicts remaining deflection to failure | All phases | < 1.5 mm under LOSO |
| TTF MAE (derived) | Same in seconds, at the specimen's actual rate | All phases | < 15 s under LOSO |
| Load RMSE per stage | Virtual sensing accuracy by regime | Phase 2 onward | < 0.02 kN in stages I and III; documented in stage II |
| Stage accuracy | Damage-stage classification | Phase 2 onward | > 0.8 balanced |
| Alarm precision | Fraction of alarms that precede a real failure | Phase 4 | ≥ 0.8 |
| Alarm recall | Fraction of true failures with a timely alarm | Phase 4 | ≥ 0.75 |
| Median alarm lead time | Seconds before the load peak at which the alarm first fires | Phase 4 | ≥ 5 s at the median rate |

These targets are initial; we will refine them once the Phase 2 baseline is in hand. If a baseline already meets a target the target is raised.

---

## 8. Risks, limitations, and what we are not claiming

- **Sample size.** Four specimens give us four LOSO test points. The variance of any reported metric is large. We will report confidence intervals where possible and avoid over-claiming.
- **Rate variability.** Loading rates differ across tests; this is a real confound. The DTF target is the principal mitigation.
- **Contact resistance.** A two-wire measurement is used; the recorded resistance includes contact effects. This is realistic for a deployed sensor but is not the cleanest signal.
- **One material, one geometry.** Results say nothing about other composites, other geometries, or other loading modes.
- **Causality.** Real-time deployment is the use case, so all features are causal. We do not assume access to future data at any point.
