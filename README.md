# Self-Sensing GFRP/CNT Composites under Flexural Loading

Undergraduate research project, AS4545, Department of Aerospace Engineering,
Indian Institute of Technology Madras.

Author: Priyanshu Kumar (AE22B048)

## Overview

Four nominally identical bars of a glass-fibre composite with carbon nanotubes
(CNTs) dispersed in the matrix were loaded to failure in three-point bending. A
universal testing machine recorded the load and deflection while a Keithley
source-meter recorded the electrical resistance of the embedded CNT network, both
at about 12 Hz. Because the glass fibres do not conduct, the resistance reflects
the state of the matrix, where damage begins. This project uses that resistance
signal to forecast failure and raise an early-warning alarm on a bar the model has
never seen.

The work is organised in five stages: a reproducible data pipeline, a causal
feature set, classical baselines, sequence models with data augmentation, and a
calibrated alarm. Every model is evaluated by leave-one-specimen-out (LOSO)
cross-validation, which trains on three bars and tests on the fourth.

## Headline results

All figures are under LOSO with a fixed random seed.

| Stage | Result |
|-------|--------|
| Classical baseline (gradient-boosted trees) | deflection-to-failure error of 1.51 mm |
| Sequence models (TCN, LSTM) | best of 1.49 mm; competitive but no clear gain over the baseline |
| Calibrated alarm | precision 1.00, recall 1.00, median warning of 73.9 s, no false alarms |

The full analysis, design decisions, and a dated progress record are in
[`PROJECT_PLAN.md`](./PROJECT_PLAN.md). The final report is in
[`final_report/`](./final_report/).

## Repository layout

```
.
├── PROJECT_PLAN.md          design document, decision log, progress log
├── README.md
├── requirements.txt
├── UTM/                     raw UTM .xls files (S1..S4)
├── Sourcemeter/             raw source-meter .csv files (s1..s4)
├── code/
│   ├── Makefile             reproducibility entry points
│   ├── src/
│   │   ├── data/            pipeline and QA figures
│   │   ├── features/        causal feature engineering and labels
│   │   ├── models/          baselines and sequence models
│   │   ├── training/        augmentation and the sequence-model trainer
│   │   └── evaluation/      LOSO harness, metrics, calibration, alarm
│   ├── tests/               causality unit test
│   ├── notebooks/           exploratory QA and EDA notebooks
│   ├── data/processed/      generated parquet tables
│   └── reports/             scorecards and figures
└── final_report/            LaTeX report and compiled PDF
```

## Setup

The code targets Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The gradient-boosted-tree models use LightGBM, which needs the OpenMP runtime. On
macOS this is provided by `libomp`:

```bash
brew install libomp
```

## Reproducing the results

All commands run from the `code/` directory.

```bash
cd code

make pipeline    # build the tidy 12 Hz dataset from the raw files
make features    # causal features and labels
make test        # the causality unit test
make phase2      # classical LOSO baselines and scorecard
make phase3      # sequence models, augmentation, and ablation
make phase4      # calibration and the alarm operating curve
make report      # compile the final report (needs a LaTeX installation)
```

To apply the trained model to a new specimen and, if it was loaded to failure,
score its accuracy:

```bash
make predict UTM=path/to/new.xls SM=path/to/new.csv NAME=S5
```

This trains a final model on all labelled specimens, runs the same causal
pipeline on the new bar, and writes a predictions CSV and a deflection-to-failure
plot to `code/reports/`.

`make all` runs the full pipeline end to end. Intermediate tables are written to
`code/data/processed/`, and the per-stage scorecards and figures to
`code/reports/`. Random seeds are fixed and recorded in each scorecard.

## Method notes

Three rules are enforced throughout and motivated in `PROJECT_PLAN.md`:

- Features are causal. A feature at time t uses only data up to time t, which a
  unit test verifies. This keeps the system deployable in real time.
- Generalisation is measured only by leave-one-specimen-out cross-validation,
  never by a random split within a bar.
- All resistance work is in the fractional change, ΔR/R₀, because the baseline
  resistance varies almost tenfold between bars.
