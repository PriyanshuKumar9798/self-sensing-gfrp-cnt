.PHONY: all pipeline qa features phase2 phase3 phase4 predict report test clean

NAME ?= NEW

# Reproduce the whole project from raw data (phase3 is the slow one, ~25 min).
all: pipeline features phase2 phase3 phase4
	@echo "All phases complete. Build the report with: make report"

# Phase 0 — build the tidy all-specimens parquet from raw UTM + Sourcemeter data.
pipeline:
	python3 -m src.data.pipeline

# Render the QA figures (also produced inline by notebooks/00_pipeline_qa.ipynb).
qa:
	python3 -m src.data.qa_figures

# Phase 1 — causal features + labels -> data/processed/features.parquet.
features:
	python3 -m src.features.build_features

test:
	python3 -m pytest -q

# Phase 2 — classical LOSO baselines -> reports/phase2_scorecard.md + figures.
phase2:
	python3 -m src.evaluation.run_phase2

# Phase 3 — sequence models + augmentation -> reports/phase3_scorecard.md + figures.
phase3:
	python3 -m src.evaluation.run_phase3

# Phase 4 — calibration + alarm policy -> reports/phase4_operating_curve.md + figures.
phase4:
	python3 -m src.evaluation.run_phase4

# Predict DTF/TTF for a new specimen from its raw files (scores accuracy if it failed).
# Usage: make predict UTM=path/to.xls SM=path/to.csv NAME=S5
predict:
	python3 -m src.evaluation.predict --utm "$(UTM)" --sm "$(SM)" --name "$(NAME)"

# Phase 5 — compile the final report (needs pdflatex on PATH; BasicTeX is enough).
report:
	cd ../final_report && pdflatex -interaction=nonstopmode GFRP-CNT_Self-Sensing_Report.tex \
	  && pdflatex -interaction=nonstopmode GFRP-CNT_Self-Sensing_Report.tex \
	  && rm -f *.aux *.toc *.out *.log

clean:
	rm -f data/processed/*.parquet
	rm -f reports/figures/phase0/*.png reports/figures/phase1/*.png reports/figures/phase2/*.png
