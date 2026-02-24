PYTHON ?= python

.PHONY: setup download ingest build_dataset correlate train_sup train_unsup eval infer clean

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev]"

download:
	$(PYTHON) scripts/00_download_datasets.py --out data/raw --ctu13_scenario 44 --cisco_folder 0

ingest:
	$(PYTHON) scripts/01_ingest.py --raw data/raw --out data/interim

build_dataset:
	$(PYTHON) scripts/02_build_dataset.py --interim data/interim --out data/processed

correlate:
	$(PYTHON) scripts/03_correlate_patterns.py --processed data/processed --out artifacts

train_sup:
	$(PYTHON) scripts/04_train_supervised.py --processed data/processed --out artifacts

train_unsup:
	$(PYTHON) scripts/05_train_unsupervised.py --processed data/processed --out artifacts

eval:
	$(PYTHON) scripts/06_evaluate.py --processed data/processed --artifacts artifacts

infer:
	$(PYTHON) scripts/07_infer.py --processed data/processed --artifacts artifacts

clean:
	rm -rf data/interim/* data/processed/* artifacts/models/* artifacts/reports/* artifacts/figures/*
