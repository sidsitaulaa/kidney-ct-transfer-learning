.PHONY: install data test lint smoke smoke-all stage1 stage2 stage3 all

DATA ?= data/ct-kidney
THREADS ?= 6

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py --source kaggle

test:
	pytest -q

lint:
	ruff check src scripts tests

smoke:
	python scripts/run_experiment.py --smoke --threads 2

smoke-all:  ## every stage on synthetic data
	python scripts/run_all.py --smoke --threads 2

stage1:  ## benchmark of the five frozen backbones (results/REPORT.md)
	python scripts/run_experiment.py --data-dir $(DATA) --threads $(THREADS)

stage3:  ## fine-tune MobileNet with the settings chosen on fold-0 validation (results/finetune/MobileNet)
	python scripts/run_finetune.py --data-dir $(DATA) --backbone MobileNet --stride 2 --unfreeze 60 \
		--finetune-lr 5e-5 --finetune-epochs 15 --patience 4 --threads $(THREADS)

stage2:  ## stronger classifiers + comparison with stage 1 and stage 3 (results/IMPROVEMENTS.md)
	python scripts/run_improvements.py --data-dir $(DATA) --threads $(THREADS)

all:  ## every stage in order, including the second fine-tuning run (deep ensemble)
	python scripts/run_all.py --data-dir $(DATA) --threads $(THREADS)
