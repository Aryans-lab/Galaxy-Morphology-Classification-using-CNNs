# One-command reproducibility:  make pipeline
#
# Assumes data/raw/ (Galaxy Zoo DR7 CSVs) and the GZ2 images in
# data/processed/images/ are already in place (see README "Data Setup").
PYTHON ?= python

.PHONY: pipeline merge process splits balance train train-resnet evaluate \
        evaluate-resnet classical compare visualise test clean

pipeline: process splits balance train evaluate classical compare visualise
	@echo "Pipeline complete. See evaluation/baseline_comparison.md"

merge:
	$(PYTHON) scripts/merge_and_filter.py

process:
	$(PYTHON) scripts/process_images.py

splits:
	$(PYTHON) scripts/make_splits.py

balance:
	$(PYTHON) scripts/balance_dataset.py

train:
	$(PYTHON) scripts/train_model.py

train-resnet:
	$(PYTHON) scripts/train_transfer_baseline.py

evaluate:
	$(PYTHON) scripts/evaluate_model.py \
	    --model-path models/galaxy_classifier.keras \
	    --output evaluation/metrics_cnn_custom.json

evaluate-resnet:
	$(PYTHON) scripts/evaluate_model.py \
	    --model-path models/galaxy_classifier_resnet50.keras \
	    --output evaluation/metrics_cnn_resnet50.json

classical:
	$(PYTHON) scripts/classical_baseline.py

compare:
	$(PYTHON) scripts/compare_baselines.py

visualise:
	$(PYTHON) scripts/visualise_result.py

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf .pytest_cache scripts/__pycache__ tests/__pycache__
