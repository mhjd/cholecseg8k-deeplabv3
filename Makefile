PYTHONPATH := .
PYTHON_BIN := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
PYTHON := PYTHONPATH=$(PYTHONPATH) $(PYTHON_BIN)

.PHONY: test-dataset inspect-mask check-dataset-access smoke-train train-baseline test-metrics eval-finetuned

test-dataset:
	$(PYTHON) scripts/test_dataset.py

smoke-train:
	$(PYTHON) scripts/smoke_train.py

train-baseline:
	$(PYTHON) scripts/train_baseline.py

eval-finetuned:
	$(PYTHON) scripts/eval_deeplabv3_finetuned.py

test-metrics:
	$(PYTHON) scripts/test_metrics.py

inspect-mask:
	$(PYTHON) scripts/inspect_one_mask.py

check-dataset-access:
	$(PYTHON) scripts/check_dataset_access.py
visualize_sample:
	$(PYTHON) scripts/visualize_sample.py
