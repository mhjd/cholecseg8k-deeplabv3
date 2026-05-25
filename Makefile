PYTHONPATH := .
PYTHON := PYTHONPATH=$(PYTHONPATH) python

.PHONY: test-dataset inspect-mask check-dataset-access

test-dataset:
	$(PYTHON) scripts/test_dataset.py

inspect-mask:
	$(PYTHON) scripts/inspect_one_mask.py

check-dataset-access:
	$(PYTHON) scripts/check_dataset_access.py
