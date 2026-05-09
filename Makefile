PYTHON ?= python3
VENV := venv
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
APP := app.py
STAMP := $(VENV)/.ready

.DEFAULT_GOAL := run
.PHONY: run clear test regression

$(STAMP): requirements.txt
	@echo "Setting up virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@touch $(STAMP)

run: $(STAMP)
	@echo "Starting Strategy Lab..."
	@$(STREAMLIT) run $(APP)

test: $(STAMP)
	@$(VENV)/bin/python -m pytest tests/ -v

regression: $(STAMP)
	@$(VENV)/bin/python -m pytest tests/regression/ -v

clear:
	@echo "Removing $(VENV)..."
	@rm -rf $(VENV)
