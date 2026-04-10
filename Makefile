PYTHON ?= python3
VENV := venv
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
APP := app.py
STAMP := $(VENV)/.ready

.DEFAULT_GOAL := run
.PHONY: run clear

$(STAMP): requirements.txt
	@echo "Setting up virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@touch $(STAMP)

run: $(STAMP)
	@echo "Starting Strategy Lab..."
	@$(STREAMLIT) run $(APP)

clear:
	@echo "Removing $(VENV)..."
	@rm -rf $(VENV)
