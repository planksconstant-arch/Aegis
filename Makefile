# Makefile for Local IDE RL Agent
# Run `make help` to see available targets.

.PHONY: help install install-dev test test-cov lint train eval dashboard clean

PYTHON ?= python
PIP    ?= pip

help:
	@echo ""
	@echo "  Local IDE RL Agent — Makefile Targets"
	@echo "  ─────────────────────────────────────"
	@echo "  make install      Install the package"
	@echo "  make install-dev  Install with dev + test dependencies"
	@echo "  make test         Run the full test suite"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make train        Train the policy for 50 episodes"
	@echo "  make eval         Run eval harness on held-out tasks"
	@echo "  make dashboard    Open the live training dashboard"
	@echo "  make clean        Remove build artifacts and .agent/ cache"
	@echo ""

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src/ tests/ || true

train:
	local-ide-agent train --episodes 50

eval:
	local-ide-agent eval

dashboard:
	local-ide-agent dashboard

clean:
	@echo "Removing build and cache artifacts..."
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache __pycache__ src/**/__pycache__
	@echo "Done. Note: .agent/ weights NOT removed. Use 'make clean-weights' to reset the agent."

clean-weights:
	@echo "WARNING: This will delete all trained weights and agent memory."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf .agent/ .shadow/
	@echo "Agent reset complete."
