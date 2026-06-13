ifeq ($(OS),Windows_NT)
	SHELL := powershell.exe
	.SHELLFLAGS := -NoLogo -NoProfile -Command
	SYSTEM_PYTHON := python
	VENV_PYTHON := .venv\Scripts\python.exe
else
	SYSTEM_PYTHON := python3
	VENV_PYTHON := .venv/bin/python
endif

.DEFAULT_GOAL := help
.PHONY: help setup run build clean

help:
	@echo "  setup  - create the venv and install the project"
	@echo "  run    - run the app"
	@echo "  build  - compile a standalone binary from app.spec"
	@echo "  clean  - remove the caches and build artifacts"

setup:
	$(SYSTEM_PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

run:
	$(VENV_PYTHON) app.py

build:
	$(VENV_PYTHON) -m PyInstaller app.spec

ifeq ($(OS),Windows_NT)
clean:
	-Remove-Item -Recurse -Force .pytest_cache, build, dist -ErrorAction SilentlyContinue
	-Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
else
clean:
	rm -rf .pytest_cache build dist
	find . -type d -name __pycache__ -exec rm -rf {} +
endif