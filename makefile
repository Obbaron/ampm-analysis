ifeq ($(OS),Windows_NT)
	SHELL := powershell.exe
	.SHELLFLAGS := -NoLogo -NoProfile -Command
	SYSTEM_PYTHON := python
	VENV_PYTHON := .venv\Scripts\python.exe
	WHEEL_DIR := wheels/windows
else
	SYSTEM_PYTHON := python3
	VENV_PYTHON := .venv/bin/python
	WHEEL_DIR := wheels/linux
endif

PYCHECK := $(SYSTEM_PYTHON) -c "import sys,struct; sys.exit(0 if (sys.version_info[:2]==(3,11) and struct.calcsize('P')*8==64) else 'Need Python 3.11.x 64-bit to match the bundled cp311 wheels; found '+sys.version.split()[0])"

.DEFAULT_GOAL := help
.PHONY: help setup offline wheels test run build clean

help:
	@echo "  setup    - create the venv and install the project (editable, online)"
	@echo "  offline  - create the venv and install from bundled wheels (no network)"
	@echo "  wheels   - (re)build the offline wheel set in $(WHEEL_DIR)"
	@echo "  test     - run the test suite"
	@echo "  run      - run the app"
	@echo "  build    - compile a standalone binary from app.spec"
	@echo "  clean    - remove caches and build artifacts"

setup:
	$(PYCHECK)
	$(SYSTEM_PYTHON) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
	$(SYSTEM_PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

offline:
	$(PYCHECK)
	$(SYSTEM_PYTHON) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
	$(SYSTEM_PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install ".[dev]" --no-index --find-links $(WHEEL_DIR) --disable-pip-version-check

wheels:
	$(PYCHECK)
	$(SYSTEM_PYTHON) -c "import shutil; shutil.rmtree('$(WHEEL_DIR)', ignore_errors=True)"
	$(SYSTEM_PYTHON) -m pip download ".[dev]" "--only-binary=:all:" -d $(WHEEL_DIR)
	$(SYSTEM_PYTHON) -m pip download setuptools wheel "--only-binary=:all:" -d $(WHEEL_DIR)

test:
	$(VENV_PYTHON) -m pytest

run:
	$(VENV_PYTHON) app.py

build:
	$(VENV_PYTHON) -m PyInstaller app.spec

ifeq ($(OS),Windows_NT)
clean:
	-Remove-Item -Recurse -Force .pytest_cache, build, dist, *.egg-info -ErrorAction SilentlyContinue
	-Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
else
clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
endif