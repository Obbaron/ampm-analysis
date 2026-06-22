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
.PHONY: help setup offline wheels test run build clean profile profile-cpu profile-cluster profile-direct profile-dhxml profile-compare profile-view

help:
	@echo "  setup    - create the venv and install the project (editable, online)"
	@echo "  offline  - create the venv and install from bundled wheels (no network)"
	@echo "  wheels   - (re)build the offline wheel set in $(WHEEL_DIR)"
	@echo "  test     - run the test suite"
	@echo "  run      - run the app"
	@echo ""
	@echo "  profile         - profile a driver with scalene (override DRIVER=, OUT=)"
	@echo "  profile-cpu     - same, CPU-only (faster, skips memory)"
	@echo "  profile-cluster - profile + view the clustering assignment path"
	@echo "  profile-direct  - profile + view the direct assignment path"
	@echo "  profile-dhxml   - profile + view the dhxml (bounding-box) assignment path"
	@echo "  profile-compare - profile + view direct vs dhxml on one masked frame"
	@echo "  profile-view    - open the last profile (override OUT= to choose)"
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

SCALENE       := $(VENV_PYTHON) -m scalene
PROFILE_SCOPE ?= ampm
DRIVER        ?= profile.py
OUT           ?= scalene-profile.json

profile:
	$(SCALENE) run --profile-only $(PROFILE_SCOPE) --outfile $(OUT) $(DRIVER)

profile-cpu:
	$(SCALENE) run --cpu-only --profile-only $(PROFILE_SCOPE) --outfile $(OUT) $(DRIVER)

profile-cluster:
	$(SCALENE) run --profile-only $(PROFILE_SCOPE) --outfile profile-cluster.json profile.py
	$(SCALENE) view profile-cluster.json

profile-direct:
	$(SCALENE) run --profile-only $(PROFILE_SCOPE) --outfile profile-direct.json profile_direct.py
	$(SCALENE) view profile-direct.json

profile-dhxml:
	$(SCALENE) run --profile-only $(PROFILE_SCOPE) --outfile profile-dhxml.json profile_dhxml.py
	$(SCALENE) view profile-dhxml.json

profile-compare:
	$(SCALENE) run --profile-only $(PROFILE_SCOPE) --outfile profile-compare.json profile_compare.py
	$(SCALENE) view profile-compare.json

profile-view:
	$(SCALENE) view $(OUT)

build:
	$(VENV_PYTHON) -m PyInstaller app.spec

ifeq ($(OS),Windows_NT)
clean:
	-Remove-Item -Recurse -Force .pytest_cache, build, dist, *.egg-info, scalene-profile.json, scalene-profile.html, profile-*.json -ErrorAction SilentlyContinue
	-Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
else
clean:
	rm -rf .pytest_cache build dist *.egg-info scalene-profile.json scalene-profile.html profile-*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
endif