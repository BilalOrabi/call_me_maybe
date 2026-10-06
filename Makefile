UV              := uv

.PHONY: all install run debug clean lint lint-strict help

all: run

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install dependencies via uv"
	@echo "  run          Execute the main pipeline (python -m src)"
	@echo "  debug        Run in debug mode via pdb"
	@echo "  clean        Remove temporary files and caches"
	@echo "  lint         Run flake8 and mypy with standard flags"
	@echo "  lint-strict  Run flake8 and mypy in strict mode"
	@echo "  help         Display this help message"

install:
	$(UV) sync

run:
	$(UV) run python -m src

debug:
	$(UV) run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	$(UV) run flake8 src --count
	$(UV) run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 src --count
	$(UV) run mypy src --strict
