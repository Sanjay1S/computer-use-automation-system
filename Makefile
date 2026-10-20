# Keep this project's environment separate from any virtualenv active in the calling shell.
unexport VIRTUAL_ENV

# Machine-local settings (gitignored, plain KEY=value lines). Read here and exported
# to the commands make runs, so `make parabank-up` and the parabank-local tenant
# URL agree on PARABANK_PORT. A command-line PARABANK_PORT=... still wins.
-include .env
export PARABANK_PORT PARABANK_USERNAME PARABANK_PASSWORD ANTHROPIC_API_KEY

PARABANK_IMAGE ?= parasoft/parabank
PARABANK_PORT ?= 8080

.PHONY: setup check fmt test test-live parabank-up parabank-down parabank-logs

setup:
	uv sync
	uv run playwright install chromium

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-live:
	uv run pytest -m live

parabank-up:
	docker run -d --name parabank -p $(PARABANK_PORT):8080 $(PARABANK_IMAGE)

parabank-down:
	docker rm -f parabank

parabank-logs:
	docker logs -f parabank
