# SwaraFlow -- common developer commands.
# On Windows, use the .bat/.py scripts directly (see scripts/) instead
# of make, unless you have a make-compatible shell (e.g. Git Bash, WSL).

.PHONY: setup dev test test-backend test-frontend lint models docker-up docker-down clean

setup:
	python scripts/setup.py

dev:
	python scripts/run-dev.py

test:
	python scripts/run-tests.py

test-backend:
	python scripts/run-tests.py --backend

test-frontend:
	python scripts/run-tests.py --frontend

lint:
	.venv/bin/ruff check backend/
	.venv/bin/mypy backend/ --ignore-missing-imports
	cd frontend && npm run lint

models:
	python scripts/download_models.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf frontend/.next backend/.pytest_cache
