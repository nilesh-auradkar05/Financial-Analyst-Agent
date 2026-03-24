# Alpha Analyst Makefile
# ======================

.PHONY: help install test eval eval-single lint format docker-up docker-down clean

# Default target
help:
	@echo "Alpha Analyst - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies with Poetry"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run tests excluding slow ones"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make format        Format code (ruff)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval          Run full evaluation suite"
	@echo "  make eval-single   Run evaluation for AAPL (default)"
	@echo "  make eval-full     Run with Ragas/DeepEval (requires API keys)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     Start full stack (API + Prometheus + Grafana)"
	@echo "  make docker-down   Stop all containers"
	@echo "  make docker-logs   View container logs"
	@echo ""
	@echo "Server:"
	@echo "  make serve         Start API server (development)"
	@echo "  make serve-prod    Start API server (production)"
	@echo ""

# =============================================================================
# SETUP
# =============================================================================

install:
	uv install
	@echo ""
	@echo "✅ Dependencies installed!"
	@echo ""
	@echo "Optional: Install evaluation dependencies:"
	@echo "  uv install --with eval"

install-all:
	uv install --all-extras --with dev,eval
	@echo ""
	@echo "✅ All dependencies installed!"

# =============================================================================
# DEVELOPMENT
# =============================================================================

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -m "not slow"

test-cov:
	uv run pytest tests/ -v --cov=. --cov-report=html --cov-report=term

smoke-test:
	uv run python scripts/smoke_test_live_pipeline.py --ticker AAPL

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy .

# =============================================================================
# EVALUATION
# =============================================================================

eval:
	uv run python3 -m evaluation.runner

eval-single:
	uv run python3 -m evaluation.runner AAPL

eval-full:
	uv run python3 -m evaluation.runner --full

eval-report:
	@echo "Generating report from latest results..."
	uv run python3 -m evaluation.report evaluation/reports/*.json

eval-retrieval:
	uv run python3 -m evaluation.retrieval_main evaluation/fixtures/retrieval_baseline.json

eval-retrieval-single:
	uv run python3 -m evaluation.retrieval_main evaluation/fixtures/retrieval_baseline.json --case appl_risk_supply_chain

# =============================================================================
# DOCKER
# =============================================================================

docker-up:
	docker compose up -d
	@echo ""
	@echo "✅ Services started!"
	@echo ""
	@echo "  API:        http://localhost:8000"
	@echo "  Swagger:    http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo ""

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-build:
	docker compose build --no-cache

# =============================================================================
# SERVER
# =============================================================================

serve:
	uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

serve-prod:
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# =============================================================================
# UTILITIES
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	@echo "✅ Cleaned!"

# Quick health check
check:
	@echo "Checking services..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ API not running"
	@echo ""
