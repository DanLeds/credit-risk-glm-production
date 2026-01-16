# Credit Risk GLM Production Makefile

.PHONY: help install test build deploy clean monitor benchmark

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
NAMESPACE := ml-models
IMAGE_NAME := credit-risk-api
IMAGE_TAG := latest
REGISTRY := ghcr.io/yourorg

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "Credit Risk GLM Production Framework"
	@echo "=============================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Tests completed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/integration/ -v -m integration

lint: ## Run linting and formatting checks
	@echo "$(YELLOW)Running linters...$(NC)"
	black --check src/ tests/
	flake8 src/ tests/
	mypy src/
	@echo "$(GREEN)Linting completed!$(NC)"

format: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(NC)"
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "$(GREEN)Image built: $(IMAGE_NAME):$(IMAGE_TAG)$(NC)"

push: ## Push Docker image to registry
	@echo "$(YELLOW)Pushing image to registry...$(NC)"
	$(DOCKER) tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	$(DOCKER) push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "$(GREEN)Image pushed to $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)$(NC)"

run-local: ## Run API locally
	@echo "$(YELLOW)Starting API locally...$(NC)"
	MODEL_PATH=models/glm_model.joblib $(PYTHON) src/api_service.py

run-docker: ## Run with Docker Compose
	@echo "$(YELLOW)Starting services with Docker Compose...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started! API available at http://localhost:5000$(NC)"

stop-docker: ## Stop Docker Compose services
	@echo "$(YELLOW)Stopping services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped!$(NC)"

deploy-k8s: ## Deploy to Kubernetes
	@echo "$(YELLOW)Deploying to Kubernetes...$(NC)"
	$(KUBECTL) create namespace $(NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -
	$(KUBECTL) apply -f k8s/ -n $(NAMESPACE)
	@echo "$(GREEN)Deployed to Kubernetes namespace: $(NAMESPACE)$(NC)"

undeploy-k8s: ## Remove from Kubernetes
	@echo "$(YELLOW)Removing from Kubernetes...$(NC)"
	$(KUBECTL) delete -f k8s/ -n $(NAMESPACE)
	@echo "$(GREEN)Removed from Kubernetes!$(NC)"

scale: ## Scale deployment (usage: make scale REPLICAS=5)
	@echo "$(YELLOW)Scaling deployment to $(REPLICAS) replicas...$(NC)"
	$(KUBECTL) scale deployment credit-risk-api --replicas=$(REPLICAS) -n $(NAMESPACE)
	@echo "$(GREEN)Scaled to $(REPLICAS) replicas!$(NC)"

monitor: ## Open monitoring dashboards
	@echo "$(YELLOW)Opening monitoring dashboards...$(NC)"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	open http://localhost:3000 || xdg-open http://localhost:3000

logs: ## View API logs
	@echo "$(YELLOW)Viewing logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f credit-risk-api

logs-k8s: ## View Kubernetes logs
	@echo "$(YELLOW)Viewing Kubernetes logs...$(NC)"
	$(KUBECTL) logs -f deployment/credit-risk-api -n $(NAMESPACE)

benchmark: ## Run performance benchmarks
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	$(PYTHON) tests/performance/benchmark.py
	@echo "$(GREEN)Benchmarks completed! Check benchmark_report.html$(NC)"

load-test: ## Run load tests
	@echo "$(YELLOW)Running load tests...$(NC)"
	locust -f tests/performance/locustfile.py --host=http://localhost:5000 --users=100 --spawn-rate=10 --time=5m --headless
	@echo "$(GREEN)Load tests completed!$(NC)"

train: ## Train a new model
	@echo "$(YELLOW)Training model...$(NC)"
	$(PYTHON) scripts/train_model.py
	@echo "$(GREEN)Model training completed!$(NC)"

migrate-db: ## Run database migrations
	@echo "$(YELLOW)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)Migrations completed!$(NC)"

backup: ## Backup models and data
	@echo "$(YELLOW)Creating backup...$(NC)"
	mkdir -p backups
	tar -czf backups/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz models/ data/
	@echo "$(GREEN)Backup created in backups/$(NC)"

clean: ## Clean temporary files
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf htmlcov/ .coverage
	rm -rf dist/ build/ *.egg-info
	@echo "$(GREEN)Cleanup completed!$(NC)"

security-scan: ## Run security scanning
	@echo "$(YELLOW)Running security scans...$(NC)"
	trivy fs --severity HIGH,CRITICAL .
	safety check -r requirements.txt
	bandit -r src/
	@echo "$(GREEN)Security scan completed!$(NC)"

docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	sphinx-apidoc -o docs/source src/
	cd docs && make html
	@echo "$(GREEN)Documentation generated in docs/_build/html/$(NC)"

version: ## Show version information
	@echo "$(GREEN)Credit Risk GLM Production Framework$(NC)"
	@echo "Version: $(shell git describe --tags --always)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Docker: $(shell $(DOCKER) --version)"
	@echo "Kubernetes: $(shell $(KUBECTL) version --client --short)"

all: install lint test build ## Run all steps

.DEFAULT_GOAL := help
