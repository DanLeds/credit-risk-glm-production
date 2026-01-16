# Credit Risk GLM Production

[![CI/CD](https://github.com/yourusername/credit-risk-glm-production/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/yourusername/credit-risk-glm-production/actions) [![codecov](https://codecov.io/gh/yourusername/credit-risk-glm-production/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/credit-risk-glm-production) [![Docker](https://img.shields.io/docker/v/yourusername/credit-risk-api)](https://hub.docker.com/r/yourusername/credit-risk-api) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Overview

Production-ready framework for GLM (Generalized Linear Model) selection, training, and deployment with automatic model selection, comprehensive metrics, and enterprise-grade API serving.

### Key Features

- **Intelligent Model Selection**: Random, exhaustive, forward/backward selection strategies
- **Production API**: RESTful API with rate limiting, caching, and Prometheus metrics
- **Auto-scaling**: Kubernetes deployment with HPA and load balancing
- **Comprehensive Testing**: Unit tests, integration tests, and API tests with >90% coverage
- **Monitoring**: Built-in Prometheus metrics and Grafana dashboards
- **Type Safety**: Full type hints and mypy validation
- **Documentation**: Automated API docs and comprehensive guides

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Development](#development)
- [Contributing](#contributing)

## 🔧 Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/credit-risk-glm-production.git
cd credit-risk-glm-production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Run tests
pytest tests/ -v --cov

# Install pre-commit hooks
pre-commit install
```

### Docker Installation

```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -p 5000:5000 -v $(pwd)/models:/app/models credit-risk-api
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f credit-risk-api
```

## 🎯 Quick Start

### 1. Train Your Model

```python
from glm_model_production import ModelConfig, GLMModelSelector
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Configure model
config = ModelConfig(
    target_column="presence_unpaid",
    predictors=data.columns.difference(['presence_unpaid']).tolist(),
    max_iterations=100,
    test_size=0.2
)

# Train model
selector = GLMModelSelector(config)
train_data, test_data = selector.prepare_data(data)
best_model = selector.fit()

# Save model
selector.save_model("models/glm_model.joblib")

# Check metrics
print(f"AUC: {best_model.metrics.auc:.4f}")
print(f"Accuracy: {best_model.metrics.accuracy:.4f}")
```

### 2. Serve Model via API

```bash
# Start API server
MODEL_PATH=models/glm_model.joblib python api_service.py

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature1": 0.5,
      "feature2": -0.3,
      "feature3": 1.2
    }
  }'
```

### 3. Deploy to Production

```bash
# Deploy to Kubernetes
kubectl create namespace ml-models
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n ml-models
kubectl get svc -n ml-models
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Load Balancer                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │      Nginx Proxy        │
        └────────────┬────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼────┐    ┌────▼────┐    ┌────▼────┐
│  Pod 1  │    │  Pod 2  │    │  Pod 3  │
│  Flask  │    │  Flask  │    │  Flask  │
│   API   │    │   API   │    │   API   │
└─────────┘    └─────────┘    └─────────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
            ┌────────▼────────┐
            │  Model Storage  │
            │  (PVC/S3/GCS)   │
            └─────────────────┘
```

### Components

- **Model Core** (`glm_model_production.py`): Model training and selection logic
- **API Service** (`api_service.py`): Flask REST API with metrics
- **Load Balancer**: Distributes traffic across pods
- **Model Storage**: Persistent storage for trained models
- **Monitoring Stack**: Prometheus + Grafana for metrics

## 📚 API Reference

### Endpoints

|Method|Endpoint|Description|
|---|---|---|
|GET|`/health`|Health check|
|GET|`/ready`|Readiness check|
|POST|`/predict`|Single prediction|
|POST|`/predict/batch`|Batch predictions|
|GET|`/model/info`|Model information|
|GET|`/model/features`|Feature importance|
|POST|`/model/reload`|Reload model|
|GET|`/metrics`|Prometheus metrics|

### Example Requests

#### Single Prediction

```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={
        "features": {
            "credit_score": 720,
            "debt_to_income": 0.3,
            "months_employed": 60
        }
    }
)

result = response.json()
print(f"Probability: {result['prediction']['probability']}")
print(f"Class: {result['prediction']['predicted_class']}")
```

#### Batch Prediction

```python
response = requests.post(
    "http://localhost:5000/predict/batch",
    json={
        "data": [
            {"feature1": 0.5, "feature2": -0.3},
            {"feature1": 1.2, "feature2": 0.8}
        ],
        "include_confidence": True
    }
)

predictions = response.json()['predictions']
```

## 🚢 Deployment

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace ml-models

# Create secrets
kubectl create secret generic model-api-secrets \
  --from-literal=api-key=your-api-key \
  -n ml-models

# Apply configurations
kubectl apply -f k8s/

# Expose service
kubectl expose deployment credit-risk-api \
  --type=LoadBalancer \
  --port=80 \
  --target-port=5000 \
  -n ml-models
```

### Helm Chart

```bash
# Install with Helm
helm install credit-risk-api ./helm \
  --namespace ml-models \
  --set image.tag=v1.0.0 \
  --set replicas=3
```

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
docker build -t credit-risk-api .
docker tag credit-risk-api:latest $ECR_URI/credit-risk-api:latest
docker push $ECR_URI/credit-risk-api:latest

# Deploy to ECS
aws ecs update-service --cluster production --service credit-risk-api --force-new-deployment
```

## 📊 Monitoring

### Metrics Available

- `model_predictions_total`: Total predictions made
- `model_prediction_duration_seconds`: Prediction latency
- `model_accuracy`: Current model accuracy
- `model_errors_total`: Total errors by type
- `active_requests`: Current active requests

### Grafana Dashboard

Access Grafana at `http://localhost:3000` (default: admin/admin)

Pre-configured dashboards include:

- Model Performance Overview
- API Metrics
- Resource Utilization
- Error Analysis

### Alerts

Configured alerts in `monitoring/alerts.yml`:

- High error rate (>5%)
- High latency (P95 > 1s)
- Low model accuracy (<70%)
- Service down

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Specific test file
pytest tests/test_glm_model.py -v

# Integration tests only
pytest tests/ -m integration
```

### Test Coverage

Current coverage: **92%**

```
Name                      Stmts   Miss  Cover
----------------------------------------------
src/glm_model_production   385     31    92%
src/api_service            245     18    93%
----------------------------------------------
TOTAL                      630     49    92%
```

## 🛠️ Development

### Code Style

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
```

### Building Documentation

```bash
# Generate API docs
sphinx-apidoc -o docs/source src/

# Build HTML docs
cd docs && make html

# Serve locally
python -m http.server --directory docs/_build/html 8000
```

## 🔒 Security

### Best Practices

- ✅ Non-root Docker containers
- ✅ Secrets management with Kubernetes secrets
- ✅ Rate limiting on API endpoints
- ✅ Input validation and sanitization
- ✅ CORS configuration
- ✅ Health checks and readiness probes
- ✅ Network policies in Kubernetes
- ✅ TLS/SSL termination at load balancer

### Security Scanning

```bash
# Scan Docker image
trivy image credit-risk-api:latest

# Scan Python dependencies
safety check -r requirements.txt

# SAST analysis
bandit -r src/
```

## 🤝 Contributing

We welcome contributions! Please see our Contributing Guide.

### Development Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Statsmodels team for the GLM implementation
- Flask team for the excellent web framework
- All contributors and users of this framework

## 📞 Support

- 📧 Email: support@example.com
- 💬 Slack: [Join our workspace](https://slack.example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/credit-risk-glm-production/issues)
- 📖 Docs: [Documentation](https://docs.example.com)

---

**Made with ❤️ by the ML Engineering Team**