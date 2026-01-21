# Credit Risk GLM — Production Reference

> Silent execution. 
> Observable systems.

Production-grade reference implementation for **credit risk scoring using GLM (logistic regression)**, focused on **deployment, reliability, and interpretability** rather than model hype.

This repository is **not a library**.  
It is a **pragmatic ML Engineering blueprint**.

---

## Why this project exists

Most credit scoring projects stop at model training.  
This one doesn’t.

In regulated or high-stakes environments, the hardest part is not fitting a model —  
it’s **shipping it safely**, **monitoring it**, and **explaining its behavior** over time.

This project exists to show how a **classical, interpretable model** can be treated as a **first-class production system**.

---

## Scope & Philosophy

This repository focuses on:

- Treating ML as **software**, not experiments
- Making **interpretability a feature**, not a constraint
- Favoring **predictability over novelty**
- Building incrementally, only when needed

Out of scope (by design):

- AutoML
- Deep learning
- Online / real-time learning
- Feature stores
- Model hype

---

## Core Features

### Model Layer
- GLM (logistic regression) using **Statsmodels**
- Automated feature selection:
  - Random
  - Exhaustive
  - Forward / backward
- Stable coefficients & diagnostics
- Reproducible training pipeline

### Production Layer
- REST API for inference (Flask)
- Deterministic predictions
- Batch & single scoring
- Input validation & schema enforcement

### Observability
- Prometheus metrics
- Latency & throughput tracking
- Error monitoring
- Model health signals

### Reliability
- CI pipeline with tests
- Clear failure modes
- Explicit contracts between components

---

## Design Decisions

Some intentional trade-offs made in this project:

- **GLM over tree-based models**  
  Interpretability, auditability, and stability matter more than raw AUC.

- **Statsmodels over scikit-learn**  
  Better statistical diagnostics and coefficient control.

- **Simple REST API over async frameworks**  
  Predictable behavior under load > theoretical throughput.

- **No online learning**  
  Credit scoring favors consistency over adaptivity.

- **No multi-cloud abstraction**  
  Clarity beats optionality.

---

## Project Structure

```yaml
.
├── src/
│ ├── glm_model.py # Training & selection logic
│ ├── api_service.py # Inference API
│ └── metrics.py # Prometheus metrics
│
├── tests/
│ ├── unit/
│ ├── integration/
│ └── api/
│
├── models/
│ └── glm_model.joblib
│
├── docker/
│ └── Dockerfile
│
├── k8s/
│ ├── deployment.yaml
│ ├── service.yaml
│ └── hpa.yaml
│
└── README.md
```

---

## Quick Start

### 1. Train a Model

```python
from glm_model_production import ModelConfig, GLMModelSelector
import pandas as pd

data = pd.read_csv("your_data.csv")

config = ModelConfig(
    target_column="default_flag",
    predictors=data.columns.difference(["default_flag"]).tolist(),
    test_size=0.2,
    max_iterations=100
)

selector = GLMModelSelector(config)
selector.prepare_data(data)
model = selector.fit()

selector.save_model("models/glm_model.joblib")

print(model.metrics)
```

### 2. Serve the Model

```bash
MODEL_PATH=models/glm_model.joblib python src/api_service.py
```

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "credit_score": 710,
      "debt_ratio": 0.32,
      "tenure_months": 48
    }
  }'
```
---
## API Endpoints

| Method | Endpoint         | Description        |
| ------ | ---------------- | ------------------ |
| GET    | `/health`        | Liveness check     |
| GET    | `/ready`         | Readiness check    |
| POST   | `/predict`       | Single prediction  |
| POST   | `/predict/batch` | Batch predictions  |
| GET    | `/model/info`    | Model metadata     |
| GET    | `/metrics`       | Prometheus metrics |

---

## Deployment
### Docker
```bash
docker build -t credit-risk-glm .
docker run -p 5000:5000 credit-risk-glm
```

### Kubernetes (reference)
```bash
kubectl apply -f k8s/
kubectl get pods
```

Includes:

- Horizontal Pod Autoscaling
- Readiness & liveness probes
- Stateless inference pods
---

## Monitoring

Exposed metrics include:

- Prediction count
- Request latency
- Error rate
- Active requests

Grafana dashboards are intentionally minimal and focused on:

- SLO breaches
- Latency regressions
- Traffic anomalies
---

## Testing
```bash
pytest tests/ -v --cov
```

Test layers:
- Unit tests (model logic)
- Integration tests (pipeline)
- API tests (contracts & failures)
- Coverage target: signal over vanity.
---

## Build Notes

This project was built incrementally:
- Model training
- Deterministic inference
- API exposure
- Observability
- Scaling constraints
- Each layer was added only when it became necessary.
---

## What this repo is (and isn’t)


- ✅ A production-minded ML Engineering reference
- ✅ A discussion starter for interpretable ML in prod
- ❌ A plug-and-play framework
- ❌ A SaaS product
- ❌ A benchmark contest

## License
MIT



---

Built quietly.
Shipped deliberately.
