"""
Shared fixtures for all tests in the credit-risk-glm-production project.
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
import json

import numpy as np
import pandas as pd
import pytest

# Mock external dependencies before importing modules
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.statsmodels"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["minio"] = MagicMock()
sys.modules["boto3"] = MagicMock()
sys.modules["great_expectations"] = MagicMock()
sys.modules["great_expectations.core"] = MagicMock()
sys.modules["great_expectations.core.batch"] = MagicMock()
sys.modules["feast"] = MagicMock()
sys.modules["feast.types"] = MagicMock()
sys.modules["evidently"] = MagicMock()
sys.modules["evidently.report"] = MagicMock()
sys.modules["evidently.metric_preset"] = MagicMock()
sys.modules["shap"] = MagicMock()
sys.modules["lime"] = MagicMock()
sys.modules["lime.lime_tabular"] = MagicMock()
sys.modules["alibi"] = MagicMock()
sys.modules["alibi.explainers"] = MagicMock()


# ====================== Data Fixtures ======================


@pytest.fixture
def sample_credit_data():
    """Create synthetic credit data for testing."""
    np.random.seed(42)
    n_samples = 500

    data = pd.DataFrame({
        "duration_credit": np.random.randint(6, 72, n_samples),
        "amount_credit": np.random.exponential(5000, n_samples),
        "effort_rate": np.random.uniform(0, 0.5, n_samples),
        "home_old": np.random.randint(0, 4, n_samples),
        "age": np.random.randint(18, 75, n_samples),
        "nb_credits": np.random.randint(1, 5, n_samples),
        "nb_of_dependants": np.random.randint(0, 5, n_samples),
    })

    # Create target with some correlation to features
    logits = (
        0.01 * (data["age"] - 40)
        + 0.0001 * data["amount_credit"]
        + 2 * data["effort_rate"]
        - 0.1 * data["nb_credits"]
        + np.random.randn(n_samples) * 0.5
    )
    probs = 1 / (1 + np.exp(-logits))
    data["presence_unpaid"] = (probs > 0.5).astype(int)

    return data


@pytest.fixture
def sample_feature_names():
    """List of feature names for credit data."""
    return [
        "duration_credit",
        "amount_credit",
        "effort_rate",
        "home_old",
        "age",
        "nb_credits",
        "nb_of_dependants",
    ]


@pytest.fixture
def sample_dataframe(sample_feature_names):
    """Create a generic sample DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({feat: np.random.randn(100) for feat in sample_feature_names})


@pytest.fixture
def sample_target_series():
    """Create a sample target series."""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100))


# ====================== Model Fixtures ======================


@pytest.fixture
def model_config():
    """Create a model configuration."""
    from src.glm_model import ModelConfig, ModelSelectionStrategy

    return ModelConfig(
        target_column="presence_unpaid",
        predictors=[
            "duration_credit",
            "amount_credit",
            "effort_rate",
            "age",
            "nb_credits",
        ],
        max_iterations=10,
        random_seed=42,
        test_size=0.2,
        min_predictors=2,
        max_predictors=4,
        selection_strategy=ModelSelectionStrategy.RANDOM,
    )


@pytest.fixture
def trained_model_selector(sample_credit_data, model_config):
    """Create a trained GLMModelSelector instance."""
    from src.glm_model import GLMModelSelector

    selector = GLMModelSelector(model_config)
    selector.prepare_data(sample_credit_data)
    selector.fit()
    return selector


@pytest.fixture
def saved_model_path(trained_model_selector, tmp_path):
    """Save a trained model and return the path."""
    model_path = tmp_path / "test_model.joblib"
    trained_model_selector.save_model(model_path)
    return model_path


@pytest.fixture
def model_serving(saved_model_path):
    """Create a ModelServing instance."""
    from src.glm_model import ModelServing

    return ModelServing(saved_model_path)


# ====================== Flask API Fixtures ======================


@pytest.fixture
def flask_app():
    """Create a Flask application for testing."""
    from api.app import app

    app.config["TESTING"] = True
    return app


@pytest.fixture
def flask_test_client(flask_app, saved_model_path, monkeypatch):
    """Create a Flask test client with a loaded model."""
    # Get the actual module (not the Flask app object imported into api namespace)
    api_module = sys.modules["api.app"]

    # Mock the model server
    mock_model_server = Mock()
    mock_model_server.predictors = [
        "duration_credit",
        "amount_credit",
        "effort_rate",
        "age",
        "nb_credits",
    ]
    mock_model_server.predict_single.return_value = {
        "probability": 0.75,
        "predicted_class": 1,
        "confidence": 0.75,
        "predictors_used": mock_model_server.predictors,
    }
    mock_model_server.predict_batch.return_value = pd.DataFrame({
        "predicted_probability": [0.7, 0.3, 0.8],
        "predicted_class": [1, 0, 1],
        "confidence": [0.7, 0.7, 0.8],
    })

    mock_selector = Mock()
    mock_best_model = Mock()
    mock_best_model.metrics = Mock()
    mock_best_model.metrics.to_dict.return_value = {
        "aic": 100.0,
        "bic": 110.0,
        "auc": 0.85,
        "accuracy": 0.80,
    }
    mock_best_model.metrics.accuracy = 0.80
    mock_selector.best_model = mock_best_model
    mock_model_server.selector = mock_selector

    # Get feature importance DataFrame
    mock_model_server.get_feature_importance.return_value = pd.DataFrame({
        "feature": ["age", "amount_credit", "effort_rate"],
        "coefficient": [0.1, 0.2, 0.3],
        "p_value": [0.01, 0.02, 0.001],
        "odds_ratio": [1.1, 1.2, 1.3],
        "significant": [True, True, True],
    })

    mock_model_info = {
        "path": str(saved_model_path),
        "version": "1.0.0",
        "loaded_at": datetime.now().isoformat(),
        "predictors": mock_model_server.predictors,
        "metrics": mock_best_model.metrics.to_dict(),
    }

    # Use monkeypatch to set module-level globals properly
    monkeypatch.setattr(api_module, "model_server", mock_model_server)
    monkeypatch.setattr(api_module, "model_info", mock_model_info)

    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def flask_test_client_no_model(flask_app, monkeypatch):
    """Create a Flask test client without a loaded model."""
    # Get the actual module (not the Flask app object imported into api namespace)
    api_module = sys.modules["api.app"]

    # Use monkeypatch to set module-level globals properly
    monkeypatch.setattr(api_module, "model_server", None)
    monkeypatch.setattr(api_module, "model_info", {})

    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def api_key():
    """Return a valid API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def auth_headers(api_key):
    """Return headers with API key authentication."""
    return {"X-API-Key": api_key}


# ====================== Mock Fixtures ======================


@pytest.fixture
def mock_redis():
    """Create a mocked Redis client."""
    mock_client = MagicMock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.setex.return_value = True
    mock_client.delete.return_value = 1
    return mock_client


@pytest.fixture
def mock_database_session():
    """Create a mocked SQLAlchemy session."""
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.rollback = MagicMock()
    mock_session.query = MagicMock()
    mock_session.close = MagicMock()

    # Context manager support
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    return mock_session


@pytest.fixture
def mock_mlflow():
    """Create a mocked MLflow module."""
    mock = MagicMock()
    mock.start_run.return_value.__enter__ = MagicMock()
    mock.start_run.return_value.__exit__ = MagicMock()
    mock.log_params = MagicMock()
    mock.log_metrics = MagicMock()
    mock.set_tags = MagicMock()
    mock.statsmodels.log_model = MagicMock()
    return mock


@pytest.fixture
def mock_boto3():
    """Create a mocked boto3 module."""
    mock = MagicMock()
    mock_cloudwatch = MagicMock()
    mock_cloudwatch.get_metric_statistics.return_value = {
        "Datapoints": [{"Average": 50.0, "Sum": 1000}]
    }
    mock.client.return_value = mock_cloudwatch
    return mock


# ====================== Resource Usage Fixtures ======================


@pytest.fixture
def sample_usage_data():
    """Create sample resource usage data for cost optimization tests."""
    from datetime import timedelta

    np.random.seed(42)
    data = []

    # Try to import ResourceUsage, fall back to a mock if not available
    try:
        from src.explainability import ResourceUsage
    except ImportError:
        # Create a simple mock dataclass
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class ResourceUsage:
            timestamp: datetime
            cpu_usage: float
            memory_usage: float
            storage_usage: float
            network_ingress: float
            network_egress: float
            gpu_usage: Optional[float] = None
            requests_count: int = 0

    for i in range(168):  # 1 week hourly data
        usage = ResourceUsage(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=168 - i),
            cpu_usage=50 + np.random.randn() * 20,
            memory_usage=4 + np.random.randn() * 1,
            storage_usage=100,
            network_ingress=np.random.exponential(1),
            network_egress=np.random.exponential(0.5),
            requests_count=int(np.random.poisson(1000)),
        )
        data.append(usage)

    return data


# ====================== Temporary Directory Fixtures ======================


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for data storage."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# ====================== Configuration Fixtures ======================


@pytest.fixture
def feature_engineering_config():
    """Create a feature engineering configuration."""
    return {
        "create_interactions": True,
        "max_interactions": 5,
        "feature_selection": True,
        "n_features": 10,
    }


@pytest.fixture
def pipeline_config():
    """Create a data pipeline configuration."""
    return {
        "validation_suite": "test_suite",
        "feature_types": {
            "duration_credit": "numeric",
            "amount_credit": "numeric",
            "effort_rate": "numeric",
            "age": "numeric",
            "nb_credits": "numeric",
        },
        "target_column": "presence_unpaid",
    }


# ====================== Mock Model Fixtures ======================


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1, 0])
    model.predict_proba.return_value = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.4, 0.6],
        [0.7, 0.3],
    ])
    return model


@pytest.fixture
def mock_statsmodels_result():
    """Create a mock statsmodels GLM result."""
    mock_result = MagicMock()
    mock_result.aic = 100.0
    mock_result.bic_llf = 110.0
    mock_result.llf = -50.0
    mock_result.predict.return_value = np.array([0.3, 0.7, 0.2, 0.8, 0.5])

    # Mock summary2
    mock_summary = MagicMock()
    mock_table = pd.DataFrame({
        "Coef.": [0.5, 0.1, 0.2, 0.3],
        "Std.Err.": [0.1, 0.05, 0.08, 0.12],
        "P>|z|": [0.001, 0.02, 0.03, 0.1],
    }, index=["Intercept", "feature1", "feature2", "feature3"])
    mock_summary.tables = [MagicMock(), mock_table]
    mock_result.summary2.return_value = mock_summary

    return mock_result
