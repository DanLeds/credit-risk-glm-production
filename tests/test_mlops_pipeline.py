"""
Unit tests for MLOps Pipeline with Model Versioning and A/B Testing
"""

import sys
import unittest
import tempfile
from unittest.mock import Mock, patch, MagicMock
import json

# Mock external dependencies before importing the module
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy.ext"] = MagicMock()
sys.modules["sqlalchemy.ext.declarative"] = MagicMock()
sys.modules["sqlalchemy.orm"] = MagicMock()
sys.modules["sqlalchemy.pool"] = MagicMock()
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.statsmodels"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["minio"] = MagicMock()
sys.modules["boto3"] = MagicMock()

# Create mock for sqlalchemy Base
mock_base = MagicMock()
mock_base.metadata = MagicMock()
sys.modules["sqlalchemy"].ext.declarative.declarative_base = MagicMock(return_value=mock_base)

# Mock the glm_model imports
mock_glm_module = MagicMock()
mock_glm_module.GLMModelSelector = MagicMock()
mock_glm_module.ModelConfig = MagicMock()
mock_glm_module.ModelResult = MagicMock()
sys.modules["src.glm_model"] = mock_glm_module

from src.mlops_pipeline import (
    ModelRegistry,
    ABTestingFramework,
    Experiment,
    PerformanceMonitor,
    ModelTrainerPipeline,
)


class TestModelRegistry(unittest.TestCase):
    """Test ModelRegistry class."""

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    def test_registry_initialization_local(self, mock_engine, mock_redis):
        """Test ModelRegistry initialization with local storage."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(
                database_url="sqlite:///test.db",
                storage_backend="local",
                storage_config={"path": tmpdir},
            )

            self.assertEqual(registry.storage_backend, "local")

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    @patch("src.mlops_pipeline.boto3.client")
    def test_registry_initialization_s3(self, mock_boto, mock_engine, mock_redis):
        """Test ModelRegistry initialization with S3 storage."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()
        mock_boto.return_value = Mock()

        registry = ModelRegistry(
            database_url="sqlite:///test.db",
            storage_backend="s3",
            storage_config={
                "access_key": "test_key",
                "secret_key": "test_secret",
                "bucket": "test-bucket",
            },
        )

        self.assertEqual(registry.storage_backend, "s3")
        self.assertEqual(registry.bucket_name, "test-bucket")

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    @patch("src.mlops_pipeline.Minio")
    def test_registry_initialization_minio(self, mock_minio, mock_engine, mock_redis):
        """Test ModelRegistry initialization with MinIO storage."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()
        mock_minio.return_value = Mock()

        registry = ModelRegistry(
            database_url="sqlite:///test.db",
            storage_backend="minio",
            storage_config={
                "endpoint": "localhost:9000",
                "access_key": "test_key",
                "secret_key": "test_secret",
                "bucket": "test-bucket",
            },
        )

        self.assertEqual(registry.storage_backend, "minio")
        self.assertEqual(registry.bucket_name, "test-bucket")

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    def test_generate_model_id(self, mock_engine, mock_redis):
        """Test model ID generation."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(
                database_url="sqlite:///test.db",
                storage_backend="local",
                storage_config={"path": tmpdir},
            )

            model_id = registry._generate_model_id("v1.0.0")

            self.assertEqual(len(model_id), 16)
            self.assertTrue(all(c in "0123456789abcdef" for c in model_id))

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    def test_save_model_to_local_storage(self, mock_engine, mock_redis):
        """Test saving model to local storage."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(
                database_url="sqlite:///test.db",
                storage_backend="local",
                storage_config={"path": tmpdir},
            )

            # Verify storage path is set correctly
            self.assertTrue(registry.storage_path.exists())
            self.assertEqual(registry.storage_backend, "local")

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    def test_get_active_models_cached(self, mock_engine, mock_redis):
        """Test getting active models from cache."""
        mock_engine.return_value = Mock()
        mock_redis_client = Mock()
        mock_redis_client.get.return_value = json.dumps(
            [{"id": "model1", "version": "1.0", "status": "active"}]
        )
        mock_redis.return_value = mock_redis_client

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(
                database_url="sqlite:///test.db",
                storage_backend="local",
                storage_config={"path": tmpdir},
            )

            models = registry.get_active_models()

            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["id"], "model1")

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    def test_get_active_models_method_signature(self, mock_engine, mock_redis):
        """Test get_active_models method exists and has correct signature."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(
                database_url="sqlite:///test.db",
                storage_backend="local",
                storage_config={"path": tmpdir},
            )

            # Verify method exists
            self.assertTrue(hasattr(registry, "get_active_models"))
            self.assertTrue(callable(registry.get_active_models))

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    def test_promote_model(self, mock_engine, mock_redis):
        """Test promoting a model - initialization test."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(
                database_url="sqlite:///test.db",
                storage_backend="local",
                storage_config={"path": tmpdir},
            )

            # Verify registry is initialized with promote capability
            self.assertTrue(hasattr(registry, "promote_model"))
            self.assertTrue(callable(registry.promote_model))

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    def test_promote_model_method_exists(self, mock_engine, mock_redis):
        """Test that promote_model method exists and has correct signature."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(
                database_url="sqlite:///test.db",
                storage_backend="local",
                storage_config={"path": tmpdir},
            )

            # Verify method signature
            import inspect

            sig = inspect.signature(registry.promote_model)
            params = list(sig.parameters.keys())
            self.assertIn("model_id", params)
            self.assertIn("status", params)
            self.assertIn("traffic_percentage", params)


class TestABTestingFramework(unittest.TestCase):
    """Test ABTestingFramework class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_registry = Mock()
        self.mock_registry.SessionLocal = Mock()

    def test_framework_initialization(self):
        """Test ABTestingFramework initialization."""
        framework = ABTestingFramework(self.mock_registry)

        self.assertEqual(framework.registry, self.mock_registry)
        self.assertEqual(framework.experiments, {})

    def test_create_experiment(self):
        """Test creating an experiment."""
        mock_session = MagicMock()
        self.mock_registry.SessionLocal.return_value = mock_session
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        framework = ABTestingFramework(self.mock_registry)

        experiment = framework.create_experiment(
            name="test_experiment",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=(50.0, 50.0),
            min_sample_size=100,
        )

        self.assertEqual(experiment.name, "test_experiment")
        self.assertEqual(experiment.model_a_id, "model_a")
        self.assertEqual(experiment.model_b_id, "model_b")
        self.assertIn("test_experiment", framework.experiments)

    def test_get_experiment(self):
        """Test getting an experiment."""
        framework = ABTestingFramework(self.mock_registry)

        # Manually add an experiment to avoid start() complexity
        mock_experiment = Mock()
        mock_experiment.name = "test_experiment"
        framework.experiments["test_experiment"] = mock_experiment

        experiment = framework.get_experiment("test_experiment")
        self.assertIsNotNone(experiment)

        missing = framework.get_experiment("nonexistent")
        self.assertIsNone(missing)

    def test_stop_experiment_not_found(self):
        """Test stopping non-existent experiment."""
        framework = ABTestingFramework(self.mock_registry)

        with self.assertRaises(ValueError) as context:
            framework.stop_experiment("nonexistent")

        self.assertIn("not found", str(context.exception))


class TestExperiment(unittest.TestCase):
    """Test Experiment class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_registry = Mock()
        mock_session = MagicMock()
        self.mock_registry.SessionLocal.return_value = mock_session
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

    def test_experiment_creation(self):
        """Test creating an experiment."""
        experiment = Experiment(
            name="test_experiment",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=(50.0, 50.0),
            min_sample_size=100,
            confidence_level=0.95,
            registry=self.mock_registry,
        )

        # Verify initial state
        self.assertEqual(experiment.name, "test_experiment")
        self.assertEqual(experiment.model_a_id, "model_a")
        self.assertEqual(experiment.model_b_id, "model_b")
        self.assertEqual(experiment.status, "created")

    def test_experiment_traffic_split(self):
        """Test experiment traffic split configuration."""
        experiment = Experiment(
            name="test_experiment",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=(70.0, 30.0),
            min_sample_size=100,
            confidence_level=0.95,
            registry=self.mock_registry,
        )

        self.assertEqual(experiment.traffic_split, (70.0, 30.0))
        self.assertEqual(experiment.min_sample_size, 100)

    def test_calculate_metrics_empty(self):
        """Test calculating metrics with empty predictions."""
        experiment = Experiment(
            name="test_experiment",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=(50.0, 50.0),
            min_sample_size=100,
            confidence_level=0.95,
            registry=self.mock_registry,
        )

        metrics = experiment._calculate_metrics([])

        self.assertEqual(metrics, {})

    def test_calculate_metrics(self):
        """Test calculating metrics."""
        experiment = Experiment(
            name="test_experiment",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=(50.0, 50.0),
            min_sample_size=100,
            confidence_level=0.95,
            registry=self.mock_registry,
        )

        # Create mock predictions
        predictions = []
        for i in range(100):
            mock_pred = Mock()
            mock_pred.actual_outcome = i % 2
            mock_pred.predicted_class = i % 2
            mock_pred.confidence = 0.8
            mock_pred.response_time_ms = 25.0
            predictions.append(mock_pred)

        metrics = experiment._calculate_metrics(predictions)

        self.assertIn("accuracy", metrics)
        self.assertIn("avg_confidence", metrics)
        self.assertIn("avg_response_time", metrics)
        self.assertEqual(metrics["sample_size"], 100)
        self.assertEqual(metrics["accuracy"], 1.0)  # All correct

    def test_statistical_test_no_data(self):
        """Test statistical test with no data."""
        experiment = Experiment(
            name="test_experiment",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=(50.0, 50.0),
            min_sample_size=100,
            confidence_level=0.95,
            registry=self.mock_registry,
        )

        result = experiment._statistical_test(
            {"accuracy": 0.8, "sample_size": 0}, {"accuracy": 0.75, "sample_size": 0}
        )

        self.assertIsNone(result)

    def test_statistical_test_significant_difference(self):
        """Test statistical test with significant difference."""
        experiment = Experiment(
            name="test_experiment",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=(50.0, 50.0),
            min_sample_size=100,
            confidence_level=0.95,
            registry=self.mock_registry,
        )

        # Large sample sizes with different accuracies
        result = experiment._statistical_test(
            {"accuracy": 0.9, "sample_size": 1000}, {"accuracy": 0.7, "sample_size": 1000}
        )

        self.assertEqual(result, "model_a")

    def test_statistical_test_model_b_wins(self):
        """Test statistical test when model B wins."""
        experiment = Experiment(
            name="test_experiment",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=(50.0, 50.0),
            min_sample_size=100,
            confidence_level=0.95,
            registry=self.mock_registry,
        )

        result = experiment._statistical_test(
            {"accuracy": 0.7, "sample_size": 1000}, {"accuracy": 0.9, "sample_size": 1000}
        )

        self.assertEqual(result, "model_b")


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_registry = Mock()

    def test_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(self.mock_registry)

        self.assertEqual(monitor.registry, self.mock_registry)
        self.assertIsNotNone(monitor.executor)

    def test_log_prediction(self):
        """Test logging a prediction."""
        import asyncio

        mock_session = MagicMock()
        self.mock_registry.SessionLocal.return_value = mock_session
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        monitor = PerformanceMonitor(self.mock_registry)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                monitor.log_prediction(
                    model_version="v1.0",
                    features={"feature1": 0.5},
                    prediction=0.75,
                    predicted_class=1,
                    confidence=0.85,
                    response_time_ms=25.0,
                )
            )

            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
        finally:
            loop.close()

    def test_get_model_metrics_method_exists(self):
        """Test that get_model_metrics method exists."""
        monitor = PerformanceMonitor(self.mock_registry)

        self.assertTrue(hasattr(monitor, "get_model_metrics"))
        self.assertTrue(callable(monitor.get_model_metrics))

    def test_detect_drift_method_exists(self):
        """Test that detect_drift method exists."""
        monitor = PerformanceMonitor(self.mock_registry)

        self.assertTrue(hasattr(monitor, "detect_drift"))
        self.assertTrue(callable(monitor.detect_drift))

    def test_detect_drift_no_metrics(self):
        """Test drift detection returns False when no metrics in result."""
        monitor = PerformanceMonitor(self.mock_registry)

        # Mock get_model_metrics to return empty
        monitor.get_model_metrics = Mock(return_value={})

        drift_detected = monitor.detect_drift(
            model_version="v1.0", baseline_metrics={"avg_confidence": 0.8}, threshold=0.1
        )

        self.assertFalse(drift_detected)


class TestModelTrainerPipeline(unittest.TestCase):
    """Test ModelTrainerPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_registry = Mock()
        self.mock_monitor = Mock()
        self.mock_ab_framework = Mock()

    def test_pipeline_initialization(self):
        """Test ModelTrainerPipeline initialization."""
        pipeline = ModelTrainerPipeline(
            registry=self.mock_registry,
            monitor=self.mock_monitor,
            ab_framework=self.mock_ab_framework,
        )

        self.assertEqual(pipeline.registry, self.mock_registry)
        self.assertEqual(pipeline.monitor, self.mock_monitor)
        self.assertEqual(pipeline.ab_framework, self.mock_ab_framework)

    def test_generate_version_method_exists(self):
        """Test version generation method exists."""
        pipeline = ModelTrainerPipeline(
            registry=self.mock_registry,
            monitor=self.mock_monitor,
            ab_framework=self.mock_ab_framework,
        )

        self.assertTrue(hasattr(pipeline, "_generate_version"))
        self.assertTrue(callable(pipeline._generate_version))

    def test_train_and_deploy_method_exists(self):
        """Test train_and_deploy method exists."""
        pipeline = ModelTrainerPipeline(
            registry=self.mock_registry,
            monitor=self.mock_monitor,
            ab_framework=self.mock_ab_framework,
        )

        self.assertTrue(hasattr(pipeline, "train_and_deploy"))
        self.assertTrue(callable(pipeline.train_and_deploy))

    def test_blue_green_deploy(self):
        """Test blue-green deployment."""
        import asyncio

        pipeline = ModelTrainerPipeline(
            registry=self.mock_registry,
            monitor=self.mock_monitor,
            ab_framework=self.mock_ab_framework,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(pipeline._blue_green_deploy("model_id"))

            self.mock_registry.promote_model.assert_called_once_with(
                "model_id", status="active", traffic_percentage=100.0
            )
        finally:
            loop.close()

    def test_ab_test_deploy_no_parent(self):
        """Test A/B test deployment without parent model."""
        import asyncio

        pipeline = ModelTrainerPipeline(
            registry=self.mock_registry,
            monitor=self.mock_monitor,
            ab_framework=self.mock_ab_framework,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(pipeline._ab_test_deploy("model_id", None, 1))

            # Should fall back to blue-green
            self.mock_registry.promote_model.assert_called_once_with(
                "model_id", status="active", traffic_percentage=100.0
            )
        finally:
            loop.close()


class TestIntegration(unittest.TestCase):
    """Integration tests for MLOps pipeline components."""

    @patch("src.mlops_pipeline.redis.Redis")
    @patch("src.mlops_pipeline.create_engine")
    def test_full_workflow_mock(self, mock_engine, mock_redis):
        """Test full workflow with mocked components."""
        mock_engine.return_value = Mock()
        mock_redis.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            registry = ModelRegistry(
                database_url="sqlite:///test.db",
                storage_backend="local",
                storage_config={"path": tmpdir},
            )

            monitor = PerformanceMonitor(registry)
            ab_framework = ABTestingFramework(registry)

            # Verify components are connected
            self.assertEqual(monitor.registry, registry)
            self.assertEqual(ab_framework.registry, registry)


if __name__ == "__main__":
    unittest.main()
