"""
MLOps Pipeline with Model Versioning and A/B Testing
====================================================
Advanced MLOps features for production model management.
"""

from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, DateTime, Integer, JSON
from sqlalchemy import String as SAString
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import mlflow
import mlflow.statsmodels
from scipy import stats
import redis  # type: ignore[import-untyped]
from minio import Minio
import boto3

from src.glm_model import GLMModelSelector, ModelConfig, ModelResult

logger = logging.getLogger(__name__)

Base = declarative_base()


# ====================== Database Models ======================


class ModelVersion(Base):
    """Database model for tracking model versions."""

    __tablename__ = "model_versions"

    id = Column(SAString, primary_key=True)
    version = Column(SAString, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_path = Column(SAString)
    config = Column(JSON)
    metrics = Column(JSON)
    status = Column(SAString, default="inactive")  # active, inactive, testing, retired
    traffic_percentage = Column(Float, default=0.0)
    tags = Column(JSON)
    parent_version = Column(SAString)  # For tracking model lineage


class PredictionLog(Base):
    """Database model for logging predictions."""

    __tablename__ = "prediction_logs"

    id = Column(SAString, primary_key=True)
    model_version = Column(SAString)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    features = Column(JSON)
    prediction = Column(Float)
    predicted_class = Column(Integer)
    confidence = Column(Float)
    actual_outcome = Column(Integer, nullable=True)  # For feedback loop
    response_time_ms = Column(Float)


class ModelPerformance(Base):
    """Database model for tracking model performance metrics."""

    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(SAString)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    metric_name = Column(SAString)
    metric_value = Column(Float)
    window_size = Column(Integer)  # Hours


class ExperimentResult(Base):
    """Database model for A/B testing results."""

    __tablename__ = "experiment_results"

    id = Column(SAString, primary_key=True)
    experiment_name = Column(SAString)
    model_a_version = Column(SAString)
    model_b_version = Column(SAString)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    status = Column(SAString)  # running, completed, stopped
    winner = Column(SAString, nullable=True)
    statistical_significance = Column(Float, nullable=True)
    metrics = Column(JSON)


# ====================== Model Registry ======================


class ModelRegistry:
    """
    Centralized model registry for version management and deployment.
    """

    def __init__(
        self,
        database_url: str,
        storage_backend: str = "s3",
        storage_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model registry.

        Args:
            database_url: Database connection URL
            storage_backend: Storage backend type (s3, minio, local)
            storage_config: Storage configuration
        """
        # Database setup
        self.engine = create_engine(
            database_url, poolclass=QueuePool, pool_size=10, max_overflow=20
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Storage setup
        self.storage_backend = storage_backend
        self.storage_config = storage_config or {}
        self._setup_storage()

        # Cache setup
        self.redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            decode_responses=True,
        )

    def _setup_storage(self):
        """Setup storage backend."""
        if self.storage_backend == "s3":
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.storage_config.get("access_key"),
                aws_secret_access_key=self.storage_config.get("secret_key"),
                region_name=self.storage_config.get("region", "us-east-1"),
            )
            self.bucket_name = self.storage_config.get("bucket", "model-registry")

        elif self.storage_backend == "minio":
            self.minio_client = Minio(
                self.storage_config.get("endpoint", "localhost:9000"),
                access_key=self.storage_config.get("access_key"),
                secret_key=self.storage_config.get("secret_key"),
                secure=self.storage_config.get("secure", False),
            )
            self.bucket_name = self.storage_config.get("bucket", "model-registry")

        elif self.storage_backend == "local":
            self.storage_path = Path(self.storage_config.get("path", "./model_storage"))
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model: ModelResult,
        version: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
    ) -> str:
        """
        Register a new model version.

        Args:
            model: Trained model result
            version: Model version string
            tags: Optional tags for the model
            parent_version: Parent model version for lineage

        Returns:
            Model ID
        """
        # Generate model ID
        model_id = self._generate_model_id(version)

        # Save model to storage
        model_path = self._save_model_to_storage(model, model_id)

        # Register in database
        with self.SessionLocal() as session:
            model_version = ModelVersion(
                id=model_id,
                version=version,
                model_path=model_path,
                config=asdict(model.config) if model.config else None,
                metrics=model.metrics.to_dict(),
                tags=tags or {},
                parent_version=parent_version,
                status="inactive",
            )
            session.add(model_version)
            session.commit()

        # Log to MLflow
        if os.environ.get("MLFLOW_TRACKING_URI"):
            self._log_to_mlflow(model, version, tags or {})

        logger.info(f"Model registered: {model_id} (version: {version})")
        return model_id

    def _generate_model_id(self, version: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{version}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _save_model_to_storage(self, model: ModelResult, model_id: str) -> str:
        """Save model to storage backend."""
        import joblib

        if self.storage_backend == "local":
            model_path = self.storage_path / f"{model_id}.joblib"
            joblib.dump(model.model, model_path)
            return str(model_path)

        elif self.storage_backend in ["s3", "minio"]:
            # Serialize model
            import io

            buffer = io.BytesIO()
            joblib.dump(model.model, buffer)
            buffer.seek(0)

            object_name = f"models/{model_id}.joblib"

            if self.storage_backend == "s3":
                self.s3_client.put_object(
                    Bucket=self.bucket_name, Key=object_name, Body=buffer.getvalue()
                )
            else:
                self.minio_client.put_object(
                    self.bucket_name, object_name, buffer, length=len(buffer.getvalue())
                )

            return f"{self.bucket_name}/{object_name}"

        else:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")

    def _log_to_mlflow(self, model: ModelResult, version: str, tags: Dict[str, Any]):
        """Log model to MLflow."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(
                {
                    "model_version": version,
                    "num_predictors": len(model.predictors),
                    "formula": model.formula,
                }
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "aic": model.metrics.aic,
                    "bic": model.metrics.bic,
                    "auc": model.metrics.auc,
                    "accuracy": model.metrics.accuracy,
                    "f1_score": model.metrics.f1_score,
                }
            )

            # Log model
            mlflow.statsmodels.log_model(model.model, "model", registered_model_name="glm_model")

            # Set tags
            if tags:
                mlflow.set_tags(tags)

    def promote_model(
        self, model_id: str, status: str = "active", traffic_percentage: float = 100.0
    ):
        """
        Promote a model to production.

        Args:
            model_id: Model ID to promote
            status: New status (active, testing)
            traffic_percentage: Traffic percentage for gradual rollout
        """
        with self.SessionLocal() as session:
            # Deactivate current active models if promoting to 100%
            if status == "active" and traffic_percentage == 100.0:
                session.query(ModelVersion).filter(ModelVersion.status == "active").update(
                    {"status": "inactive", "traffic_percentage": 0.0}
                )

            # Update model status
            model = session.query(ModelVersion).filter(ModelVersion.id == model_id).first()

            if not model:
                raise ValueError(f"Model {model_id} not found")

            model.status = status
            model.traffic_percentage = traffic_percentage
            session.commit()

            # Clear cache
            self.redis_client.delete("active_models")

        logger.info(f"Model {model_id} promoted to {status} with {traffic_percentage}% traffic")

    def get_active_models(self) -> List[Dict[str, Any]]:
        """Get currently active models for serving."""
        # Check cache
        cached = self.redis_client.get("active_models")
        if cached:
            result: List[Dict[str, Any]] = json.loads(cached)
            return result

        with self.SessionLocal() as session:
            models = (
                session.query(ModelVersion)
                .filter(ModelVersion.status.in_(["active", "testing"]))
                .all()
            )

            result = []
            for model in models:
                result.append(
                    {
                        "id": model.id,
                        "version": model.version,
                        "model_path": model.model_path,
                        "traffic_percentage": model.traffic_percentage,
                        "status": model.status,
                    }
                )

        # Cache for 1 minute
        self.redis_client.setex("active_models", 60, json.dumps(result))

        return result


# ====================== A/B Testing Framework ======================


class ABTestingFramework:
    """
    Framework for conducting A/B tests between model versions.
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize A/B testing framework.

        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self.experiments: Dict[str, "Experiment"] = {}

    def create_experiment(
        self,
        name: str,
        model_a_id: str,
        model_b_id: str,
        traffic_split: Tuple[float, float] = (50.0, 50.0),
        min_sample_size: int = 1000,
        confidence_level: float = 0.95,
    ) -> "Experiment":
        """
        Create a new A/B testing experiment.

        Args:
            name: Experiment name
            model_a_id: Control model ID
            model_b_id: Treatment model ID
            traffic_split: Traffic split percentage (A, B)
            min_sample_size: Minimum sample size for significance
            confidence_level: Statistical confidence level

        Returns:
            Experiment instance
        """
        experiment = Experiment(
            name=name,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            traffic_split=traffic_split,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            registry=self.registry,
        )

        self.experiments[name] = experiment
        experiment.start()

        return experiment

    def get_experiment(self, name: str) -> Optional["Experiment"]:
        """Get experiment by name."""
        return self.experiments.get(name)

    def stop_experiment(self, name: str, promote_winner: bool = False):
        """
        Stop an experiment and optionally promote winner.

        Args:
            name: Experiment name
            promote_winner: Whether to promote winning model
        """
        experiment = self.experiments.get(name)
        if not experiment:
            raise ValueError(f"Experiment {name} not found")

        winner = experiment.analyze_results()
        experiment.stop()

        if promote_winner and winner:
            self.registry.promote_model(winner, status="active")

        return winner


@dataclass
class Experiment:
    """
    A/B testing experiment.
    """

    name: str
    model_a_id: str
    model_b_id: str
    traffic_split: Tuple[float, float]
    min_sample_size: int
    confidence_level: float
    registry: ModelRegistry

    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    status: str = "created"

    def start(self):
        """Start the experiment."""
        self.status = "running"

        # Set traffic split
        self.registry.promote_model(
            self.model_a_id, status="testing", traffic_percentage=self.traffic_split[0]
        )
        self.registry.promote_model(
            self.model_b_id, status="testing", traffic_percentage=self.traffic_split[1]
        )

        # Log experiment start
        with self.registry.SessionLocal() as session:
            experiment_result = ExperimentResult(
                id=hashlib.sha256(self.name.encode()).hexdigest()[:16],
                experiment_name=self.name,
                model_a_version=self.model_a_id,
                model_b_version=self.model_b_id,
                start_time=self.start_time,
                status="running",
            )
            session.add(experiment_result)
            session.commit()

        logger.info(f"Experiment {self.name} started")

    def stop(self):
        """Stop the experiment."""
        self.end_time = datetime.now(timezone.utc)
        self.status = "completed"

        # Reset traffic
        self.registry.promote_model(self.model_a_id, status="inactive", traffic_percentage=0.0)
        self.registry.promote_model(self.model_b_id, status="inactive", traffic_percentage=0.0)

        # Update experiment status
        with self.registry.SessionLocal() as session:
            experiment = (
                session.query(ExperimentResult)
                .filter(ExperimentResult.experiment_name == self.name)
                .first()
            )

            if experiment:
                experiment.end_time = self.end_time
                experiment.status = "completed"
                session.commit()

        logger.info(f"Experiment {self.name} stopped")

    def analyze_results(self) -> Optional[str]:
        """
        Analyze experiment results and determine winner.

        Returns:
            Winning model ID or None if no significant difference
        """
        with self.registry.SessionLocal() as session:
            # Get predictions for each model
            model_a_predictions = (
                session.query(PredictionLog)
                .filter(
                    PredictionLog.model_version == self.model_a_id,
                    PredictionLog.timestamp >= self.start_time,
                )
                .all()
            )

            model_b_predictions = (
                session.query(PredictionLog)
                .filter(
                    PredictionLog.model_version == self.model_b_id,
                    PredictionLog.timestamp >= self.start_time,
                )
                .all()
            )

        # Check sample size
        if (
            len(model_a_predictions) < self.min_sample_size
            or len(model_b_predictions) < self.min_sample_size
        ):
            logger.warning(f"Insufficient sample size for experiment {self.name}")
            return None

        # Calculate metrics
        metrics_a = self._calculate_metrics(model_a_predictions)
        metrics_b = self._calculate_metrics(model_b_predictions)

        # Perform statistical test
        winner = self._statistical_test(metrics_a, metrics_b)

        # Log results
        with self.registry.SessionLocal() as session:
            experiment = (
                session.query(ExperimentResult)
                .filter(ExperimentResult.experiment_name == self.name)
                .first()
            )

            if experiment:
                experiment.winner = winner
                experiment.metrics = {"model_a": metrics_a, "model_b": metrics_b}
                session.commit()

        return winner

    def _calculate_metrics(self, predictions: List[PredictionLog]) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not predictions:
            return {}

        # Calculate accuracy (assuming we have actual outcomes)
        correct = sum(
            1
            for p in predictions
            if p.actual_outcome is not None and p.predicted_class == p.actual_outcome
        )
        total = sum(1 for p in predictions if p.actual_outcome is not None)

        accuracy = correct / total if total > 0 else 0

        # Calculate average confidence
        avg_confidence = np.mean([p.confidence for p in predictions])

        # Calculate average response time
        avg_response_time = np.mean([p.response_time_ms for p in predictions])

        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_response_time": avg_response_time,
            "sample_size": len(predictions),
        }

    def _statistical_test(
        self, metrics_a: Dict[str, float], metrics_b: Dict[str, float]
    ) -> Optional[str]:
        """
        Perform statistical significance test.

        Returns:
            Winning model ID or None if no significant difference
        """
        # Use accuracy as primary metric
        accuracy_a = metrics_a.get("accuracy", 0)
        accuracy_b = metrics_b.get("accuracy", 0)
        n_a = metrics_a.get("sample_size", 0)
        n_b = metrics_b.get("sample_size", 0)

        if n_a == 0 or n_b == 0:
            return None

        # Perform two-proportion z-test
        p_pool = (accuracy_a * n_a + accuracy_b * n_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

        if se == 0:
            return None

        z_score = (accuracy_a - accuracy_b) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Check significance
        alpha = 1 - self.confidence_level
        if p_value < alpha:
            # Significant difference found
            if accuracy_a > accuracy_b:
                logger.info(f"Model A wins with accuracy {accuracy_a:.4f} vs {accuracy_b:.4f}")
                return self.model_a_id
            else:
                logger.info(f"Model B wins with accuracy {accuracy_b:.4f} vs {accuracy_a:.4f}")
                return self.model_b_id
        else:
            logger.info(f"No significant difference found (p-value: {p_value:.4f})")
            return None


# ====================== Performance Monitor ======================


class PerformanceMonitor:
    """
    Monitor model performance in production.
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize performance monitor.

        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def log_prediction(
        self,
        model_version: str,
        features: Dict[str, Any],
        prediction: float,
        predicted_class: int,
        confidence: float,
        response_time_ms: float,
    ):
        """
        Log a prediction asynchronously.

        Args:
            model_version: Model version used
            features: Input features
            prediction: Probability prediction
            predicted_class: Predicted class
            confidence: Prediction confidence
            response_time_ms: Response time in milliseconds
        """
        prediction_id = hashlib.sha256(
            f"{model_version}_{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        with self.registry.SessionLocal() as session:
            log_entry = PredictionLog(
                id=prediction_id,
                model_version=model_version,
                features=features,
                prediction=prediction,
                predicted_class=predicted_class,
                confidence=confidence,
                response_time_ms=response_time_ms,
            )
            session.add(log_entry)
            session.commit()

        # Update metrics asynchronously
        self.executor.submit(self._update_metrics, model_version)

    def _update_metrics(self, model_version: str):
        """Update performance metrics for a model."""
        with self.registry.SessionLocal() as session:
            # Calculate metrics for last hour
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

            predictions = (
                session.query(PredictionLog)
                .filter(
                    PredictionLog.model_version == model_version,
                    PredictionLog.timestamp >= one_hour_ago,
                )
                .all()
            )

            if not predictions:
                return

            # Calculate average response time
            avg_response_time = np.mean([p.response_time_ms for p in predictions])

            # Calculate average confidence
            avg_confidence = np.mean([p.confidence for p in predictions])

            # Store metrics
            metrics = [
                ModelPerformance(
                    model_version=model_version,
                    metric_name="avg_response_time",
                    metric_value=avg_response_time,
                    window_size=1,
                ),
                ModelPerformance(
                    model_version=model_version,
                    metric_name="avg_confidence",
                    metric_value=avg_confidence,
                    window_size=1,
                ),
                ModelPerformance(
                    model_version=model_version,
                    metric_name="predictions_per_hour",
                    metric_value=len(predictions),
                    window_size=1,
                ),
            ]

            session.bulk_save_objects(metrics)
            session.commit()

    def get_model_metrics(self, model_version: str, window_hours: int = 24) -> Dict[str, Any]:
        """
        Get performance metrics for a model.

        Args:
            model_version: Model version
            window_hours: Time window in hours

        Returns:
            Dictionary of metrics
        """
        with self.registry.SessionLocal() as session:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=window_hours)

            metrics = (
                session.query(ModelPerformance)
                .filter(
                    ModelPerformance.model_version == model_version,
                    ModelPerformance.timestamp >= cutoff_time,
                )
                .all()
            )

            # Aggregate metrics
            result: Dict[str, List[Dict[str, Any]]] = {}
            for metric in metrics:
                if metric.metric_name not in result:
                    result[metric.metric_name] = []
                result[metric.metric_name].append(
                    {"timestamp": metric.timestamp.isoformat(), "value": metric.metric_value}
                )

        return result

    def detect_drift(
        self, model_version: str, baseline_metrics: Dict[str, float], threshold: float = 0.1
    ) -> bool:
        """
        Detect model drift.

        Args:
            model_version: Model version
            baseline_metrics: Baseline metrics to compare
            threshold: Drift threshold

        Returns:
            True if drift detected
        """
        current_metrics = self.get_model_metrics(model_version, window_hours=1)

        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in current_metrics:
                current_values = [m["value"] for m in current_metrics[metric_name]]
                if current_values:
                    current_value = np.mean(current_values)

                    # Calculate relative change
                    if baseline_value != 0:
                        change = abs((current_value - baseline_value) / baseline_value)
                        if change > threshold:
                            logger.warning(
                                f"Drift detected for {metric_name}: "
                                f"{baseline_value:.4f} -> {current_value:.4f} "
                                f"({change:.2%} change)"
                            )
                            return True

        return False


# ====================== Model Trainer Pipeline ======================


class ModelTrainerPipeline:
    """
    Automated model training and deployment pipeline.
    """

    def __init__(
        self, registry: ModelRegistry, monitor: PerformanceMonitor, ab_framework: ABTestingFramework
    ):
        """
        Initialize training pipeline.

        Args:
            registry: Model registry
            monitor: Performance monitor
            ab_framework: A/B testing framework
        """
        self.registry = registry
        self.monitor = monitor
        self.ab_framework = ab_framework

    async def train_and_deploy(
        self,
        data: pd.DataFrame,
        config: ModelConfig,
        deployment_strategy: str = "blue_green",
        test_duration_hours: int = 24,
    ) -> str:
        """
        Train a new model and deploy with specified strategy.

        Args:
            data: Training data
            config: Model configuration
            deployment_strategy: Deployment strategy (blue_green, canary, ab_test)
            test_duration_hours: Testing duration for canary/AB deployments

        Returns:
            Deployed model ID
        """
        # Train model
        logger.info("Starting model training...")
        selector = GLMModelSelector(config)
        train_data, test_data = selector.prepare_data(data)
        best_model = selector.fit()

        # Generate version
        version = self._generate_version()

        # Get parent version (current active model)
        active_models = self.registry.get_active_models()
        parent_version = active_models[0]["id"] if active_models else None

        # Register model
        model_id = self.registry.register_model(
            best_model,
            version=version,
            tags={"deployment_strategy": deployment_strategy},
            parent_version=parent_version,
        )

        # Deploy based on strategy
        if deployment_strategy == "blue_green":
            await self._blue_green_deploy(model_id)

        elif deployment_strategy == "canary":
            await self._canary_deploy(model_id, test_duration_hours)

        elif deployment_strategy == "ab_test":
            await self._ab_test_deploy(model_id, parent_version, test_duration_hours)

        else:
            raise ValueError(f"Unknown deployment strategy: {deployment_strategy}")

        return model_id

    def _generate_version(self) -> str:
        """Generate semantic version number."""
        with self.registry.SessionLocal() as session:
            latest = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()

            if not latest:
                return "1.0.0"

            # Parse version and increment
            parts = latest.version.split(".")
            patch = int(parts[2]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"

    async def _blue_green_deploy(self, model_id: str):
        """Blue-green deployment strategy."""
        logger.info(f"Performing blue-green deployment for {model_id}")

        # Switch traffic immediately
        self.registry.promote_model(model_id, status="active", traffic_percentage=100.0)

        logger.info(f"Model {model_id} deployed successfully")

    async def _canary_deploy(self, model_id: str, test_duration_hours: int):
        """Canary deployment strategy."""
        logger.info(f"Starting canary deployment for {model_id}")

        # Start with 10% traffic
        traffic_percentages = [10, 25, 50, 75, 100]

        for percentage in traffic_percentages:
            self.registry.promote_model(
                model_id,
                status="testing" if percentage < 100 else "active",
                traffic_percentage=percentage,
            )

            logger.info(f"Traffic increased to {percentage}% for {model_id}")

            # Monitor for issues
            await asyncio.sleep(test_duration_hours * 3600 / len(traffic_percentages))

            # Check for drift or errors
            baseline_metrics = {"avg_confidence": 0.8, "avg_response_time": 50}
            if self.monitor.detect_drift(model_id, baseline_metrics):
                logger.error(f"Drift detected during canary deployment of {model_id}")
                # Rollback
                self.registry.promote_model(model_id, status="inactive", traffic_percentage=0.0)
                raise RuntimeError("Canary deployment failed due to drift")

        logger.info(f"Canary deployment completed for {model_id}")

    async def _ab_test_deploy(self, model_id: str, parent_version: str, test_duration_hours: int):
        """A/B testing deployment strategy."""
        if not parent_version:
            # No existing model to compare against
            await self._blue_green_deploy(model_id)
            return

        logger.info(f"Starting A/B test between {parent_version} and {model_id}")

        # Create experiment
        experiment = self.ab_framework.create_experiment(
            name=f"deploy_{model_id}",
            model_a_id=parent_version,
            model_b_id=model_id,
            traffic_split=(50.0, 50.0),
            min_sample_size=1000,
        )

        # Wait for test duration
        await asyncio.sleep(test_duration_hours * 3600)

        # Analyze results and promote winner
        winner = self.ab_framework.stop_experiment(experiment.name, promote_winner=True)

        logger.info(f"A/B test completed. Winner: {winner}")


# ====================== Example Usage ======================


async def main():
    """Example usage of MLOps pipeline."""

    # Initialize components (use environment variables in production)
    registry = ModelRegistry(
        database_url=os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/mlops"),
        storage_backend="s3",
        storage_config={
            "bucket": os.environ.get("S3_BUCKET", "model-registry"),
            "access_key": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "region": os.environ.get("AWS_REGION", "us-east-1"),
        },
    )

    monitor = PerformanceMonitor(registry)
    ab_framework = ABTestingFramework(registry)
    pipeline = ModelTrainerPipeline(registry, monitor, ab_framework)

    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "feature1": np.random.randn(1000),
            "feature2": np.random.randn(1000),
            "feature3": np.random.randn(1000),
            "presence_unpaid": np.random.randint(0, 2, 1000),
        }
    )

    # Configure model
    config = ModelConfig(
        target_column="presence_unpaid",
        predictors=["feature1", "feature2", "feature3"],
        max_iterations=50,
    )

    # Train and deploy with canary strategy
    model_id = await pipeline.train_and_deploy(
        data=data, config=config, deployment_strategy="canary", test_duration_hours=1
    )

    print(f"Model deployed: {model_id}")

    # Simulate predictions and monitoring
    for _ in range(100):
        await monitor.log_prediction(
            model_version=model_id,
            features={"feature1": 0.5, "feature2": -0.3, "feature3": 1.2},
            prediction=0.75,
            predicted_class=1,
            confidence=0.85,
            response_time_ms=25.5,
        )

    # Get metrics
    metrics = monitor.get_model_metrics(model_id, window_hours=1)
    print(f"Model metrics: {metrics}")


if __name__ == "__main__":
    # Set up MLflow
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

    # Run async main
    asyncio.run(main())
