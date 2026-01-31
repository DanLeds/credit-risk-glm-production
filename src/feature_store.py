"""
Feature Store and Data Validation Pipeline
==========================================
Production-grade feature store with data quality validation.
"""

import os
import hashlib
import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import feast
from feast import Entity, FeatureView, FileSource, Field, ValueType
from feast.types import Float32, Float64, Int64, Bool, String
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import redis  # type: ignore[import-untyped]
from sqlalchemy import create_engine, Column, Float, DateTime, Integer
from sqlalchemy import String as SAString, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


# ====================== Data Quality Models ======================


class DataQualityIssue(Base):
    """Database model for tracking data quality issues."""

    __tablename__ = "data_quality_issues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    dataset_name = Column(SAString)
    issue_type = Column(SAString)
    severity = Column(SAString)  # critical, warning, info
    affected_columns = Column(JSON)
    details = Column(JSON)
    resolved = Column(SAString, default="false")


class FeatureStatistics(Base):
    """Database model for feature statistics tracking."""

    __tablename__ = "feature_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    feature_name = Column(SAString)
    mean = Column(Float)
    std = Column(Float)
    min_val = Column(Float)
    max_val = Column(Float)
    median = Column(Float)
    missing_count = Column(Integer)
    unique_count = Column(Integer)


# ====================== Data Validation ======================


@dataclass
class ValidationRule:
    """Data validation rule definition."""

    name: str
    column: str
    rule_type: str  # range, regex, unique, not_null, etc.
    parameters: Dict[str, Any]
    severity: str = "warning"  # info, warning, critical


class DataValidator:
    """
    Comprehensive data validation system.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data validator.

        Args:
            config_path: Path to validation configuration
        """
        self.config_path = config_path
        self.expectations: Dict[str, Any] = {}
        self.validation_results: List[Any] = []
        self.ge_context = ge.get_context()

        # Initialize database
        self.engine = create_engine(os.environ.get("DATABASE_URL", "sqlite:///data_quality.db"))
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_expectation_suite(self, suite_name: str, dataset: pd.DataFrame) -> ExpectationSuite:
        """
        Create Great Expectations suite for dataset.

        Args:
            suite_name: Name of the expectation suite
            dataset: Sample dataset for profiling

        Returns:
            ExpectationSuite object
        """
        suite = self.ge_context.create_expectation_suite(
            expectation_suite_name=suite_name, overwrite_existing=True
        )

        # Profile the dataset
        _ = ge.dataset.PandasDataset(dataset)  # Used for profiling context

        # Add expectations for each column
        for column in dataset.columns:
            dtype = dataset[column].dtype

            # Basic expectations
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist", kwargs={"column": column}
                )
            )

            # Numeric columns
            if np.issubdtype(dtype, np.number):
                # Range expectations
                min_val = float(dataset[column].min())
                max_val = float(dataset[column].max())
                mean_val = float(dataset[column].mean())
                std_val = float(dataset[column].std())

                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_between",
                        kwargs={
                            "column": column,
                            "min_value": min_val - 3 * std_val,
                            "max_value": max_val + 3 * std_val,
                            "mostly": 0.99,
                        },
                    )
                )

                # Check for outliers
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_mean_to_be_between",
                        kwargs={
                            "column": column,
                            "min_value": mean_val - 2 * std_val,
                            "max_value": mean_val + 2 * std_val,
                        },
                    )
                )

            # Categorical columns
            elif dtype == "object":
                unique_values = dataset[column].unique().tolist()

                if len(unique_values) < 100:  # Only for low cardinality
                    suite.add_expectation(
                        ExpectationConfiguration(
                            expectation_type="expect_column_values_to_be_in_set",
                            kwargs={"column": column, "value_set": unique_values, "mostly": 0.95},
                        )
                    )

        self.expectations[suite_name] = suite
        return suite

    def validate_data(
        self, data: pd.DataFrame, suite_name: str, save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Validate data against expectation suite.

        Args:
            data: Data to validate
            suite_name: Name of expectation suite to use
            save_results: Whether to save validation results

        Returns:
            Validation results dictionary
        """
        if suite_name not in self.expectations:
            raise ValueError(f"Expectation suite '{suite_name}' not found")

        # Create batch request
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name="runtime_data",
            runtime_parameters={"batch_data": data},
            batch_identifiers={"default_identifier_name": "default_identifier"},
        )

        # Run validation
        results = self.ge_context.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[batch_request],
            run_id=f"{suite_name}_{datetime.now(timezone.utc).isoformat()}",
        )

        # Process results
        validation_summary = {
            "success": results.success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suite_name": suite_name,
            "statistics": {
                "evaluated_expectations": results.statistics.evaluated_expectations,
                "successful_expectations": results.statistics.successful_expectations,
                "unsuccessful_expectations": results.statistics.unsuccessful_expectations,
                "success_percent": results.statistics.success_percent,
            },
            "failed_expectations": [],
        }

        # Extract failed expectations
        for result in results.results:
            if not result.success:
                validation_summary["failed_expectations"].append(
                    {
                        "expectation_type": result.expectation_config.expectation_type,
                        "kwargs": result.expectation_config.kwargs,
                        "result": result.result,
                    }
                )

        # Save to database if requested
        if save_results:
            self._save_validation_results(validation_summary)

        return validation_summary

    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to database."""
        with self.SessionLocal() as session:
            for failed in results.get("failed_expectations", []):
                issue = DataQualityIssue(
                    dataset_name=results["suite_name"],
                    issue_type=failed["expectation_type"],
                    severity="critical"
                    if results["statistics"]["success_percent"] < 80
                    else "warning",
                    affected_columns=[failed["kwargs"].get("column", "unknown")],
                    details=failed,
                )
                session.add(issue)
            session.commit()

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None,
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.

        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            column_mapping: Column mapping for Evidently

        Returns:
            Drift detection results
        """
        # Create drift report
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])

        report.run(
            reference_data=reference_data, current_data=current_data, column_mapping=column_mapping
        )

        # Get report results
        report_dict = report.as_dict()

        # Extract key metrics
        drift_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_drift": report_dict["metrics"][0]["result"]["dataset_drift"],
            "share_of_drifted_columns": report_dict["metrics"][0]["result"][
                "share_of_drifted_columns"
            ],
            "drifted_columns": [],
        }

        # Find drifted columns
        for column_name, column_drift in report_dict["metrics"][0]["result"][
            "drift_by_columns"
        ].items():
            if column_drift["drift_detected"]:
                drift_summary["drifted_columns"].append(
                    {
                        "column": column_name,
                        "drift_score": column_drift.get("drift_score", None),
                        "stattest": column_drift.get("stattest_name", None),
                        "p_value": column_drift.get("p_value", None),
                    }
                )

        # Save drift detection results
        if drift_summary["dataset_drift"]:
            self._log_drift_issue(drift_summary)

        return drift_summary

    def _log_drift_issue(self, drift_summary: Dict[str, Any]):
        """Log drift issue to database."""
        with self.SessionLocal() as session:
            issue = DataQualityIssue(
                dataset_name="production_data",
                issue_type="data_drift",
                severity="critical"
                if drift_summary["share_of_drifted_columns"] > 0.5
                else "warning",
                affected_columns=[col["column"] for col in drift_summary["drifted_columns"]],
                details=drift_summary,
            )
            session.add(issue)
            session.commit()


# ====================== Feature Store ======================


class FeatureStore:
    """
    Production feature store with versioning and monitoring.
    """

    def __init__(self, repo_path: str, redis_host: str = "localhost", redis_port: int = 6379):
        """
        Initialize feature store.

        Args:
            repo_path: Path to Feast repository
            redis_host: Redis host for online store
            redis_port: Redis port
        """
        self.repo_path = repo_path
        self.store = feast.FeatureStore(repo_path=repo_path)

        # Redis for caching
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Feature statistics tracking
        self.engine = create_engine(os.environ.get("DATABASE_URL", "sqlite:///feature_store.db"))
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def register_features(self, features: List[Dict[str, Any]], entity_name: str, source_path: str):
        """
        Register new features in the feature store.

        Args:
            features: List of feature definitions
            entity_name: Entity name for features
            source_path: Path to feature data source
        """
        # Create entity
        entity = Entity(
            name=entity_name, value_type=ValueType.INT64, description=f"Entity for {entity_name}"
        )

        # Create feature view
        feature_fields = []
        for feature in features:
            feat_field = Field(name=feature["name"], dtype=self._get_feast_dtype(feature["dtype"]))
            feature_fields.append(feat_field)

        # Create source
        source = FileSource(path=source_path, timestamp_field="event_timestamp")

        # Create feature view
        feature_view = FeatureView(
            name=f"{entity_name}_features",
            entities=[entity.name],
            ttl=timedelta(days=30),
            features=feature_fields,
            source=source,
            tags={"version": "1.0", "owner": "ml-team"},
        )

        # Apply to feature store
        self.store.apply([entity, feature_view])

        logger.info(f"Registered {len(features)} features for entity {entity_name}")

    def _get_feast_dtype(self, dtype_str: str):
        """Convert string dtype to Feast dtype."""
        dtype_map = {
            "float32": Float32,
            "float64": Float64,
            "int64": Int64,
            "string": String,
            "bool": Bool,
        }
        return dtype_map.get(dtype_str, Float64)

    def get_online_features(
        self, entity_ids: List[int], feature_names: List[str], cache_ttl: int = 60
    ) -> pd.DataFrame:
        """
        Get features from online store with caching.

        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names
            cache_ttl: Cache TTL in seconds

        Returns:
            DataFrame with features
        """
        # Check cache (MD5 used for cache key only, not security)
        cache_key = f"features:{hashlib.md5(str(entity_ids + feature_names).encode(), usedforsecurity=False).hexdigest()}"
        cached = self.redis_client.get(cache_key)

        if cached:
            return pd.read_json(cached)

        # Get from feature store
        feature_vector = self.store.get_online_features(
            features=feature_names, entity_rows=[{"entity_id": eid} for eid in entity_ids]
        ).to_df()

        # Cache results
        self.redis_client.setex(cache_key, cache_ttl, feature_vector.to_json())

        return feature_vector

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
        timestamp_column: str = "event_timestamp",
    ) -> pd.DataFrame:
        """
        Get historical features for training.

        Args:
            entity_df: DataFrame with entity IDs and timestamps
            features: List of feature names
            timestamp_column: Name of timestamp column

        Returns:
            DataFrame with historical features
        """
        training_df = self.store.get_historical_features(
            entity_df=entity_df, features=features
        ).to_df()

        # Track feature statistics
        self._track_feature_statistics(training_df, features)

        return training_df

    def _track_feature_statistics(self, df: pd.DataFrame, features: List[str]):
        """Track statistics for features."""
        with self.SessionLocal() as session:
            for feature in features:
                if feature in df.columns:
                    stats = FeatureStatistics(
                        feature_name=feature,
                        mean=float(df[feature].mean())
                        if pd.api.types.is_numeric_dtype(df[feature])
                        else 0,
                        std=float(df[feature].std())
                        if pd.api.types.is_numeric_dtype(df[feature])
                        else 0,
                        min_val=float(df[feature].min())
                        if pd.api.types.is_numeric_dtype(df[feature])
                        else 0,
                        max_val=float(df[feature].max())
                        if pd.api.types.is_numeric_dtype(df[feature])
                        else 0,
                        median=float(df[feature].median())
                        if pd.api.types.is_numeric_dtype(df[feature])
                        else 0,
                        missing_count=int(df[feature].isna().sum()),
                        unique_count=int(df[feature].nunique()),
                    )
                    session.add(stats)
            session.commit()

    def materialize_features(self, start_date: datetime, end_date: datetime):
        """
        Materialize features to online store.

        Args:
            start_date: Start date for materialization
            end_date: End date for materialization
        """
        self.store.materialize_incremental(start_date=start_date, end_date=end_date)

        logger.info(f"Materialized features from {start_date} to {end_date}")


# ====================== Feature Engineering Pipeline ======================


class FeatureEngineeringPipeline:
    """
    Automated feature engineering pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineering pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.transformers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}

    def create_features(
        self, df: pd.DataFrame, target_column: str, feature_types: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Create engineered features.

        Args:
            df: Input DataFrame
            target_column: Target column name
            feature_types: Dictionary mapping columns to types

        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()

        # Numeric features
        numeric_cols = [col for col, dtype in feature_types.items() if dtype == "numeric"]
        if numeric_cols:
            df_engineered = self._create_numeric_features(df_engineered, numeric_cols)

        # Categorical features
        categorical_cols = [col for col, dtype in feature_types.items() if dtype == "categorical"]
        if categorical_cols:
            df_engineered = self._create_categorical_features(df_engineered, categorical_cols)

        # Date features
        date_cols = [col for col, dtype in feature_types.items() if dtype == "datetime"]
        if date_cols:
            df_engineered = self._create_date_features(df_engineered, date_cols)

        # Interaction features
        if self.config.get("create_interactions", False):
            df_engineered = self._create_interaction_features(
                df_engineered, numeric_cols, max_interactions=self.config.get("max_interactions", 5)
            )

        # Feature selection
        if self.config.get("feature_selection", False):
            df_engineered = self._select_features(
                df_engineered, target_column, k=self.config.get("n_features", 20)
            )

        return df_engineered

    def _create_numeric_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create features from numeric columns."""
        # Log transformations
        for col in numeric_cols:
            if (df[col] > 0).all():
                df[f"{col}_log"] = np.log1p(df[col])

        # Polynomial features
        if len(numeric_cols) > 1:
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(df[numeric_cols])
            poly_names = poly.get_feature_names_out(numeric_cols)

            # Add only interaction terms
            for i, name in enumerate(poly_names):
                if " " in name:  # Interaction term
                    df[name.replace(" ", "_x_")] = poly_features[:, i]

        # Rolling statistics (if time-based)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
            for col in numeric_cols:
                df[f"{col}_rolling_mean_7"] = df[col].rolling(window=7, min_periods=1).mean()
                df[f"{col}_rolling_std_7"] = df[col].rolling(window=7, min_periods=1).std()

        # Scaling
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.transformers["scaler"] = scaler

        return df

    def _create_categorical_features(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Create features from categorical columns."""
        # Target encoding
        if len(df) > 1000:  # Only for sufficient data
            from category_encoders import TargetEncoder

            encoder = TargetEncoder(cols=categorical_cols)
            df[categorical_cols] = encoder.fit_transform(
                df[categorical_cols], df.get("target", None)
            )
            self.transformers["target_encoder"] = encoder
        else:
            # One-hot encoding for small datasets
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

        return df

    def _create_date_features(self, df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        """Create features from date columns."""
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])

            # Extract date components
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)

            # Cyclical encoding
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
            df[f"{col}_day_sin"] = np.sin(2 * np.pi * df[col].dt.day / 31)
            df[f"{col}_day_cos"] = np.cos(2 * np.pi * df[col].dt.day / 31)

        return df

    def _create_interaction_features(
        self, df: pd.DataFrame, numeric_cols: List[str], max_interactions: int = 5
    ) -> pd.DataFrame:
        """Create interaction features."""
        from itertools import combinations

        interactions_created = 0
        for col1, col2 in combinations(numeric_cols, 2):
            if interactions_created >= max_interactions:
                break

            # Multiplication
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

            # Division (with protection against division by zero)
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)

            interactions_created += 1

        return df

    def _select_features(self, df: pd.DataFrame, target_column: str, k: int = 20) -> pd.DataFrame:
        """Select top k features."""
        if target_column not in df.columns:
            return df

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(X.columns)))
        selector.fit(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()

        # Store feature importance
        self.feature_importance = dict(zip(X.columns, selector.scores_))

        # Return selected features plus target
        return df[selected_features + [target_column]]


# ====================== Data Pipeline Orchestrator ======================


class DataPipelineOrchestrator:
    """
    Orchestrate complete data pipeline from ingestion to serving.
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        validator: DataValidator,
        engineering_pipeline: FeatureEngineeringPipeline,
    ):
        """
        Initialize orchestrator.

        Args:
            feature_store: Feature store instance
            validator: Data validator instance
            engineering_pipeline: Feature engineering pipeline
        """
        self.feature_store = feature_store
        self.validator = validator
        self.engineering_pipeline = engineering_pipeline
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_batch(self, raw_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of data through the complete pipeline.

        Args:
            raw_data: Raw input data
            config: Processing configuration

        Returns:
            Processing results
        """
        stages: Dict[str, Any] = {}
        results: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "started",
            "stages": stages,
        }

        try:
            # Stage 1: Validation
            logger.info("Stage 1: Validating data...")
            validation_results = await self._validate_stage(raw_data, config)
            stages["validation"] = validation_results

            if not validation_results["success"]:
                results["status"] = "failed"
                return results

            # Stage 2: Feature Engineering
            logger.info("Stage 2: Engineering features...")
            engineered_data = await self._engineering_stage(raw_data, config)
            stages["engineering"] = {
                "features_created": list(engineered_data.columns),
                "shape": engineered_data.shape,
            }

            # Stage 3: Drift Detection
            logger.info("Stage 3: Checking for drift...")
            drift_results = await self._drift_detection_stage(engineered_data, config)
            stages["drift_detection"] = drift_results

            # Stage 4: Feature Store Update
            logger.info("Stage 4: Updating feature store...")
            store_results = await self._update_feature_store(engineered_data, config)
            stages["feature_store"] = store_results

            results["status"] = "completed"

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    async def _validate_stage(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation stage."""
        loop = asyncio.get_event_loop()

        # Create or load expectation suite
        suite_name = config.get("validation_suite", "default_suite")

        if suite_name not in self.validator.expectations:
            await loop.run_in_executor(
                self.executor,
                self.validator.create_expectation_suite,
                suite_name,
                data.sample(min(1000, len(data))),
            )

        # Validate data
        validation_results = await loop.run_in_executor(
            self.executor, self.validator.validate_data, data, suite_name
        )

        return validation_results

    async def _engineering_stage(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Run feature engineering stage."""
        loop = asyncio.get_event_loop()

        feature_types = config.get("feature_types", {})
        target_column = config.get("target_column", "target")

        engineered_data = await loop.run_in_executor(
            self.executor,
            self.engineering_pipeline.create_features,
            data,
            target_column,
            feature_types,
        )

        return engineered_data

    async def _drift_detection_stage(
        self, data: pd.DataFrame, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run drift detection stage."""
        loop = asyncio.get_event_loop()

        # Load reference data
        reference_path = config.get("reference_data_path")
        if not reference_path:
            return {"skipped": True, "reason": "No reference data path provided"}

        reference_data = pd.read_parquet(reference_path)

        # Detect drift
        drift_results = await loop.run_in_executor(
            self.executor, self.validator.detect_drift, reference_data, data
        )

        return drift_results

    async def _update_feature_store(
        self, data: pd.DataFrame, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update feature store with new data."""
        loop = asyncio.get_event_loop()

        # Prepare data for feature store
        data["event_timestamp"] = pd.Timestamp.now()

        # Save to parquet for feature store
        default_path = os.path.join(tempfile.gettempdir(), "features.parquet")
        output_path = config.get("feature_store_path", default_path)
        await loop.run_in_executor(self.executor, data.to_parquet, output_path)

        # Register features if needed
        if config.get("register_features", False):
            features = [
                {"name": col, "dtype": "float64"}
                for col in data.columns
                if col not in ["event_timestamp", "entity_id"]
            ]

            await loop.run_in_executor(
                self.executor,
                self.feature_store.register_features,
                features,
                config.get("entity_name", "default_entity"),
                output_path,
            )

        # Materialize features
        await loop.run_in_executor(
            self.executor,
            self.feature_store.materialize_features,
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )

        return {
            "features_updated": len(data.columns),
            "rows_processed": len(data),
            "output_path": output_path,
        }


# ====================== Example Usage ======================


async def main():
    """Example usage of the data pipeline."""

    # Initialize components
    validator = DataValidator()

    feature_store = FeatureStore(
        repo_path="./feature_repo", redis_host="localhost", redis_port=6379
    )

    engineering_config = {
        "create_interactions": True,
        "max_interactions": 5,
        "feature_selection": True,
        "n_features": 20,
    }
    engineering_pipeline = FeatureEngineeringPipeline(engineering_config)

    # Create orchestrator
    orchestrator = DataPipelineOrchestrator(
        feature_store=feature_store, validator=validator, engineering_pipeline=engineering_pipeline
    )

    # Generate sample data
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "feature1": np.random.randn(1000),
            "feature2": np.random.randn(1000),
            "feature3": np.random.randn(1000),
            "category1": np.random.choice(["A", "B", "C"], 1000),
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="H"),
            "target": np.random.randint(0, 2, 1000),
        }
    )

    # Process configuration
    process_config = {
        "validation_suite": "production_suite",
        "feature_types": {
            "feature1": "numeric",
            "feature2": "numeric",
            "feature3": "numeric",
            "category1": "categorical",
            "timestamp": "datetime",
        },
        "target_column": "target",
        "reference_data_path": None,  # Would be actual path in production
        "feature_store_path": os.path.join(tempfile.gettempdir(), "features.parquet"),
        "register_features": True,
        "entity_name": "customer",
    }

    # Process batch
    results = await orchestrator.process_batch(sample_data, process_config)

    print("Pipeline Results:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    os.makedirs("./feature_repo", exist_ok=True)
    asyncio.run(main())
