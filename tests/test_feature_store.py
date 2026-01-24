"""
Unit tests for Feature Store and Data Validation Pipeline
"""

import sys
import unittest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

import numpy as np
import pandas as pd

# Mock external dependencies before importing the module
sys.modules['great_expectations'] = MagicMock()
sys.modules['great_expectations.core'] = MagicMock()
sys.modules['great_expectations.core.batch'] = MagicMock()
sys.modules['feast'] = MagicMock()
sys.modules['feast.types'] = MagicMock()
sys.modules['evidently'] = MagicMock()
sys.modules['evidently.report'] = MagicMock()
sys.modules['evidently.metric_preset'] = MagicMock()
sys.modules['evidently.test_suite'] = MagicMock()
sys.modules['evidently.test_preset'] = MagicMock()
sys.modules['mlflow'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['sqlalchemy'] = MagicMock()
sys.modules['sqlalchemy.ext'] = MagicMock()
sys.modules['sqlalchemy.ext.declarative'] = MagicMock()
sys.modules['sqlalchemy.orm'] = MagicMock()
sys.modules['sqlalchemy.pool'] = MagicMock()
sys.modules['category_encoders'] = MagicMock()

# Create mock for sqlalchemy Base
mock_base = MagicMock()
mock_base.metadata = MagicMock()
sys.modules['sqlalchemy'].ext.declarative.declarative_base = MagicMock(return_value=mock_base)

from src.feature_store import (
    ValidationRule,
    DataValidator,
    FeatureStore,
    FeatureEngineeringPipeline,
    DataPipelineOrchestrator,
)


class TestValidationRule(unittest.TestCase):
    """Test ValidationRule dataclass."""

    def test_validation_rule_creation(self):
        """Test creating a validation rule."""
        rule = ValidationRule(
            name="test_rule",
            column="test_column",
            rule_type="range",
            parameters={"min": 0, "max": 100},
            severity="warning"
        )

        self.assertEqual(rule.name, "test_rule")
        self.assertEqual(rule.column, "test_column")
        self.assertEqual(rule.rule_type, "range")
        self.assertEqual(rule.parameters["min"], 0)
        self.assertEqual(rule.severity, "warning")

    def test_validation_rule_default_severity(self):
        """Test default severity for validation rule."""
        rule = ValidationRule(
            name="test_rule",
            column="test_column",
            rule_type="not_null",
            parameters={}
        )

        self.assertEqual(rule.severity, "warning")


class TestDataValidator(unittest.TestCase):
    """Test DataValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create sample data
        self.df = pd.DataFrame({
            'numeric_col': np.random.randn(100),
            'category_col': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

    @patch('src.feature_store.ge.get_context')
    @patch('src.feature_store.create_engine')
    def test_validator_initialization(self, mock_engine, mock_ge_context):
        """Test DataValidator initialization."""
        mock_ge_context.return_value = Mock()
        mock_engine.return_value = Mock()

        validator = DataValidator()

        self.assertIsNotNone(validator.ge_context)
        self.assertEqual(validator.expectations, {})
        self.assertEqual(validator.validation_results, [])

    @patch('src.feature_store.ge.get_context')
    @patch('src.feature_store.create_engine')
    @patch('src.feature_store.ge.dataset.PandasDataset')
    def test_create_expectation_suite(self, mock_pandas_dataset, mock_engine, mock_ge_context):
        """Test creating expectation suite."""
        # Setup mocks
        mock_context = Mock()
        mock_suite = Mock()
        mock_context.create_expectation_suite.return_value = mock_suite
        mock_ge_context.return_value = mock_context
        mock_engine.return_value = Mock()

        validator = DataValidator()

        # Use only numeric data to avoid dtype issues
        numeric_df = pd.DataFrame({
            'numeric_col': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        suite = validator.create_expectation_suite("test_suite", numeric_df)

        mock_context.create_expectation_suite.assert_called_once_with(
            expectation_suite_name="test_suite",
            overwrite_existing=True
        )
        self.assertEqual(validator.expectations["test_suite"], mock_suite)

    @patch('src.feature_store.ge.get_context')
    @patch('src.feature_store.create_engine')
    def test_validate_data_missing_suite(self, mock_engine, mock_ge_context):
        """Test validation with missing expectation suite."""
        mock_ge_context.return_value = Mock()
        mock_engine.return_value = Mock()

        validator = DataValidator()

        with self.assertRaises(ValueError) as context:
            validator.validate_data(self.df, "nonexistent_suite")

        self.assertIn("not found", str(context.exception))

    @patch('src.feature_store.ge.get_context')
    @patch('src.feature_store.create_engine')
    @patch('src.feature_store.Report')
    def test_detect_drift(self, mock_report_class, mock_engine, mock_ge_context):
        """Test drift detection."""
        mock_ge_context.return_value = Mock()
        mock_engine.return_value = Mock()

        # Setup mock report
        mock_report = Mock()
        mock_report.as_dict.return_value = {
            "metrics": [{
                "result": {
                    "dataset_drift": True,
                    "share_of_drifted_columns": 0.5,
                    "drift_by_columns": {
                        "numeric_col": {
                            "drift_detected": True,
                            "drift_score": 0.8,
                            "stattest_name": "ks",
                            "p_value": 0.01
                        }
                    }
                }
            }]
        }
        mock_report_class.return_value = mock_report

        validator = DataValidator()

        # Mock the session to avoid database calls
        validator.SessionLocal = Mock()
        mock_session = MagicMock()
        validator.SessionLocal.return_value.__enter__ = Mock(return_value=mock_session)
        validator.SessionLocal.return_value.__exit__ = Mock(return_value=False)

        reference_data = self.df.copy()
        current_data = self.df.copy()
        current_data['numeric_col'] = current_data['numeric_col'] + 5  # Introduce drift

        result = validator.detect_drift(reference_data, current_data)

        self.assertTrue(result["dataset_drift"])
        self.assertEqual(result["share_of_drifted_columns"], 0.5)
        self.assertEqual(len(result["drifted_columns"]), 1)
        self.assertEqual(result["drifted_columns"][0]["column"], "numeric_col")


class TestFeatureStore(unittest.TestCase):
    """Test FeatureStore class."""

    @patch('src.feature_store.redis.Redis')
    @patch('src.feature_store.feast.FeatureStore')
    @patch('src.feature_store.create_engine')
    def test_feature_store_initialization(self, mock_engine, mock_feast, mock_redis):
        """Test FeatureStore initialization."""
        mock_redis.return_value = Mock()
        mock_feast.return_value = Mock()
        mock_engine.return_value = Mock()

        store = FeatureStore(
            repo_path="./test_repo",
            redis_host="localhost",
            redis_port=6379
        )

        self.assertEqual(store.repo_path, "./test_repo")
        mock_redis.assert_called_once_with(
            host="localhost",
            port=6379,
            decode_responses=True
        )

    @patch('src.feature_store.redis.Redis')
    @patch('src.feature_store.feast.FeatureStore')
    @patch('src.feature_store.create_engine')
    def test_get_feast_dtype(self, mock_engine, mock_feast, mock_redis):
        """Test dtype conversion."""
        mock_redis.return_value = Mock()
        mock_feast.return_value = Mock()
        mock_engine.return_value = Mock()

        store = FeatureStore(repo_path="./test_repo")

        # Test that _get_feast_dtype returns something (mocked types)
        result = store._get_feast_dtype("float32")
        self.assertIsNotNone(result)

        result = store._get_feast_dtype("unknown")
        self.assertIsNotNone(result)  # Should return default

    @patch('src.feature_store.redis.Redis')
    @patch('src.feature_store.feast.FeatureStore')
    @patch('src.feature_store.create_engine')
    @patch('src.feature_store.pd.read_json')
    def test_get_online_features_cached(self, mock_read_json, mock_engine, mock_feast, mock_redis):
        """Test getting online features from cache."""
        mock_redis_client = Mock()
        mock_redis_client.get.return_value = '{"col1": [1, 2], "col2": [3, 4]}'
        mock_redis.return_value = mock_redis_client
        mock_feast.return_value = Mock()
        mock_engine.return_value = Mock()
        mock_read_json.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        store = FeatureStore(repo_path="./test_repo")

        result = store.get_online_features(
            entity_ids=[1, 2],
            feature_names=["feature1"],
            cache_ttl=60
        )

        self.assertIsInstance(result, pd.DataFrame)
        mock_redis_client.get.assert_called_once()

    @patch('src.feature_store.redis.Redis')
    @patch('src.feature_store.feast.FeatureStore')
    @patch('src.feature_store.create_engine')
    def test_get_online_features_not_cached(self, mock_engine, mock_feast, mock_redis):
        """Test getting online features when not in cache."""
        mock_redis_client = Mock()
        mock_redis_client.get.return_value = None
        mock_redis.return_value = mock_redis_client

        mock_feast_store = Mock()
        mock_feature_vector = Mock()
        mock_feature_vector.to_df.return_value = pd.DataFrame({'col1': [1, 2]})
        mock_feast_store.get_online_features.return_value = mock_feature_vector
        mock_feast.return_value = mock_feast_store

        mock_engine.return_value = Mock()

        store = FeatureStore(repo_path="./test_repo")

        result = store.get_online_features(
            entity_ids=[1, 2],
            feature_names=["feature1"],
            cache_ttl=60
        )

        self.assertIsInstance(result, pd.DataFrame)
        mock_redis_client.setex.assert_called_once()

    @patch('src.feature_store.redis.Redis')
    @patch('src.feature_store.feast.FeatureStore')
    @patch('src.feature_store.create_engine')
    def test_materialize_features(self, mock_engine, mock_feast, mock_redis):
        """Test feature materialization."""
        mock_redis.return_value = Mock()
        mock_feast_store = Mock()
        mock_feast.return_value = mock_feast_store
        mock_engine.return_value = Mock()

        store = FeatureStore(repo_path="./test_repo")

        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()

        store.materialize_features(start_date, end_date)

        mock_feast_store.materialize_incremental.assert_called_once_with(
            start_date=start_date,
            end_date=end_date
        )


class TestFeatureEngineeringPipeline(unittest.TestCase):
    """Test FeatureEngineeringPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        self.config = {
            "create_interactions": False,
            "feature_selection": False
        }

        self.df = pd.DataFrame({
            'numeric1': np.random.randn(100) + 10,  # Positive values for log
            'numeric2': np.random.randn(100) + 10,
            'category1': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = FeatureEngineeringPipeline(self.config)

        self.assertEqual(pipeline.config, self.config)
        self.assertEqual(pipeline.transformers, {})
        self.assertEqual(pipeline.feature_importance, {})

    @patch('src.feature_store.StandardScaler')
    def test_create_numeric_features(self, mock_scaler_class):
        """Test numeric feature creation."""
        # Mock the scaler to avoid sklearn/pyarrow issues
        mock_scaler = Mock()
        mock_scaler.fit_transform.return_value = self.df[['numeric1', 'numeric2']].values
        mock_scaler_class.return_value = mock_scaler

        pipeline = FeatureEngineeringPipeline(self.config)

        df_result = pipeline._create_numeric_features(
            self.df.copy(),
            ['numeric1', 'numeric2']
        )

        # Check log features were created (values are positive)
        self.assertIn('numeric1_log', df_result.columns)
        self.assertIn('numeric2_log', df_result.columns)

        # Check scaler was stored
        self.assertIn('scaler', pipeline.transformers)

    def test_create_categorical_features_small_dataset(self):
        """Test categorical feature creation with small dataset."""
        pipeline = FeatureEngineeringPipeline(self.config)

        df_result = pipeline._create_categorical_features(
            self.df.copy(),
            ['category1']
        )

        # For small datasets, one-hot encoding is used
        self.assertIn('category1_A', df_result.columns)
        self.assertIn('category1_B', df_result.columns)
        self.assertIn('category1_C', df_result.columns)

    def test_create_date_features(self):
        """Test date feature creation."""
        pipeline = FeatureEngineeringPipeline(self.config)

        df_with_dates = self.df.copy()
        df_with_dates['date_col'] = pd.date_range('2024-01-01', periods=100, freq='D')

        df_result = pipeline._create_date_features(df_with_dates, ['date_col'])

        # Check date features
        self.assertIn('date_col_year', df_result.columns)
        self.assertIn('date_col_month', df_result.columns)
        self.assertIn('date_col_day', df_result.columns)
        self.assertIn('date_col_dayofweek', df_result.columns)
        self.assertIn('date_col_quarter', df_result.columns)
        self.assertIn('date_col_is_weekend', df_result.columns)

        # Check cyclical encoding
        self.assertIn('date_col_month_sin', df_result.columns)
        self.assertIn('date_col_month_cos', df_result.columns)

    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        pipeline = FeatureEngineeringPipeline(self.config)

        df_result = pipeline._create_interaction_features(
            self.df.copy(),
            ['numeric1', 'numeric2'],
            max_interactions=5
        )

        # Check interaction features
        self.assertIn('numeric1_x_numeric2', df_result.columns)
        self.assertIn('numeric1_div_numeric2', df_result.columns)

    @patch('src.feature_store.SelectKBest')
    def test_select_features(self, mock_selector_class):
        """Test feature selection."""
        pipeline = FeatureEngineeringPipeline(self.config)

        # Mock the selector - match the number of feature columns (2, excluding target)
        mock_selector = Mock()
        mock_selector.get_support.return_value = np.array([True, True])
        mock_selector.scores_ = np.array([0.5, 0.3])
        mock_selector_class.return_value = mock_selector

        # Use only numeric columns for this test
        df_numeric = self.df[['numeric1', 'numeric2', 'target']].copy()

        df_result = pipeline._select_features(
            df_numeric,
            'target',
            k=2
        )

        # Check that target is in result
        self.assertIn('target', df_result.columns)

        # Check feature importance was stored
        self.assertGreater(len(pipeline.feature_importance), 0)

    @patch('src.feature_store.StandardScaler')
    def test_create_features_full_pipeline(self, mock_scaler_class):
        """Test full feature creation pipeline."""
        # Mock the scaler to avoid sklearn/pyarrow issues
        mock_scaler = Mock()
        mock_scaler.fit_transform.return_value = self.df[['numeric1', 'numeric2']].values
        mock_scaler_class.return_value = mock_scaler

        config = {
            "create_interactions": True,
            "max_interactions": 2,
            "feature_selection": False
        }
        pipeline = FeatureEngineeringPipeline(config)

        feature_types = {
            'numeric1': 'numeric',
            'numeric2': 'numeric',
            'category1': 'categorical'
        }

        df_result = pipeline.create_features(
            self.df.copy(),
            'target',
            feature_types
        )

        # Original columns should still exist (modified)
        self.assertIn('target', df_result.columns)

        # New features should be created
        self.assertGreater(len(df_result.columns), len(self.df.columns))

    def test_select_features_missing_target(self):
        """Test feature selection with missing target column."""
        pipeline = FeatureEngineeringPipeline(self.config)

        df_no_target = self.df.drop(columns=['target'])

        df_result = pipeline._select_features(
            df_no_target,
            'target',  # Missing column
            k=2
        )

        # Should return original dataframe unchanged
        self.assertEqual(len(df_result.columns), len(df_no_target.columns))


class TestDataPipelineOrchestrator(unittest.TestCase):
    """Test DataPipelineOrchestrator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        self.df = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        mock_feature_store = Mock()
        mock_validator = Mock()
        mock_engineering = Mock()

        orchestrator = DataPipelineOrchestrator(
            feature_store=mock_feature_store,
            validator=mock_validator,
            engineering_pipeline=mock_engineering
        )

        self.assertEqual(orchestrator.feature_store, mock_feature_store)
        self.assertEqual(orchestrator.validator, mock_validator)
        self.assertEqual(orchestrator.engineering_pipeline, mock_engineering)

    def test_validate_stage(self):
        """Test validation stage."""
        import asyncio

        mock_feature_store = Mock()
        mock_validator = Mock()
        mock_validator.expectations = {}
        mock_validator.create_expectation_suite = Mock()
        mock_validator.validate_data = Mock(return_value={"success": True})
        mock_engineering = Mock()

        orchestrator = DataPipelineOrchestrator(
            feature_store=mock_feature_store,
            validator=mock_validator,
            engineering_pipeline=mock_engineering
        )

        config = {"validation_suite": "test_suite"}

        # Run the async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                orchestrator._validate_stage(self.df, config)
            )
            self.assertEqual(result, {"success": True})
        finally:
            loop.close()

    def test_engineering_stage(self):
        """Test feature engineering stage."""
        import asyncio

        mock_feature_store = Mock()
        mock_validator = Mock()
        mock_engineering = Mock()
        mock_engineering.create_features = Mock(return_value=self.df)

        orchestrator = DataPipelineOrchestrator(
            feature_store=mock_feature_store,
            validator=mock_validator,
            engineering_pipeline=mock_engineering
        )

        config = {
            "feature_types": {"numeric1": "numeric"},
            "target_column": "target"
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                orchestrator._engineering_stage(self.df, config)
            )
            self.assertIsInstance(result, pd.DataFrame)
        finally:
            loop.close()

    def test_drift_detection_stage_no_reference(self):
        """Test drift detection stage without reference data."""
        import asyncio

        mock_feature_store = Mock()
        mock_validator = Mock()
        mock_engineering = Mock()

        orchestrator = DataPipelineOrchestrator(
            feature_store=mock_feature_store,
            validator=mock_validator,
            engineering_pipeline=mock_engineering
        )

        config = {}  # No reference_data_path

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                orchestrator._drift_detection_stage(self.df, config)
            )
            self.assertTrue(result["skipped"])
            self.assertIn("No reference data", result["reason"])
        finally:
            loop.close()


if __name__ == '__main__':
    unittest.main()
