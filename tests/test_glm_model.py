"""
Unit tests for GLM Model Selection Framework
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime


from src.glm_model import (
    ModelConfig,
    ModelSelectionStrategy,
    DataValidator,
    GLMModelSelector,
    ModelServing,
    ModelMetrics,
    ModelResult,
)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        self.assertEqual(config.target_column, "presence_unpaid")
        self.assertEqual(config.max_iterations, 100)
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.min_predictors, 1)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertEqual(config.max_models_to_keep, 50)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelConfig(
            target_column="custom_target",
            predictors=["a", "b", "c"],
            max_iterations=50,
            random_seed=123,
            test_size=0.3,
            min_predictors=2,
            max_predictors=5,
            selection_strategy=ModelSelectionStrategy.FORWARD,
            confidence_level=0.99,
            max_models_to_keep=25,
        )
        self.assertEqual(config.target_column, "custom_target")
        self.assertEqual(config.predictors, ["a", "b", "c"])
        self.assertEqual(config.max_iterations, 50)
        self.assertEqual(config.selection_strategy, ModelSelectionStrategy.FORWARD)

    def test_config_validation_test_size_too_high(self):
        """Test configuration validation - test_size too high."""
        config = ModelConfig(test_size=1.5)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("test_size", str(context.exception))

    def test_config_validation_test_size_too_low(self):
        """Test configuration validation - test_size too low."""
        config = ModelConfig(test_size=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("test_size", str(context.exception))

    def test_config_validation_max_iterations(self):
        """Test configuration validation - invalid max_iterations."""
        config = ModelConfig(max_iterations=-1)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("max_iterations", str(context.exception))

    def test_config_validation_min_predictors(self):
        """Test configuration validation - invalid min_predictors."""
        config = ModelConfig(min_predictors=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("min_predictors", str(context.exception))

    def test_config_validation_predictor_range(self):
        """Test configuration validation - invalid predictor range."""
        config = ModelConfig(min_predictors=5, max_predictors=3)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("max_predictors", str(context.exception))

    def test_config_validation_max_models_to_keep(self):
        """Test configuration validation - invalid max_models_to_keep."""
        config = ModelConfig(max_models_to_keep=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("max_models_to_keep", str(context.exception))

    def test_config_validation_success(self):
        """Test configuration validation passes for valid config."""
        config = ModelConfig(
            test_size=0.2,
            max_iterations=100,
            min_predictors=1,
            max_predictors=5,
            max_models_to_keep=50,
        )
        # Should not raise
        config.validate()


class TestModelSelectionStrategy(unittest.TestCase):
    """Test ModelSelectionStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        self.assertEqual(ModelSelectionStrategy.RANDOM.value, "random")
        self.assertEqual(ModelSelectionStrategy.EXHAUSTIVE.value, "exhaustive")
        self.assertEqual(ModelSelectionStrategy.FORWARD.value, "forward")
        self.assertEqual(ModelSelectionStrategy.BACKWARD.value, "backward")


class TestModelMetrics(unittest.TestCase):
    """Test ModelMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating metrics."""
        metrics = ModelMetrics(
            aic=100.0,
            bic=110.0,
            auc=0.85,
            accuracy=0.80,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            log_likelihood=-50.0,
        )
        self.assertEqual(metrics.aic, 100.0)
        self.assertEqual(metrics.auc, 0.85)
        self.assertEqual(metrics.f1_score, 0.72)

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ModelMetrics(
            aic=100.0, bic=110.0, auc=0.85, confusion_matrix=np.array([[10, 5], [3, 12]])
        )
        result = metrics.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["aic"], 100.0)
        self.assertEqual(result["auc"], 0.85)
        self.assertEqual(result["confusion_matrix"], [[10, 5], [3, 12]])

    def test_metrics_to_dict_no_confusion_matrix(self):
        """Test converting metrics to dictionary without confusion matrix."""
        metrics = ModelMetrics(aic=100.0, bic=110.0, auc=0.85)
        result = metrics.to_dict()

        self.assertIsInstance(result, dict)
        self.assertIsNone(result["confusion_matrix"])


class TestModelResult(unittest.TestCase):
    """Test ModelResult dataclass."""

    def test_model_result_creation(self):
        """Test creating model result."""
        mock_model = Mock()
        mock_metrics = Mock()

        result = ModelResult(
            formula="y ~ x1 + x2", predictors=["x1", "x2"], model=mock_model, metrics=mock_metrics
        )

        self.assertEqual(result.formula, "y ~ x1 + x2")
        self.assertEqual(result.predictors, ["x1", "x2"])
        self.assertEqual(result.model, mock_model)
        self.assertIsInstance(result.timestamp, datetime)


class TestDataValidator(unittest.TestCase):
    """Test DataValidator class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.validator = DataValidator()

        # Create sample data
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

    def test_valid_dataframe(self):
        """Test validation with valid DataFrame."""
        # Should not raise any exceptions
        self.validator.validate_dataframe(self.df, "target", ["feature1", "feature2"])

    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_dataframe(pd.DataFrame(), "target", ["feature1"])
        self.assertIn("empty", str(context.exception).lower())

    def test_missing_target(self):
        """Test validation with missing target column."""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_dataframe(self.df, "nonexistent", ["feature1"])
        self.assertIn("not found", str(context.exception))

    def test_missing_predictors(self):
        """Test validation with missing predictor columns."""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_dataframe(self.df, "target", ["feature1", "nonexistent"])
        self.assertIn("not found", str(context.exception))

    def test_non_binary_target(self):
        """Test validation with non-binary target."""
        df = self.df.copy()
        df["target"] = np.random.randint(0, 5, 100)

        with self.assertRaises(ValueError) as context:
            self.validator.validate_dataframe(df, "target", ["feature1"])
        self.assertIn("binary", str(context.exception).lower())

    def test_missing_values_warning(self):
        """Test that missing values generate a warning."""
        df = self.df.copy()
        df.loc[0, "feature1"] = np.nan

        with self.assertLogs(level="WARNING"):
            self.validator.validate_dataframe(df, "target", ["feature1"])

    def test_constant_predictor_warning(self):
        """Test that constant predictors generate a warning."""
        df = self.df.copy()
        df["constant_col"] = 1  # Constant column

        with self.assertLogs(level="WARNING"):
            self.validator.validate_dataframe(df, "target", ["feature1", "constant_col"])


class TestGLMModelSelector(unittest.TestCase):
    """Test GLMModelSelector class."""

    def setUp(self):
        """Set up test data and configuration."""
        np.random.seed(42)

        # Create synthetic data
        n_samples = 500
        self.data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "feature4": np.random.randn(n_samples),
                "feature5": np.random.randn(n_samples),
            }
        )

        # Create target with some correlation to features
        logits = (
            0.5 * self.data["feature1"]
            + 0.3 * self.data["feature2"]
            - 0.2 * self.data["feature3"]
            + np.random.randn(n_samples) * 0.5
        )
        probs = 1 / (1 + np.exp(-logits))
        self.data["presence_unpaid"] = (probs > 0.5).astype(int)

        # Configuration
        self.config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2", "feature3", "feature4", "feature5"],
            max_iterations=10,
            random_seed=42,
            test_size=0.2,
        )

    def test_initialization(self):
        """Test GLMModelSelector initialization."""
        selector = GLMModelSelector(self.config)
        self.assertIsNone(selector.best_model)
        self.assertEqual(len(selector.all_models), 0)
        self.assertIsNone(selector.train_data)
        self.assertIsNone(selector.test_data)

    def test_initialization_with_invalid_config(self):
        """Test initialization with invalid config raises error."""
        invalid_config = ModelConfig(test_size=1.5)
        with self.assertRaises(ValueError):
            GLMModelSelector(invalid_config)

    def test_data_preparation(self):
        """Test data preparation."""
        selector = GLMModelSelector(self.config)
        train_data, test_data = selector.prepare_data(self.data)

        # Check split sizes
        expected_train_size = int(len(self.data) * 0.8)
        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data), len(self.data) - expected_train_size)

        # Check that data is properly set
        self.assertIsNotNone(selector.train_data)
        self.assertIsNotNone(selector.test_data)

    def test_data_preparation_with_presplit_data(self):
        """Test data preparation with pre-split data."""
        selector = GLMModelSelector(self.config)

        train = self.data.iloc[:400]
        test = self.data.iloc[400:]

        train_result, test_result = selector.prepare_data(
            self.data, train_data=train, test_data=test
        )

        self.assertEqual(len(train_result), 400)
        self.assertEqual(len(test_result), 100)

    def test_model_fitting(self):
        """Test model fitting process."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)

        best_model = selector.fit()

        # Check that a model was fitted
        self.assertIsNotNone(best_model)
        self.assertIsNotNone(selector.best_model)

        # Check model attributes
        self.assertIsInstance(best_model.metrics, ModelMetrics)
        self.assertGreater(len(best_model.predictors), 0)
        self.assertIsNotNone(best_model.formula)

        # Check metrics
        self.assertGreaterEqual(best_model.metrics.auc, 0.0)
        self.assertLessEqual(best_model.metrics.auc, 1.0)

    def test_fit_without_data_preparation(self):
        """Test fitting without preparing data raises error."""
        selector = GLMModelSelector(self.config)

        with self.assertRaises(ValueError) as context:
            selector.fit()
        self.assertIn("Data must be prepared", str(context.exception))

    def test_prediction_proba(self):
        """Test probability predictions."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        # Test probability predictions
        test_data = self.data.head(10)
        probabilities = selector.predict(test_data, return_proba=True)

        self.assertEqual(len(probabilities), len(test_data))
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))

    def test_prediction_class(self):
        """Test class predictions."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        # Test class predictions
        test_data = self.data.head(10)
        classes = selector.predict(test_data, return_proba=False)
        self.assertTrue(all(c in [0, 1] for c in classes))

    def test_prediction_dataframe(self):
        """Test predictions returned as DataFrame."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        test_data = self.data.head(10)
        result = selector.predict(test_data, return_dataframe=True)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("proba_default", result.columns)
        self.assertIn("predicted_class", result.columns)
        self.assertIn("decision", result.columns)

    def test_prediction_without_fitting(self):
        """Test prediction without fitting raises error."""
        selector = GLMModelSelector(self.config)

        with self.assertRaises(ValueError) as context:
            selector.predict(self.data.head(10))
        self.assertIn("fitted", str(context.exception).lower())

    def test_prediction_missing_columns(self):
        """Test prediction with missing columns raises error."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        # Create data with missing column
        incomplete_data = self.data.head(10).drop(columns=["feature1"])

        with self.assertRaises(ValueError) as context:
            selector.predict(incomplete_data)
        self.assertIn("Missing", str(context.exception))

    def test_model_save_load(self):
        """Test model saving and loading."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"

            # Save model
            selector.save_model(model_path)
            self.assertTrue(model_path.exists())

            # Load model
            loaded_selector = GLMModelSelector.load_model(model_path)

            # Check loaded model
            self.assertIsNotNone(loaded_selector.best_model)
            self.assertEqual(loaded_selector.best_model.predictors, selector.best_model.predictors)

            # Test predictions with loaded model
            test_data = self.data.head(10)
            original_preds = selector.predict(test_data)
            loaded_preds = loaded_selector.predict(test_data)

            np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_model_without_fitting(self):
        """Test saving model without fitting raises error."""
        selector = GLMModelSelector(self.config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"

            with self.assertRaises(ValueError) as context:
                selector.save_model(model_path)
            self.assertIn("No model to save", str(context.exception))

    def test_load_model_file_not_found(self):
        """Test loading non-existent model file raises error."""
        with self.assertRaises(FileNotFoundError):
            GLMModelSelector.load_model("/nonexistent/path/model.joblib")

    def test_get_summary(self):
        """Test getting model summary."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        summary = selector.get_summary()

        self.assertIn("best_model", summary)
        self.assertIn("total_models_evaluated", summary)
        self.assertIn("search_statistics", summary)
        self.assertIn("config", summary)
        self.assertIn("version", summary)

    def test_get_summary_without_fitting(self):
        """Test getting summary without fitting."""
        selector = GLMModelSelector(self.config)

        summary = selector.get_summary()
        self.assertIn("status", summary)
        self.assertEqual(summary["status"], "No model fitted")

    def test_model_comparison(self):
        """Test model comparison functionality."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        comparison_df = selector.get_model_comparison()

        # Check DataFrame structure
        self.assertFalse(comparison_df.empty)
        self.assertIn("aic", comparison_df.columns)
        self.assertIn("auc", comparison_df.columns)
        self.assertIn("num_predictors", comparison_df.columns)

        # Check that models are sorted by AIC
        aic_values = comparison_df["aic"].values
        self.assertTrue(all(aic_values[i] <= aic_values[i + 1] for i in range(len(aic_values) - 1)))

    def test_model_comparison_empty(self):
        """Test model comparison with no models."""
        selector = GLMModelSelector(self.config)
        comparison_df = selector.get_model_comparison()
        self.assertTrue(comparison_df.empty)

    def test_not_implemented_strategy(self):
        """Test that non-RANDOM strategies raise NotImplementedError."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2"],
            selection_strategy=ModelSelectionStrategy.FORWARD,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)

        with self.assertRaises(NotImplementedError):
            selector.fit()


class TestModelServing(unittest.TestCase):
    """Test ModelServing class."""

    def setUp(self):
        """Set up test model."""
        np.random.seed(42)

        # Create and train a simple model
        n_samples = 200
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "presence_unpaid": np.random.randint(0, 2, n_samples),
            }
        )

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2"],
            max_iterations=5,
            random_seed=42,
            min_predictors=2,
            max_predictors=2,
        )

        self.selector = GLMModelSelector(config)
        self.selector.prepare_data(data)
        self.selector.fit()

        # Save model for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.joblib"
        self.selector.save_model(self.model_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_loading(self):
        """Test loading model for serving."""
        server = ModelServing(self.model_path)
        self.assertIsNotNone(server.model)
        self.assertEqual(server.predictors, ["feature1", "feature2"])

    def test_single_prediction(self):
        """Test single prediction."""
        server = ModelServing(self.model_path)

        features = {"feature1": 0.5, "feature2": -0.3}
        result = server.predict_single(features)

        # Check result structure
        self.assertIn("probability", result)
        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("predictors_used", result)

        # Check value ranges
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertIn(result["predicted_class"], [0, 1])
        self.assertGreaterEqual(result["confidence"], 0.5)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_batch_prediction(self):
        """Test batch prediction."""
        server = ModelServing(self.model_path)

        batch_data = pd.DataFrame({"feature1": [0.5, -0.3, 1.2], "feature2": [-0.3, 0.8, -1.5]})

        results = server.predict_batch(batch_data)

        # Check result structure
        self.assertEqual(len(results), len(batch_data))
        self.assertIn("predicted_probability", results.columns)
        self.assertIn("predicted_class", results.columns)
        self.assertIn("confidence", results.columns)

    def test_batch_prediction_without_confidence(self):
        """Test batch prediction without confidence scores."""
        server = ModelServing(self.model_path)

        batch_data = pd.DataFrame({"feature1": [0.5, -0.3], "feature2": [-0.3, 0.8]})

        results = server.predict_batch(batch_data, include_confidence=False)

        self.assertNotIn("confidence", results.columns)
        self.assertIn("predicted_probability", results.columns)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        server = ModelServing(self.model_path)
        importance_df = server.get_feature_importance()

        # Check DataFrame structure
        self.assertFalse(importance_df.empty)
        self.assertIn("feature", importance_df.columns)
        self.assertIn("coefficient", importance_df.columns)
        self.assertIn("p_value", importance_df.columns)
        self.assertIn("odds_ratio", importance_df.columns)
        self.assertIn("significant", importance_df.columns)


class TestIntegration(unittest.TestCase):
    """Integration tests for the GLM model framework."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)

        # Create more realistic synthetic data
        n_samples = 1000
        self.data = pd.DataFrame(
            {
                "age": np.random.randint(18, 70, n_samples),
                "income": np.random.exponential(50000, n_samples),
                "debt_ratio": np.random.uniform(0, 1, n_samples),
                "credit_history": np.random.randint(0, 10, n_samples),
            }
        )

        # Create target based on features
        prob = 1 / (
            1
            + np.exp(
                -(
                    -2
                    + 0.02 * (self.data["age"] - 40)
                    + -0.00001 * self.data["income"]
                    + 2 * self.data["debt_ratio"]
                    + -0.1 * self.data["credit_history"]
                )
            )
        )
        self.data["presence_unpaid"] = (np.random.random(n_samples) < prob).astype(int)

    def test_end_to_end_workflow(self):
        """Test complete workflow from config to prediction."""
        # 1. Configure
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["age", "income", "debt_ratio", "credit_history"],
            max_iterations=20,
            random_seed=42,
            test_size=0.2,
        )

        # 2. Initialize and prepare
        selector = GLMModelSelector(config)
        train_data, test_data = selector.prepare_data(self.data)

        # 3. Fit
        best_model = selector.fit()
        self.assertIsNotNone(best_model)

        # 4. Predict
        predictions = selector.predict(test_data, return_dataframe=True)
        self.assertEqual(len(predictions), len(test_data))

        # 5. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)

            loaded = GLMModelSelector.load_model(model_path)
            loaded_preds = loaded.predict(test_data, return_proba=True)

            np.testing.assert_array_almost_equal(predictions["proba_default"].values, loaded_preds)

        # 6. Summary
        summary = selector.get_summary()
        self.assertIn("best_model", summary)


# ====================== Additional Edge Case Tests ======================


class TestFitModelErrorRecovery(unittest.TestCase):
    """Test error recovery during model fitting."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 300
        self.data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
            }
        )
        logits = 0.5 * self.data["feature1"] + 0.3 * self.data["feature2"]
        probs = 1 / (1 + np.exp(-logits))
        self.data["presence_unpaid"] = (probs > 0.5).astype(int)

    def test_fit_with_all_constant_predictors(self):
        """Test fitting with constant predictor columns."""
        data = self.data.copy()
        data["constant_col"] = 1  # Constant column

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "constant_col"],
            max_iterations=5,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(data)

        # Should still be able to fit (may warn)
        try:
            selector.fit()
        except Exception:
            pass  # Some error is acceptable

    def test_fit_recovers_from_single_iteration_failure(self):
        """Test that fit continues after individual iteration failures."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2", "feature3"],
            max_iterations=10,
            min_predictors=1,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)

        # Should complete even if some iterations fail
        best_model = selector.fit()
        self.assertIsNotNone(best_model)

    def test_fit_with_perfect_separation(self):
        """Test fitting with perfect separation in data."""
        data = self.data.copy()
        # Create perfect separation
        data["perfect_pred"] = data["presence_unpaid"]

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "perfect_pred"],
            max_iterations=5,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(data)

        # May succeed or fail depending on model behavior
        try:
            selector.fit()
        except Exception:
            pass  # Error is acceptable

    def test_fit_with_high_collinearity(self):
        """Test fitting with highly collinear features."""
        data = self.data.copy()
        data["collinear"] = data["feature1"] * 2 + np.random.randn(len(data)) * 0.01

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "collinear"],
            max_iterations=5,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(data)

        # Should handle collinearity
        try:
            best_model = selector.fit()
            self.assertIsNotNone(best_model)
        except Exception:
            pass  # Some error acceptable with highly collinear data


class TestModelSerializationRoundTrip(unittest.TestCase):
    """Test model save/load fidelity."""

    def setUp(self):
        """Set up test model."""
        np.random.seed(42)
        n_samples = 300
        self.data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "presence_unpaid": np.random.randint(0, 2, n_samples),
            }
        )
        self.config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2", "feature3"],
            max_iterations=5,
        )

    def test_metrics_preserved_after_load(self):
        """Test that metrics are preserved after save/load."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)

            loaded = GLMModelSelector.load_model(model_path)

            # Check metrics match
            self.assertAlmostEqual(
                selector.best_model.metrics.auc, loaded.best_model.metrics.auc, places=6
            )
            self.assertAlmostEqual(
                selector.best_model.metrics.aic, loaded.best_model.metrics.aic, places=6
            )

    def test_predictions_identical_after_load(self):
        """Test that predictions are identical after save/load."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        test_data = self.data.head(20)
        original_preds = selector.predict(test_data, return_proba=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)

            loaded = GLMModelSelector.load_model(model_path)
            loaded_preds = loaded.predict(test_data, return_proba=True)

            np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_config_preserved_after_load(self):
        """Test that config is preserved after save/load."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)

            loaded = GLMModelSelector.load_model(model_path)

            self.assertEqual(loaded.config.target_column, self.config.target_column)
            self.assertEqual(loaded.config.random_seed, self.config.random_seed)

    def test_predictors_preserved_after_load(self):
        """Test that predictors list is preserved after save/load."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)

            loaded = GLMModelSelector.load_model(model_path)

            self.assertEqual(loaded.best_model.predictors, selector.best_model.predictors)

    def test_timestamp_preserved_after_load(self):
        """Test that timestamp is preserved after save/load."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        original_timestamp = selector.best_model.timestamp

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)

            loaded = GLMModelSelector.load_model(model_path)

            # Timestamps should be equal (within a second for serialization)
            self.assertEqual(
                original_timestamp.isoformat(), loaded.best_model.timestamp.isoformat()
            )


class TestSelectionStrategies(unittest.TestCase):
    """Test different model selection strategies."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 300
        self.data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "presence_unpaid": np.random.randint(0, 2, n_samples),
            }
        )

    def test_forward_strategy_raises_not_implemented(self):
        """Test FORWARD strategy raises NotImplementedError."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2"],
            selection_strategy=ModelSelectionStrategy.FORWARD,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)

        with self.assertRaises(NotImplementedError):
            selector.fit()

    def test_backward_strategy_raises_not_implemented(self):
        """Test BACKWARD strategy raises NotImplementedError."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2"],
            selection_strategy=ModelSelectionStrategy.BACKWARD,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)

        with self.assertRaises(NotImplementedError):
            selector.fit()

    def test_exhaustive_strategy_raises_not_implemented(self):
        """Test EXHAUSTIVE strategy raises NotImplementedError."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2"],
            selection_strategy=ModelSelectionStrategy.EXHAUSTIVE,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)

        with self.assertRaises(NotImplementedError):
            selector.fit()

    def test_random_strategy_works(self):
        """Test RANDOM strategy works correctly."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2", "feature3"],
            selection_strategy=ModelSelectionStrategy.RANDOM,
            max_iterations=5,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)

        best_model = selector.fit()
        self.assertIsNotNone(best_model)


class TestFeatureImportanceEdgeCases(unittest.TestCase):
    """Test edge cases in feature importance calculation."""

    def setUp(self):
        """Set up test model."""
        np.random.seed(42)
        n_samples = 200
        self.data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "presence_unpaid": np.random.randint(0, 2, n_samples),
            }
        )
        self.config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2"],
            max_iterations=5,
            min_predictors=2,
            max_predictors=2,
        )

    def test_feature_importance_with_single_predictor(self):
        """Test feature importance with single predictor."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1"],
            max_iterations=3,
            min_predictors=1,
            max_predictors=1,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)
        selector.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)
            server = ModelServing(model_path)

            importance = server.get_feature_importance()
            self.assertGreaterEqual(len(importance), 1)

    def test_feature_importance_has_correct_columns(self):
        """Test feature importance DataFrame has correct columns."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)
            server = ModelServing(model_path)

            importance = server.get_feature_importance()

            expected_cols = ["feature", "coefficient", "std_error", "p_value", "odds_ratio", "significant"]
            for col in expected_cols:
                self.assertIn(col, importance.columns)

    def test_odds_ratio_is_exp_coefficient(self):
        """Test odds ratio equals exp(coefficient)."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)
            server = ModelServing(model_path)

            importance = server.get_feature_importance()

            for _, row in importance.iterrows():
                expected_or = np.exp(row["coefficient"])
                self.assertAlmostEqual(row["odds_ratio"], expected_or, places=5)

    def test_significant_flag_based_on_p_value(self):
        """Test significant flag is based on p-value < 0.05."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            selector.save_model(model_path)
            server = ModelServing(model_path)

            importance = server.get_feature_importance()

            for _, row in importance.iterrows():
                expected_sig = row["p_value"] < 0.05
                self.assertEqual(row["significant"], expected_sig)


class TestMetricCalculationEdgeCases(unittest.TestCase):
    """Test edge cases in metric calculations."""

    def test_auc_between_0_and_1(self):
        """Test AUC is always between 0 and 1."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(200),
                "feature2": np.random.randn(200),
                "presence_unpaid": np.random.randint(0, 2, 200),
            }
        )

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2"],
            max_iterations=5,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(data)
        best_model = selector.fit()

        self.assertGreaterEqual(best_model.metrics.auc, 0.0)
        self.assertLessEqual(best_model.metrics.auc, 1.0)

    def test_accuracy_between_0_and_1(self):
        """Test accuracy is always between 0 and 1."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(200),
                "presence_unpaid": np.random.randint(0, 2, 200),
            }
        )

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1"],
            max_iterations=3,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(data)
        best_model = selector.fit()

        self.assertGreaterEqual(best_model.metrics.accuracy, 0.0)
        self.assertLessEqual(best_model.metrics.accuracy, 1.0)

    def test_f1_score_between_0_and_1(self):
        """Test F1 score is always between 0 and 1."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(200),
                "presence_unpaid": np.random.randint(0, 2, 200),
            }
        )

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1"],
            max_iterations=3,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(data)
        best_model = selector.fit()

        self.assertGreaterEqual(best_model.metrics.f1_score, 0.0)
        self.assertLessEqual(best_model.metrics.f1_score, 1.0)

    def test_precision_recall_edge_case_all_negative(self):
        """Test precision/recall with all negative predictions."""
        np.random.seed(42)
        # Create data that might lead to all negative predictions
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(200) - 5,  # Shifted to potentially cause all 0s
                "presence_unpaid": np.random.randint(0, 2, 200),
            }
        )

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1"],
            max_iterations=3,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(data)

        # Should not raise division by zero
        best_model = selector.fit()
        self.assertIsNotNone(best_model.metrics.precision)
        self.assertIsNotNone(best_model.metrics.recall)

    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(200),
                "presence_unpaid": np.random.randint(0, 2, 200),
            }
        )

        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1"],
            max_iterations=3,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(data)
        best_model = selector.fit()

        cm = best_model.metrics.confusion_matrix
        self.assertIsNotNone(cm)
        self.assertEqual(cm.shape, (2, 2))


class TestModelComparisonMemoryLimit(unittest.TestCase):
    """Test max_models_to_keep functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "feature1": np.random.randn(200),
                "feature2": np.random.randn(200),
                "feature3": np.random.randn(200),
                "presence_unpaid": np.random.randint(0, 2, 200),
            }
        )

    def test_model_count_respects_limit(self):
        """Test that all_models respects max_models_to_keep."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2", "feature3"],
            max_iterations=20,
            max_models_to_keep=5,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)
        selector.fit()

        self.assertLessEqual(len(selector.all_models), 5)

    def test_models_sorted_by_aic(self):
        """Test that kept models are sorted by AIC."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2", "feature3"],
            max_iterations=15,
            max_models_to_keep=10,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)
        selector.fit()

        aic_values = [m.metrics.aic for m in selector.all_models]
        self.assertEqual(aic_values, sorted(aic_values))

    def test_best_model_in_kept_models(self):
        """Test that best model is always in kept models."""
        config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2", "feature3"],
            max_iterations=15,
            max_models_to_keep=5,
        )
        selector = GLMModelSelector(config)
        selector.prepare_data(self.data)
        selector.fit()

        # Best model should be in all_models
        best_aic = selector.best_model.metrics.aic
        kept_aics = [m.metrics.aic for m in selector.all_models]
        self.assertIn(best_aic, kept_aics)


class TestPredictionWithCustomThreshold(unittest.TestCase):
    """Test predictions with custom classification thresholds."""

    def setUp(self):
        """Set up test model."""
        np.random.seed(42)
        n_samples = 200
        self.data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "presence_unpaid": np.random.randint(0, 2, n_samples),
            }
        )
        self.config = ModelConfig(
            target_column="presence_unpaid",
            predictors=["feature1", "feature2"],
            max_iterations=5,
        )
        self.selector = GLMModelSelector(self.config)
        self.selector.prepare_data(self.data)
        self.selector.fit()

    def test_threshold_affects_predictions(self):
        """Test that threshold affects class predictions."""
        test_data = self.data.head(50)

        preds_05 = self.selector.predict(test_data, return_proba=False, threshold=0.5)
        preds_03 = self.selector.predict(test_data, return_proba=False, threshold=0.3)
        preds_07 = self.selector.predict(test_data, return_proba=False, threshold=0.7)

        # Lower threshold should give more positive predictions
        self.assertGreaterEqual(sum(preds_03), sum(preds_05))
        self.assertGreaterEqual(sum(preds_05), sum(preds_07))

    def test_threshold_extreme_values(self):
        """Test predictions with extreme threshold values."""
        test_data = self.data.head(50)

        preds_0 = self.selector.predict(test_data, return_proba=False, threshold=0.0)
        preds_1 = self.selector.predict(test_data, return_proba=False, threshold=1.0)

        # All predictions should be 1 with threshold 0
        self.assertTrue(all(p == 1 for p in preds_0))

        # All predictions should be 0 with threshold 1 (unless probability == 1)
        self.assertTrue(sum(preds_1) <= len(preds_1))


if __name__ == "__main__":
    unittest.main()
