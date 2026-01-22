"""
Unit tests for GLM Model Selection Framework
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import json

from src.glm_model import (
    ModelConfig, ModelSelectionStrategy, DataValidator,
    GLMModelSelector, ModelServing, ModelMetrics, ModelResult
)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        self.assertEqual(config.target_column, "presence_unpaid")
        self.assertEqual(config.max_iterations, 100)
        self.assertEqual(config.test_size, 0.2)
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid test_size
        config = ModelConfig(test_size=1.5)
        with self.assertRaises(ValueError):
            config.validate()
            
        # Invalid max_iterations
        config = ModelConfig(max_iterations=-1)
        with self.assertRaises(ValueError):
            config.validate()
            
        # Invalid predictor range
        config = ModelConfig(min_predictors=5, max_predictors=3)
        with self.assertRaises(ValueError):
            config.validate()


class TestDataValidator(unittest.TestCase):
    """Test DataValidator class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.validator = DataValidator()
        
        # Create sample data
        self.df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
    def test_valid_dataframe(self):
        """Test validation with valid DataFrame."""
        # Should not raise any exceptions
        self.validator.validate_dataframe(
            self.df,
            'target',
            ['feature1', 'feature2']
        )
        
    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_dataframe(
                pd.DataFrame(),
                'target',
                ['feature1']
            )
        self.assertIn("empty", str(context.exception).lower())
        
    def test_missing_target(self):
        """Test validation with missing target column."""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_dataframe(
                self.df,
                'nonexistent',
                ['feature1']
            )
        self.assertIn("not found", str(context.exception))
        
    def test_missing_predictors(self):
        """Test validation with missing predictor columns."""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_dataframe(
                self.df,
                'target',
                ['feature1', 'nonexistent']
            )
        self.assertIn("not found", str(context.exception))
        
    def test_non_binary_target(self):
        """Test validation with non-binary target."""
        df = self.df.copy()
        df['target'] = np.random.randint(0, 5, 100)
        
        with self.assertRaises(ValueError) as context:
            self.validator.validate_dataframe(
                df,
                'target',
                ['feature1']
            )
        self.assertIn("binary", str(context.exception).lower())


class TestGLMModelSelector(unittest.TestCase):
    """Test GLMModelSelector class."""
    
    def setUp(self):
        """Set up test data and configuration."""
        np.random.seed(42)
        
        # Create synthetic data
        n_samples = 500
        self.data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
            'feature5': np.random.randn(n_samples)
        })
        
        # Create target with some correlation to features
        logits = (
            0.5 * self.data['feature1'] +
            0.3 * self.data['feature2'] -
            0.2 * self.data['feature3'] +
            np.random.randn(n_samples) * 0.5
        )
        probs = 1 / (1 + np.exp(-logits))
        self.data['presence_unpaid'] = (probs > 0.5).astype(int)
        
        # Configuration
        self.config = ModelConfig(
            target_column='presence_unpaid',
            predictors=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            max_iterations=10,
            random_seed=42,
            test_size=0.2
        )
        
    def test_initialization(self):
        """Test GLMModelSelector initialization."""
        selector = GLMModelSelector(self.config)
        self.assertIsNone(selector.best_model)
        self.assertEqual(len(selector.all_models), 0)
        
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
        
    def test_prediction(self):
        """Test prediction functionality."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()
        
        # Test probability predictions
        test_data = self.data.head(10)
        probabilities = selector.predict(test_data, return_proba=True)
        
        self.assertEqual(len(probabilities), len(test_data))
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))
        
        # Test class predictions
        classes = selector.predict(test_data, return_proba=False)
        self.assertTrue(all(c in [0, 1] for c in classes))
        
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
            self.assertEqual(
                loaded_selector.best_model.predictors,
                selector.best_model.predictors
            )
            
            # Test predictions with loaded model
            test_data = self.data.head(10)
            original_preds = selector.predict(test_data)
            loaded_preds = loaded_selector.predict(test_data)
            
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)
            
    def test_model_comparison(self):
        """Test model comparison functionality."""
        selector = GLMModelSelector(self.config)
        selector.prepare_data(self.data)
        selector.fit()
        
        comparison_df = selector.get_model_comparison()
        
        # Check DataFrame structure
        self.assertFalse(comparison_df.empty)
        self.assertIn('aic', comparison_df.columns)
        self.assertIn('auc', comparison_df.columns)
        self.assertIn('num_predictors', comparison_df.columns)
        
        # Check that models are sorted by AIC
        aic_values = comparison_df['aic'].values
        self.assertTrue(all(aic_values[i] <= aic_values[i+1] 
                           for i in range(len(aic_values)-1)))


class TestModelServing(unittest.TestCase):
    """Test ModelServing class."""
    
    def setUp(self):
        """Set up test model."""
        np.random.seed(42)
        
        # Create and train a simple model
        n_samples = 200
        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'presence_unpaid': np.random.randint(0, 2, n_samples)
        })
        
        config = ModelConfig(
            target_column='presence_unpaid',
            predictors=['feature1', 'feature2'],
            max_iterations=5,
            random_seed=42,
            min_predictors=2,
            max_predictors=2
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
        self.assertEqual(server.predictors, ['feature1', 'feature2'])
        
    def test_single_prediction(self):
        """Test single prediction."""
        server = ModelServing(self.model_path)
        
        features = {'feature1': 0.5, 'feature2': -0.3}
        result = server.predict_single(features)
        
        # Check result structure
        self.assertIn('probability', result)
        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('predictors_used', result)
        
        # Check value ranges
        self.assertGreaterEqual(result['probability'], 0.0)
        self.assertLessEqual(result['probability'], 1.0)
        self.assertIn(result['predicted_class'], [0, 1])
        
    def test_batch_prediction(self):
        """Test batch prediction."""
        server = ModelServing(self.model_path)
        
        batch_data = pd.DataFrame({
            'feature1': [0.5, -0.3, 1.2],
            'feature2': [-0.3, 0.8, -1.5]
        })
        
        results = server.predict_batch(batch_data)
        
        # Check result structure
        self.assertEqual(len(results), len(batch_data))
        self.assertIn('predicted_probability', results.columns)
        self.assertIn('predicted_class', results.columns)
        self.assertIn('confidence', results.columns)
        
    def test_feature_importance(self):
        """Test feature importance extraction."""
        server = ModelServing(self.model_path)
        importance_df = server.get_feature_importance()
        
        # Check DataFrame structure
        self.assertFalse(importance_df.empty)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('coefficient', importance_df.columns)
        self.assertIn('p_value', importance_df.columns)
        self.assertIn('odds_ratio', importance_df.columns)


class TestAPIEndpoints(unittest.TestCase):
    """Test Flask API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        from api.app import app, load_model
        
        # Create test model
        np.random.seed(42)
        n_samples = 200
        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'presence_unpaid': np.random.randint(0, 2, n_samples)
        })
        
        config = ModelConfig(
            target_column='presence_unpaid',
            predictors=['feature1', 'feature2'],
            max_iterations=5
        )
        
        selector = GLMModelSelector(config)
        selector.prepare_data(data)
        selector.fit()
        
        # Save model
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.joblib"
        selector.save_model(self.model_path)
        
        # Load model in API
        load_model(str(self.model_path))
        
        # Create test client
        self.app = app
        self.client = app.test_client()
        
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        
    def test_single_prediction_endpoint(self):
        """Test single prediction endpoint."""
        payload = {
            'features': {
                'feature1': 0.5,
                'feature2': -0.3
            }
        }
        
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('prediction', data)
        self.assertIn('probability', data['prediction'])
        self.assertIn('predicted_class', data['prediction'])
        
    def test_batch_prediction_endpoint(self):
        """Test batch prediction endpoint."""
        payload = {
            'data': [
                {'feature1': 0.5, 'feature2': -0.3},
                {'feature1': -0.2, 'feature2': 0.8}
            ]
        }
        
        response = self.client.post(
            '/predict/batch',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('predictions', data)
        self.assertEqual(len(data['predictions']), 2)
        
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = self.client.get('/model/info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('predictors', data)
        self.assertIn('metrics', data)
        
    def test_feature_importance_endpoint(self):
        """Test feature importance endpoint."""
        response = self.client.get('/model/features')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('features', data)
        self.assertIn('total_features', data)


if __name__ == '__main__':
    unittest.main()
