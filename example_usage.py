"""
Example usage of the GLM Model Production Framework
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path

from glm_model_production import (
    ModelConfig,
    GLMModelSelector,
    ModelSelectionStrategy,
    ModelServing
)


def generate_sample_data(n_samples=1000):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income': np.random.uniform(0, 1, n_samples),
        'months_employed': np.random.randint(0, 240, n_samples),
        'num_credit_lines': np.random.randint(0, 20, n_samples),
        'payment_history': np.random.choice([0, 1, 2], n_samples),
        'loan_amount': np.random.uniform(1000, 50000, n_samples)
    })
    
    # Create target based on features with some noise
    logits = (
        -0.005 * data['credit_score'] +
        2.0 * data['debt_to_income'] +
        -0.01 * data['months_employed'] +
        0.1 * data['num_credit_lines'] +
        0.5 * data['payment_history'] +
        0.00001 * data['loan_amount'] +
        np.random.randn(n_samples) * 0.5
    )
    
    probs = 1 / (1 + np.exp(-logits))
    data['presence_unpaid'] = (probs > 0.5).astype(int)
    
    return data


def train_model():
    """Train and save the model."""
    print("Generating sample data...")
    data = generate_sample_data(1000)
    
    print("\nConfiguring model...")
    config = ModelConfig(
        target_column='presence_unpaid',
        predictors=data.columns.difference(['presence_unpaid']).tolist(),
        max_iterations=50,
        random_seed=42,
        test_size=0.2,
        selection_strategy=ModelSelectionStrategy.RANDOM
    )
    
    print("\nInitializing model selector...")
    selector = GLMModelSelector(config)
    
    print("\nPreparing data...")
    train_data, test_data = selector.prepare_data(data)
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    print("\nTraining model (this may take a moment)...")
    best_model = selector.fit()
    
    print("\n" + "="*50)
    print("BEST MODEL RESULTS")
    print("="*50)
    print(f"Formula: {best_model.formula}")
    print(f"Number of predictors: {len(best_model.predictors)}")
    print(f"AIC: {best_model.metrics.aic:.2f}")
    print(f"BIC: {best_model.metrics.bic:.2f}")
    print(f"AUC: {best_model.metrics.auc:.4f}")
    print(f"Accuracy: {best_model.metrics.accuracy:.4f}")
    print(f"F1 Score: {best_model.metrics.f1_score:.4f}")
    
    # Save model
    model_path = Path("models") / "glm_model.joblib"
    model_path.parent.mkdir(exist_ok=True)
    
    print(f"\nSaving model to {model_path}...")
    selector.save_model(model_path)
    
    # Get model comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON (Top 5)")
    print("="*50)
    comparison = selector.get_model_comparison()
    print(comparison.head())
    
    # Get summary
    summary = selector.get_summary()
    print(f"\nTotal models evaluated: {summary['total_models_evaluated']}")
    
    return model_path


def test_api_locally(model_path):
    """Test the API locally."""
    print("\n" + "="*50)
    print("TESTING MODEL SERVING")
    print("="*50)
    
    # Load model for serving
    server = ModelServing(model_path)
    
    # Test single prediction
    test_features = {
        'credit_score': 720,
        'debt_to_income': 0.3,
        'months_employed': 60,
        'num_credit_lines': 5,
        'payment_history': 1,
        'loan_amount': 15000
    }
    
    print("\nTest features:")
    for key, value in test_features.items():
        print(f"  {key}: {value}")
    
    result = server.predict_single(test_features)
    print("\nPrediction result:")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Predicted class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    # Test batch prediction
    batch_data = pd.DataFrame([
        test_features,
        {k: v * 0.8 for k, v in test_features.items()},
        {k: v * 1.2 for k, v in test_features.items()}
    ])
    
    print("\nBatch prediction (3 samples):")
    results = server.predict_batch(batch_data)
    print(results[['predicted_probability', 'predicted_class', 'confidence']])
    
    # Get feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    importance = server.get_feature_importance()
    print(importance)


def test_api_endpoints():
    """Test the Flask API endpoints (requires API to be running)."""
    base_url = "http://localhost:5000"
    
    print("\n" + "="*50)
    print("TESTING API ENDPOINTS")
    print("="*50)
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        print(f"\nHealth check: {response.json()}")
        
        # Test model info
        response = requests.get(f"{base_url}/model/info")
        info = response.json()
        print(f"\nModel version: {info.get('version')}")
        print(f"Predictors: {info.get('predictors')}")
        
        # Test single prediction
        test_data = {
            "features": {
                "credit_score": 720,
                "debt_to_income": 0.3,
                "months_employed": 60,
                "num_credit_lines": 5,
                "payment_history": 1,
                "loan_amount": 15000
            }
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nAPI Prediction:")
            print(f"  Probability: {result['prediction']['probability']:.4f}")
            print(f"  Class: {result['prediction']['predicted_class']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\nAPI is not running. Start it with: python api_service.py")


if __name__ == "__main__":
    # Train model
    model_path = train_model()
    
    # Test serving locally
    test_api_locally(model_path)
    
    # Test API endpoints (if running)
    test_api_endpoints()
