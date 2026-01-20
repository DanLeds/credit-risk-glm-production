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


def load_credit_data(filepath: str = "data/credit.csv"):
    """Load credit data from CSV file.

    Args:
        filepath: Path to the CSV file (default: data/credit.csv)

    Returns:
        DataFrame with credit data ready for modeling
    """
    data = pd.read_csv(filepath)

    # Select numeric columns for the model
    numeric_cols = ['duration_credit', 'amount_credit', 'effort_rate',
                    'home_old', 'age', 'nb_credits', 'nb_of_dependants']
    available_numeric = [col for col in numeric_cols if col in data.columns]

    # Create a clean dataset with numeric features and target
    clean_data = data[available_numeric + ['presence_unpaid']].copy()
    clean_data = clean_data.dropna()

    print(f"Loaded {len(clean_data)} rows with columns: {list(clean_data.columns)}")
    return clean_data


def train_model():
    """Train and save the model."""
    print("Loading credit data from data/credit.csv...")
    data = load_credit_data("data/credit.csv")
    
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
    
    # Test single prediction with features from credit.csv
    test_features = {
        'duration_credit': 24,
        'amount_credit': 5000,
        'effort_rate': 3,
        'home_old': 3,
        'age': 35,
        'nb_credits': 2,
        'nb_of_dependants': 1
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
        
        # Test single prediction with features from credit.csv
        test_data = {
            "features": {
                "duration_credit": 24,
                "amount_credit": 5000,
                "effort_rate": 3,
                "home_old": 3,
                "age": 35,
                "nb_credits": 2,
                "nb_of_dependants": 1
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
