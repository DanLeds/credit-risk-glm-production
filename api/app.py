"""
Flask API Service for GLM Model
================================
RESTful API for serving GLM predictions with monitoring and health checks.
"""

import os
import sys
import json
import logging
import hmac
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import wraps
import time
import traceback

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from werkzeug.exceptions import BadRequest

# Import our model module
from src.glm_model import ModelServing, ModelConfig, GLMModelSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configure CORS with restricted origins
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'https://app.example.com').split(',')
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# Configure caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per hour", "100 per minute"]
)

# API Key Authentication
API_KEY = os.environ.get('API_KEY')
API_KEY_HEADER = 'X-API-Key'
# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {'/', '/health', '/ready', '/metrics'}


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            # If no API key configured, skip authentication (dev mode)
            logger.warning("API_KEY not configured - authentication disabled")
            return f(*args, **kwargs)

        provided_key = request.headers.get(API_KEY_HEADER)
        if not provided_key:
            return jsonify({'error': 'Missing API key', 'header': API_KEY_HEADER}), 401

        if not hmac.compare_digest(provided_key, API_KEY):
            return jsonify({'error': 'Invalid API key'}), 403

        return f(*args, **kwargs)
    return decorated

# Prometheus metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['model_version', 'endpoint']
)
prediction_latency = Histogram(
    'model_prediction_duration_seconds',
    'Prediction latency in seconds',
    ['endpoint']
)
model_accuracy_gauge = Gauge(
    'model_accuracy',
    'Current model accuracy'
)
active_requests_gauge = Gauge(
    'active_requests',
    'Number of active requests'
)
error_counter = Counter(
    'model_errors_total',
    'Total number of errors',
    ['error_type']
)

# Global model server instance
model_server: Optional[ModelServing] = None
model_info: Dict[str, Any] = {}


class ModelNotLoadedError(Exception):
    """Exception raised when model is not loaded."""
    pass


def track_performance(endpoint: str):
    """Decorator to track API performance metrics."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            active_requests_gauge.inc()
            start_time = time.time()
            
            try:
                result = f(*args, **kwargs)
                prediction_counter.labels(
                    model_version=model_info.get('version', 'unknown'),
                    endpoint=endpoint
                ).inc()
                return result
                
            except Exception as e:
                error_counter.labels(error_type=type(e).__name__).inc()
                raise
                
            finally:
                duration = time.time() - start_time
                prediction_latency.labels(endpoint=endpoint).observe(duration)
                active_requests_gauge.dec()
                
        return wrapped
    return decorator


def validate_request_data(schema: Dict[str, type]):
    """Decorator to validate request JSON data."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
                
            data = request.get_json()
            
            # Validate required fields
            for field, field_type in schema.items():
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
                    
                # Basic type checking
                if not isinstance(data[field], field_type):
                    return jsonify({
                        'error': f'Invalid type for field {field}. Expected {field_type.__name__}'
                    }), 400
                    
            return f(*args, **kwargs)
        return wrapped
    return decorator


def load_model(model_path: str) -> None:
    """
    Load the model from disk.
    
    Args:
        model_path: Path to the saved model file
    """
    global model_server, model_info
    
    try:
        logger.info(f"Loading model from {model_path}")
        model_server = ModelServing(model_path)
        
        # Extract model information
        model_info = {
            'path': model_path,
            'version': os.environ.get('MODEL_VERSION', '1.0.0'),
            'loaded_at': datetime.now().isoformat(),
            'predictors': model_server.predictors,
            'metrics': model_server.selector.best_model.metrics.to_dict()
        }
        
        # Update accuracy metric
        model_accuracy_gauge.set(model_info['metrics']['accuracy'])
        
        logger.info(f"Model loaded successfully. Version: {model_info['version']}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.before_request
def check_model_loaded():
    """Ensure model is loaded before processing requests."""
    # Skip for health and metrics endpoints
    if request.path in ['/health', '/ready', '/metrics', '/']:
        return None
        
    if model_server is None:
        return jsonify({'error': 'Model not loaded'}), 503


@app.route('/', methods=['GET'])
def home():
    """API root endpoint."""
    return jsonify({
        'service': 'GLM Model Prediction API',
        'version': model_info.get('version', 'unknown'),
        'status': 'running' if model_server else 'model not loaded',
        'endpoints': {
            'health': '/health',
            'ready': '/ready',
            'predict': '/predict',
            'predict_batch': '/predict/batch',
            'model_info': '/model/info',
            'feature_importance': '/model/features',
            'metrics': '/metrics'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check endpoint."""
    if model_server is None:
        return jsonify({'status': 'not ready', 'reason': 'model not loaded'}), 503
    return jsonify({'status': 'ready'}), 200


@app.route('/predict', methods=['POST'])
@limiter.limit("100 per minute")
@require_api_key
@track_performance('predict_single')
@validate_request_data({'features': dict})
def predict_single():
    """
    Single prediction endpoint.
    
    Request body:
        {
            "features": {
                "feature1": value1,
                "feature2": value2,
                ...
            }
        }
    
    Response:
        {
            "prediction": {
                "probability": 0.75,
                "predicted_class": 1,
                "confidence": 0.75,
                "predictors_used": ["feature1", "feature2"]
            },
            "model_version": "1.0.0",
            "timestamp": "2024-01-01T10:00:00"
        }
    """
    try:
        data = request.get_json()
        features = data['features']
        
        # Validate features
        missing_features = set(model_server.predictors) - set(features.keys())
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {list(missing_features)}'
            }), 400
        
        # Make prediction
        prediction = model_server.predict_single(features)
        
        response = {
            'prediction': prediction,
            'model_version': model_info.get('version', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        logger.warning(f"Prediction validation error: {str(e)}")
        return jsonify({'error': 'Invalid input', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed'}), 500


@app.route('/predict/batch', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
@track_performance('predict_batch')
def predict_batch():
    """
    Batch prediction endpoint.
    
    Request body:
        {
            "data": [
                {"feature1": value1, "feature2": value2, ...},
                {"feature1": value1, "feature2": value2, ...},
                ...
            ],
            "include_confidence": true
        }
    
    Response:
        {
            "predictions": [
                {
                    "index": 0,
                    "probability": 0.75,
                    "predicted_class": 1,
                    "confidence": 0.75
                },
                ...
            ],
            "model_version": "1.0.0",
            "timestamp": "2024-01-01T10:00:00",
            "total_predictions": 100
        }
    """
    try:
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'Missing required field: data'}), 400
            
        batch_data = data['data']
        include_confidence = data.get('include_confidence', True)
        
        # Validate batch size
        max_batch_size = int(os.environ.get('MAX_BATCH_SIZE', 1000))
        if len(batch_data) > max_batch_size:
            return jsonify({
                'error': f'Batch size exceeds maximum of {max_batch_size}'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # Validate features
        missing_features = set(model_server.predictors) - set(df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {list(missing_features)}'
            }), 400
        
        # Make predictions
        results = model_server.predict_batch(df, include_confidence=include_confidence)

        # Format response - optimized without iterrows()
        predictions = results.reset_index().rename(columns={'index': 'idx'}).to_dict('records')
        predictions = [
            {
                'index': i,
                'probability': float(p['predicted_probability']),
                'predicted_class': int(p['predicted_class']),
                **(({'confidence': float(p['confidence'])} if include_confidence else {}))
            }
            for i, p in enumerate(predictions)
        ]

        response = {
            'predictions': predictions,
            'model_version': model_info.get('version', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions)
        }

        return jsonify(response), 200

    except ValueError as e:
        logger.warning(f"Batch validation error: {str(e)}")
        return jsonify({'error': 'Invalid input', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Batch prediction failed'}), 500


@app.route('/model/info', methods=['GET'])
@cache.cached(timeout=300)
def get_model_info():
    """Get information about the loaded model."""
    return jsonify(model_info), 200


@app.route('/model/features', methods=['GET'])
@cache.cached(timeout=300)
def get_feature_importance():
    """Get feature importance from the model."""
    try:
        importance_df = model_server.get_feature_importance()
        
        features = []
        for _, row in importance_df.iterrows():
            features.append({
                'feature': row['feature'],
                'coefficient': float(row['coefficient']),
                'p_value': float(row['p_value']),
                'odds_ratio': float(row['odds_ratio']),
                'significant': bool(row['significant'])
            })
        
        return jsonify({
            'features': features,
            'total_features': len(features)
        }), 200
        
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        return jsonify({'error': 'Failed to get feature importance'}), 500


@app.route('/model/reload', methods=['POST'])
@limiter.limit("1 per minute")
@require_api_key
def reload_model():
    """
    Reload the model from disk.

    Request body:
        {
            "model_path": "/path/to/model.joblib"  # Optional, must be within ALLOWED_MODEL_DIR
        }
    """
    try:
        data = request.get_json() or {}
        model_path = data.get('model_path', os.environ.get('MODEL_PATH', 'model/glm_model.joblib'))

        # Security: Validate model path to prevent path traversal attacks
        allowed_model_dir = Path(os.environ.get('ALLOWED_MODEL_DIR', '/app/models')).resolve()
        requested_path = Path(model_path).resolve()

        # Check if the requested path is within the allowed directory
        if not str(requested_path).startswith(str(allowed_model_dir)):
            logger.warning(f"Attempted path traversal attack: {model_path}")
            return jsonify({
                'error': 'Invalid model path',
                'message': 'Model path must be within the allowed models directory'
            }), 403

        # Check file extension
        if not str(requested_path).endswith('.joblib'):
            return jsonify({
                'error': 'Invalid model file',
                'message': 'Model file must have .joblib extension'
            }), 400

        load_model(str(requested_path))

        return jsonify({
            'status': 'success',
            'message': 'Model reloaded successfully',
            'model_info': model_info
        }), 200

    except FileNotFoundError:
        return jsonify({
            'error': 'Model not found',
            'message': 'The specified model file does not exist'
        }), 404
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        return jsonify({
            'error': 'Failed to reload model'
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype='text/plain')


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


def create_app(model_path: Optional[str] = None) -> Flask:
    """
    Application factory.
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        Flask application instance
    """
    if model_path:
        load_model(model_path)
    elif 'MODEL_PATH' in os.environ:
        load_model(os.environ['MODEL_PATH'])
        
    return app


if __name__ == '__main__':
    # Load model from environment variable or default path
    model_path = os.environ.get('MODEL_PATH', 'model/glm_model.joblib')
    
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        logger.warning(f"Model file not found at {model_path}. API will start without model.")
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
