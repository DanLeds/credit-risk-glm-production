"""
GLM Model Selection and Deployment Framework
=============================================
A production-ready framework for GLM model selection, training, and serving.
"""
import json
import logging
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelSelectionStrategy(Enum):
    """Strategy for model selection."""
    RANDOM = "random"
    EXHAUSTIVE = "exhaustive"
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class ModelConfig:
    """Configuration for GLM model selection."""
    target_column: str = "presence_unpaid"
    predictors: List[str] = field(default_factory=list)
    max_iterations: int = 100
    random_seed: int = 42
    test_size: float = 0.2
    min_predictors: int = 1
    max_predictors: Optional[int] = None
    selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.RANDOM
    confidence_level: float = 0.95
    max_models_to_keep: int = 50  # Limit memory usage by keeping only top N models

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.min_predictors <= 0:
            raise ValueError("min_predictors must be positive")
        if self.max_predictors and self.max_predictors < self.min_predictors:
            raise ValueError("max_predictors must be >= min_predictors")
        if self.max_models_to_keep <= 0:
            raise ValueError("max_models_to_keep must be positive")


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""
    aic: float
    bic: float
    auc: float
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    log_likelihood: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    roc_curve: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)
        if self.confusion_matrix is not None:
            data['confusion_matrix'] = self.confusion_matrix.tolist()
        return data


@dataclass
class ModelResult:
    """Result of a fitted model."""
    formula: str
    predictors: List[str]
    model: Any
    metrics: ModelMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[ModelConfig] = None


class DataValidator:
    """Validator for input data."""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        target_column: str,
        predictors: List[str]
    ) -> None:
        """
        Validate input DataFrame for model training.

        Args:
            df: Input DataFrame
            target_column: Name of the target column
            predictors: List of predictor column names

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        missing_predictors = set(predictors) - set(df.columns)
        if missing_predictors:
            raise ValueError(f"Predictors not found: {missing_predictors}")

        null_counts = df[predictors + [target_column]].isnull().sum()
        if null_counts.any():
            logger.warning(f"Missing values detected:\n{null_counts[null_counts > 0]}")

        unique_targets = df[target_column].unique()
        if len(unique_targets) != 2:
            raise ValueError(f"Target must be binary, found {len(unique_targets)} unique values")

        constant_cols = [col for col in predictors if df[col].nunique() == 1]
        if constant_cols:
            logger.warning(f"Constant predictors detected: {constant_cols}")


class GLMModelSelector:
    """
    GLM Model Selector for credit scoring.

    This class handles model selection, training, evaluation, and persistence
    for Generalized Linear Models used in credit scoring applications.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the model selector.

        Args:
            config: Model configuration
        """
        config.validate()
        self.config = config
        self.best_model: Optional[ModelResult] = None
        self.all_models: List[ModelResult] = []
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    def prepare_data(
        self,
        data: pd.DataFrame,
        train_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training.

        Args:
            data: Full dataset (used if train/test not provided)
            train_data: Optional pre-split training data
            test_data: Optional pre-split test data

        Returns:
            Tuple of (train_data, test_data)
        """
        if train_data is not None and test_data is not None:
            self.train_data = train_data.copy()
            self.test_data = test_data.copy()
        else:
            self.train_data, self.test_data = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=data[self.config.target_column]
            )

        DataValidator.validate_dataframe(
            self.train_data,
            self.config.target_column,
            self.config.predictors
        )
        DataValidator.validate_dataframe(
            self.test_data,
            self.config.target_column,
            self.config.predictors
        )

        logger.info(f'Data prepared: {len(self.train_data)} train, {len(self.test_data)} test samples')
        return self.train_data, self.test_data

    def _fit_model(
        self,
        predictors: List[str],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> ModelResult:
        """
        Fit a single GLM model with given predictors.

        Args:
            predictors: List of predictor names
            train_data: Training data
            test_data: Test data

        Returns:
            ModelResult with fitted model and metrics
        """
        formula = f"{self.config.target_column} ~ {' + '.join(predictors)}"

        try:
            # Fit model
            model = smf.glm(
                formula=formula,
                data=train_data,
                family=sm.families.Binomial()
            ).fit()

            # Predictions
            y_test = test_data[self.config.target_column]
            predicted_probs = model.predict(test_data)

            # Calculate metrics
            auc = roc_auc_score(y_test, predicted_probs)

            threshold = 0.5
            predicted_classes = (predicted_probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_classes).ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, predicted_probs)

            metrics = ModelMetrics(
                aic=model.aic,
                bic=model.bic_llf,
                auc=auc,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                log_likelihood=model.llf,
                confusion_matrix=confusion_matrix(y_test, predicted_classes),
                roc_curve={
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            )

            return ModelResult(
                formula=formula,
                predictors=predictors,
                model=model,
                metrics=metrics,
                config=self.config
            )

        except Exception as e:
            logger.error(f"Failed to fit model with predictors {predictors}: {str(e)}")
            raise

    def _random_search(self) -> ModelResult:
        """
        Perform random search for best model.

        Returns:
            Best ModelResult found
        """
        best_aic = float('inf')
        best_model = None

        for iteration in range(self.config.max_iterations):
            # Random number of predictors
            max_k = self.config.max_predictors or len(self.config.predictors)
            k = random.randint(
                self.config.min_predictors,
                min(max_k, len(self.config.predictors))
            )

            # Random selection of predictors
            selected_predictors = random.sample(self.config.predictors, k)

            # Fit model
            try:
                model_result = self._fit_model(
                    selected_predictors,
                    self.train_data,
                    self.test_data
                )

                self.all_models.append(model_result)

                # Limit memory usage: keep only top N models by AIC
                if len(self.all_models) > self.config.max_models_to_keep:
                    self.all_models.sort(key=lambda m: m.metrics.aic)
                    self.all_models = self.all_models[:self.config.max_models_to_keep]

                # Update best model
                if model_result.metrics.aic < best_aic:
                    best_aic = model_result.metrics.aic
                    best_model = model_result
                    logger.info(
                        f"Iteration {iteration + 1}: New best model found "
                        f"(AIC={best_aic:.2f}, AUC={model_result.metrics.auc:.4f})"
                    )

            except Exception as e:
                logger.warning(f"Iteration {iteration + 1} failed: {str(e)}")
                continue

        if best_model is None:
            raise ValueError(
                f"No valid model found after {self.config.max_iterations} iterations"
            )

        logger.info(
            f"Random search completed: Best AIC={best_model.metrics.aic:.2f}, "
            f"AUC={best_model.metrics.auc:.4f}, "
            f"Variables={best_model.predictors}"
        )

        return best_model

    def fit(self) -> ModelResult:
        """
        Fit model using configured selection strategy.

        Returns:
            Best ModelResult
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data must be prepared before fitting")

        logger.info(f"Starting model selection with strategy: {self.config.selection_strategy.value}")

        if self.config.selection_strategy == ModelSelectionStrategy.RANDOM:
            self.best_model = self._random_search()
        else:
            raise NotImplementedError(f"Strategy {self.config.selection_strategy} not implemented")

        if self.best_model is None:
            raise RuntimeError("No valid model found")

        logger.info(
            f"Best model selected with {len(self.best_model.predictors)} predictors, "
            f"AIC={self.best_model.metrics.aic:.2f}, AUC={self.best_model.metrics.auc:.4f}"
        )

        return self.best_model

    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = True,
        threshold: float = 0.5,
        return_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions with the fitted model.

        Args:
            X: Input features
            return_proba: Whether to return probabilities
            threshold: Classification threshold
            return_dataframe: Whether to return DataFrame with full results

        Returns:
            Predictions as array or DataFrame
        """
        if self.best_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        missing_cols = set(self.best_model.predictors) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required: {self.best_model.predictors}"
            )

        X_filtered = X[self.best_model.predictors]
        probabilities = self.best_model.model.predict(X_filtered)

        if return_dataframe:
            return pd.DataFrame({
                'proba_default': probabilities,
                'predicted_class': (probabilities >= threshold).astype(int),
                'decision': ['REFUSE' if p >= threshold else 'ACCEPT' for p in probabilities]
            }, index=X.index)

        if return_proba:
            return probabilities

        return (probabilities >= threshold).astype(int)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self.config)
        config_dict['selection_strategy'] = self.config.selection_strategy.value

        model_data = {
            'model': self.best_model.model,
            'formula': self.best_model.formula,
            'predictors': self.best_model.predictors,
            'metrics': self.best_model.metrics.to_dict(),
            'config': config_dict,
            'timestamp': self.best_model.timestamp.isoformat(),
            'version': '1.0.0'
        }

        joblib.dump(model_data, filepath, compress=3)

        file_size = filepath.stat().st_size / 1024
        logger.info(
            f"Model saved to {filepath} "
            f"({file_size:.1f} KB, AUC={self.best_model.metrics.auc:.4f})"
        )

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'GLMModelSelector':
        """
        Load a model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            GLMModelSelector instance with loaded model
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        logger.info(f"Loading model from {filepath}")

        config_dict = model_data['config'].copy()
        if 'selection_strategy' in config_dict and isinstance(config_dict['selection_strategy'], str):
            config_dict['selection_strategy'] = ModelSelectionStrategy(config_dict['selection_strategy'])

        config = ModelConfig(**config_dict)
        selector = cls(config)

        metrics_dict = model_data['metrics']
        if 'confusion_matrix' in metrics_dict and metrics_dict['confusion_matrix'] is not None:
            metrics_dict['confusion_matrix'] = np.array(metrics_dict['confusion_matrix'])

        metrics = ModelMetrics(**{
            k: v for k, v in metrics_dict.items()
            if k in ModelMetrics.__annotations__
        })

        selector.best_model = ModelResult(
            formula=model_data['formula'],
            predictors=model_data['predictors'],
            model=model_data['model'],
            metrics=metrics,
            timestamp=datetime.fromisoformat(model_data['timestamp']),
            config=config
        )

        logger.info(
            f"Model loaded from {filepath} - "
            f"AUC={selector.best_model.metrics.auc:.4f}, "
            f"trained on {selector.best_model.timestamp.strftime('%Y-%m-%d')}"
        )

        return selector

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of model selection results.

        Returns:
            Dictionary with model summary
        """
        config_dict = asdict(self.config)
        config_dict['selection_strategy'] = self.config.selection_strategy.value

        if self.best_model is None:
            return {"status": "No model fitted"}

        all_auc = [m.metrics.auc for m in self.all_models]
        all_aic = [m.metrics.aic for m in self.all_models]

        return {
            "best_model": {
                "formula": self.best_model.formula,
                "predictors": self.best_model.predictors,
                "metrics": self.best_model.metrics.to_dict(),
                "timestamp": self.best_model.timestamp.isoformat()
            },
            "version": "1.0.0",
            "total_models_evaluated": len(self.all_models),
            "search_statistics": {
                "auc_mean": float(np.mean(all_auc)),
                "auc_std": float(np.std(all_auc)),
                "auc_min": float(np.min(all_auc)),
                "auc_max": float(np.max(all_auc)),
                "aic_mean": float(np.mean(all_aic)),
                "aic_min": float(np.min(all_aic)),
                "aic_max": float(np.max(all_aic))
            },
            "config": config_dict
        }

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison DataFrame of all evaluated models.

        Returns:
            DataFrame with model comparison metrics
        """
        if not self.all_models:
            return pd.DataFrame()

        comparison_data = []
        for model in self.all_models:
            comparison_data.append({
                'num_predictors': len(model.predictors),
                'predictors': ', '.join(model.predictors),
                'aic': model.metrics.aic,
                'bic': model.metrics.bic,
                'auc': model.metrics.auc,
                'accuracy': model.metrics.accuracy,
                'f1_score': model.metrics.f1_score
            })

        df = pd.DataFrame(comparison_data)
        return df.sort_values('aic')


class ModelServing:
    """Service class for model inference in production."""

    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize model serving.

        Args:
            model_path: Path to the saved model
        """
        self.selector = GLMModelSelector.load_model(model_path)
        self.model = self.selector.best_model.model
        self.predictors = self.selector.best_model.predictors

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single instance.

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with prediction results
        """
        df = pd.DataFrame([features])
        probability = float(self.selector.predict(df, return_proba=True)[0])
        predicted_class = int(probability >= 0.5)

        return {
            'probability': probability,
            'predicted_class': predicted_class,
            'confidence': max(probability, 1 - probability),
            'predictors_used': self.predictors
        }

    def predict_batch(
        self,
        data: pd.DataFrame,
        include_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of instances.

        Args:
            data: DataFrame with features
            include_confidence: Whether to include confidence scores

        Returns:
            DataFrame with predictions
        """
        results = data.copy()
        probabilities = self.selector.predict(data, return_proba=True)
        results['predicted_probability'] = probabilities
        results['predicted_class'] = (probabilities >= 0.5).astype(int)

        if include_confidence:
            results['confidence'] = np.maximum(probabilities, 1 - probabilities)

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Returns:
            DataFrame with feature importance metrics

        Raises:
            RuntimeError: If feature importance cannot be extracted
        """
        try:
            summary = self.model.summary2().tables[1]

            importance_df = pd.DataFrame({
                'feature': summary.index[1:],
                'coefficient': summary['Coef.'].values[1:],
                'std_error': summary['Std.Err.'].values[1:],
                'p_value': summary['P>|z|'].values[1:],
                'significant': summary['P>|z|'].values[1:] < 0.05
            })

            importance_df['odds_ratio'] = np.exp(importance_df['coefficient'])

            return importance_df.sort_values('p_value')

        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to extract feature importance: {e}")
            raise RuntimeError(f"Could not extract feature importance: {e}") from e


def main_example():
    """Example usage of the production pipeline."""
    print("=" * 60)
    print("CREDIT SCORING PIPELINE - EXAMPLE")
    print("=" * 60)

    # 1. Configuration
    print("\n[1/8] Model configuration...")
    config = ModelConfig(
        target_column="presence_unpaid",
        max_iterations=100,
        random_seed=42,
        test_size=0.2,
        min_predictors=1,
        selection_strategy=ModelSelectionStrategy.RANDOM
    )
    print(f"Config created: {config.max_iterations} iterations, seed={config.random_seed}")

    # 2. Initialize selector
    print("\n[2/8] Selector initialization...")
    selector = GLMModelSelector(config)
    print("Selector initialized")

    # 3. Load and prepare data
    print("\n[3/8] Loading and preparing data...")
    data = pd.read_csv("data/credit.csv")
    print(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")

    # Select numeric columns for the model
    numeric_cols = ['duration_credit', 'amount_credit', 'effort_rate',
                    'home_old', 'age', 'nb_credits', 'nb_of_dependants']
    available_numeric = [col for col in numeric_cols if col in data.columns]

    # Keep only numeric features and target
    data = data[available_numeric + ['presence_unpaid']].dropna()
    config.predictors = available_numeric
    print(f"Predictive variables: {config.predictors}")
    train_data, test_data = selector.prepare_data(data)
    print(f"Split: {len(train_data)} train, {len(test_data)} test")

    # 4. Fit model
    print("\n[4/8] Model training (random search)...")
    start = time.time()
    best_model = selector.fit()
    duration = time.time() - start
    print(f"Training completed in {duration:.1f}s")
    print(f"  Variables selected: {best_model.predictors}")
    print(f"  AIC: {best_model.metrics.aic:.2f}")
    print(f"  AUC: {best_model.metrics.auc:.4f}")

    # 5. Save model
    print("\n[5/8] Saving model...")
    filepath = "models/best_glm_model.joblib"
    selector.save_model(filepath)
    size_kb = Path(filepath).stat().st_size / 1024
    print(f"Model saved: {filepath} ({size_kb:.1f} KB)")

    # 6. Get summary
    print("\n[6/8] Generating summary...")
    summary = selector.get_summary()

    print("\n=== BEST MODEL SUMMARY ===")
    print(f"Training date: {summary['best_model']['timestamp']}")
    print(f"Models tested: {summary['total_models_evaluated']}")
    print(f"Variables: {', '.join(summary['best_model']['predictors'])}")
    print("\nPerformance:")
    metrics = summary['best_model']['metrics']
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")

    # Save summary
    Path('reports').mkdir(exist_ok=True)
    with open('reports/model_summary.json', 'w') as f:
        f.write(json.dumps(summary, indent=2, default=str))
    print("\nSummary saved: reports/model_summary.json")

    # 7. Model comparison
    print("\n[7/8] Model comparison...")
    comparison = selector.get_model_comparison()
    print("\n=== TOP 10 MODELS ===")
    print(comparison.head(10)[['num_predictors', 'aic', 'auc', 'f1_score']])

    comparison.to_csv('reports/model_comparison.csv', index=False)
    print("\nComparison saved: reports/model_comparison.csv")

    # 8. Test serving
    print("\n[8/8] Testing model serving...")
    server = ModelServing(filepath)

    # Test client
    if len(best_model.predictors) >= 4:
        client_test = {
            best_model.predictors[0]: 35,
            best_model.predictors[1]: 45000,
            best_model.predictors[2]: 12000,
            best_model.predictors[3]: 2
        }
    else:
        client_test = {p: 1 for p in best_model.predictors}

    prediction = server.predict_single(client_test)

    print("\n=== PREDICTION TEST ===")
    print(f"  Client: {client_test}")
    print(f"  Default probability: {prediction['probability']:.2%}")
    print(f"  Predicted class: {prediction['predicted_class']} "
          f"({'Default' if prediction['predicted_class'] == 1 else 'Good payer'})")
    print(f"  Confidence: {prediction['confidence']:.2%}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {filepath}")
    print("  - reports/model_summary.json")
    print("  - reports/model_comparison.csv")


if __name__ == "__main__":
    main_example()
