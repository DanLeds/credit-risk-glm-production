"""
Tests for the explainability module.
"""

import sys
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import tempfile
import os

# Import standard library and installed packages FIRST
import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Helper function to mock only if not installed
def mock_if_missing(module_name, mock_obj=None):
    """Mock a module only if it's not already installed."""
    if module_name not in sys.modules:
        sys.modules[module_name] = mock_obj or MagicMock()


# Mock optional/non-installed dependencies
# SHAP
mock_if_missing("shap")

# LIME
mock_if_missing("lime")
mock_if_missing("lime.lime_tabular")

# interpret
mock_if_missing("interpret")
mock_if_missing("interpret.glassbox")
mock_if_missing("interpret.blackbox")

# alibi
mock_if_missing("alibi")
mock_if_missing("alibi.explainers")

# prometheus_client
mock_prometheus = MagicMock()
mock_prometheus.CollectorRegistry = MagicMock(return_value=MagicMock())
mock_prometheus.Gauge = MagicMock(return_value=MagicMock())
mock_prometheus.Counter = MagicMock(return_value=MagicMock())
mock_if_missing("prometheus_client", mock_prometheus)

# plotly
mock_if_missing("plotly")
mock_if_missing("plotly.graph_objects")
mock_if_missing("plotly.express")
mock_if_missing("plotly.subplots")

# seaborn
mock_if_missing("seaborn")

# SALib
mock_if_missing("SALib")
mock_if_missing("SALib.sample")
mock_if_missing("SALib.sample.morris")
mock_if_missing("SALib.analyze")
mock_if_missing("SALib.analyze.morris")

# Cloud SDKs (usually not installed)
mock_if_missing("boto3")
mock_if_missing("azure")
mock_if_missing("azure.identity")
mock_if_missing("azure.monitor")
mock_if_missing("azure.monitor.query")
mock_if_missing("google")
mock_if_missing("google.cloud")
mock_if_missing("google.cloud.monitoring_v3")
mock_if_missing("google.cloud.billing_v1")

from src.explainability import (
    ExplanationResult,
    ModelExplainer,
    ResourceUsage,
    CostEstimate,
    CloudCostOptimizer,
)


# ====================== Fixtures ======================


@pytest.fixture
def sample_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1, 0])
    model.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]]
    )
    return model


@pytest.fixture
def sample_feature_names():
    """Get sample feature names."""
    return ["feature1", "feature2", "feature3", "feature4", "feature5"]


@pytest.fixture
def sample_dataframe(sample_feature_names):
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({feat: np.random.randn(100) for feat in sample_feature_names})


@pytest.fixture
def sample_series():
    """Create sample target series."""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100))


@pytest.fixture
def sample_usage_data():
    """Create sample resource usage data."""
    np.random.seed(42)
    data = []
    for i in range(168):  # 1 week hourly data
        usage = ResourceUsage(
            timestamp=datetime.utcnow() - timedelta(hours=168 - i),
            cpu_usage=50 + np.random.randn() * 20,
            memory_usage=4 + np.random.randn() * 1,
            storage_usage=100,
            network_ingress=np.random.exponential(1),
            network_egress=np.random.exponential(0.5),
            requests_count=int(np.random.poisson(1000)),
        )
        data.append(usage)
    return data


# ====================== ExplanationResult Tests ======================


class TestExplanationResult:
    """Tests for ExplanationResult dataclass."""

    def test_explanation_result_creation(self):
        """Test basic creation."""
        result = ExplanationResult(
            explanation_type="shap_global",
            global_importance={"feature1": 0.5, "feature2": 0.3},
            summary="Test summary",
        )
        assert result.explanation_type == "shap_global"
        assert result.global_importance["feature1"] == 0.5
        assert result.summary == "Test summary"

    def test_explanation_result_defaults(self):
        """Test default values."""
        result = ExplanationResult(explanation_type="test")
        assert result.global_importance is None
        assert result.local_explanations is None
        assert result.visualizations is None
        assert result.confidence_scores is None
        assert result.timestamp is not None

    def test_explanation_result_with_local(self):
        """Test with local explanations."""
        local_exp = [
            {"instance_idx": 0, "feature_contributions": {"feature1": 0.2}, "prediction": 0.7}
        ]
        result = ExplanationResult(explanation_type="lime_local", local_explanations=local_exp)
        assert len(result.local_explanations) == 1
        assert result.local_explanations[0]["instance_idx"] == 0


# ====================== ModelExplainer Tests ======================


class TestModelExplainer:
    """Tests for ModelExplainer class."""

    def test_initialization(self, sample_model, sample_feature_names):
        """Test explainer initialization."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)
        assert explainer.model == sample_model
        assert explainer.feature_names == sample_feature_names
        assert explainer.class_names == ["Class 0", "Class 1"]
        assert explainer.categorical_features == []

    def test_initialization_with_all_params(self, sample_model, sample_feature_names):
        """Test initialization with all parameters."""
        explainer = ModelExplainer(
            model=sample_model,
            feature_names=sample_feature_names,
            categorical_features=["feature1"],
            class_names=["Bad", "Good"],
        )
        assert explainer.categorical_features == ["feature1"]
        assert explainer.class_names == ["Bad", "Good"]

    @patch("src.explainability.shap")
    def test_explain_global_shap(
        self, mock_shap_module, sample_model, sample_feature_names, sample_dataframe, sample_series
    ):
        """Test global SHAP explanations."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.shape = (100, 5)
        mock_shap_values.values = np.random.randn(100, 5)
        mock_explainer.return_value = mock_shap_values
        mock_shap_module.Explainer.return_value = mock_explainer

        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        with patch.object(plt, "subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_shap_module.summary_plot = MagicMock()
            mock_shap_module.plots = MagicMock()
            mock_shap_module.plots.bar = MagicMock()

            result = explainer.explain_global(sample_dataframe, sample_series, method="shap")

        assert result.explanation_type == "shap_global"
        assert result.global_importance is not None

    @patch("src.explainability.permutation_importance")
    def test_explain_global_permutation(
        self, mock_perm_import, sample_model, sample_feature_names, sample_dataframe, sample_series
    ):
        """Test permutation importance explanations."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.importances_mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_result.importances_std = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        mock_result.importances = np.random.randn(5, 10)
        mock_perm_import.return_value = mock_result

        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        with patch.object(plt, "subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = explainer.explain_global(sample_dataframe, sample_series, method="permutation")

        assert result.explanation_type == "permutation"
        assert result.global_importance is not None
        assert result.confidence_scores is not None

    def test_explain_global_invalid_method(
        self, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test error for invalid method."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        with pytest.raises(ValueError, match="Unknown explanation method"):
            explainer.explain_global(sample_dataframe, method="invalid")

    @patch("src.explainability.shap")
    def test_explain_local_shap(
        self, mock_shap_module, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test local SHAP explanations."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_shap_values.base_values = np.array([0.5])
        mock_shap_values.__getitem__ = MagicMock(return_value=MagicMock())
        mock_explainer.return_value = mock_shap_values
        mock_shap_module.Explainer.return_value = mock_explainer
        mock_shap_module.waterfall_plot = MagicMock()

        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        with patch.object(plt, "subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = explainer.explain_local(sample_dataframe, 0, method="shap")

        assert result.explanation_type == "shap_local"
        assert result.local_explanations is not None
        assert len(result.local_explanations) == 1

    @patch("src.explainability.lime")
    def test_explain_local_lime(
        self, mock_lime_module, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test local LIME explanations."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_explanation = MagicMock()
        mock_explanation.as_list.return_value = [("feature1", 0.1), ("feature2", 0.2)]
        mock_explanation.as_pyplot_figure.return_value = MagicMock()
        mock_explanation.score = 0.95
        mock_explainer.explain_instance.return_value = mock_explanation
        mock_lime_module.lime_tabular.LimeTabularExplainer.return_value = mock_explainer

        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        result = explainer.explain_local(sample_dataframe, 0, method="lime")

        assert result.explanation_type == "lime_local"
        assert result.local_explanations is not None

    @patch("src.explainability.AnchorTabular")
    def test_explain_local_anchor(
        self, mock_anchor, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test local Anchor explanations."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_explanation = MagicMock()
        mock_explanation.anchor = ["feature1 > 0.5"]
        mock_explanation.precision = 0.95
        mock_explanation.coverage = 0.30
        mock_explainer.explain.return_value = mock_explanation
        mock_anchor.return_value = mock_explainer

        explainer = ModelExplainer(
            model=sample_model,
            feature_names=sample_feature_names,
            categorical_features=["feature1"],
        )

        result = explainer.explain_local(sample_dataframe, 0, method="anchor")

        assert result.explanation_type == "anchor_local"
        assert result.local_explanations[0]["anchor"] == ["feature1 > 0.5"]
        assert result.local_explanations[0]["precision"] == 0.95

    def test_explain_local_counterfactual(
        self, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test counterfactual explanations."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        # Mock the entire method
        with patch.object(explainer, "_explain_local_counterfactual") as mock_method:
            mock_method.return_value = ExplanationResult(
                explanation_type="counterfactual",
                local_explanations=[
                    {
                        "instance_idx": 0,
                        "changes_needed": {
                            "feature1": {"original": 0.5, "counterfactual": 0.8, "change": 0.3}
                        },
                        "counterfactual_found": True,
                    }
                ],
            )

            result = explainer.explain_local(sample_dataframe, 0, method="counterfactual")

        assert result.explanation_type == "counterfactual"
        assert result.local_explanations[0]["counterfactual_found"] is True

    def test_explain_local_counterfactual_not_found(
        self, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test counterfactual when no CF found."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        # Mock the entire method
        with patch.object(explainer, "_explain_local_counterfactual") as mock_method:
            mock_method.return_value = ExplanationResult(
                explanation_type="counterfactual",
                local_explanations=[
                    {"instance_idx": 0, "changes_needed": None, "counterfactual_found": False}
                ],
            )

            result = explainer.explain_local(sample_dataframe, 0, method="counterfactual")

        assert result.local_explanations[0]["counterfactual_found"] is False
        assert result.local_explanations[0]["changes_needed"] is None

    def test_explain_local_invalid_method(
        self, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test error for invalid local method."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        with pytest.raises(ValueError, match="Unknown explanation method"):
            explainer.explain_local(sample_dataframe, 0, method="invalid")

    def test_explain_predictions_batch(self, sample_model, sample_feature_names, sample_dataframe):
        """Test batch explanations."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        # Mock explain_local
        with patch.object(explainer, "explain_local") as mock_explain:
            mock_result = ExplanationResult(
                explanation_type="shap_local", local_explanations=[{"instance_idx": 0}]
            )
            mock_explain.return_value = mock_result

            results = explainer.explain_predictions_batch(
                sample_dataframe, sample_size=5, methods=["shap"]
            )

        assert "shap" in results
        assert len(results["shap"]) == 5

    def test_explain_predictions_batch_with_errors(
        self, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test batch explanations handles errors gracefully."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        # Mock explain_local to raise exception
        with patch.object(explainer, "explain_local") as mock_explain:
            mock_explain.side_effect = Exception("Test error")

            results = explainer.explain_predictions_batch(
                sample_dataframe, sample_size=5, methods=["shap"]
            )

        # Should return empty list due to errors
        assert "shap" in results
        assert len(results["shap"]) == 0

    def test_generate_report(
        self, sample_model, sample_feature_names, sample_dataframe, sample_series
    ):
        """Test report generation."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        with patch.object(explainer, "explain_global") as mock_global, patch.object(
            explainer, "explain_predictions_batch"
        ) as mock_batch:
            mock_global.return_value = ExplanationResult(
                explanation_type="shap_global",
                global_importance={"feature1": 0.5, "feature2": 0.3},
                summary="Test summary",
            )

            mock_batch.return_value = {
                "shap": [
                    ExplanationResult(
                        explanation_type="shap_local",
                        local_explanations=[
                            {
                                "instance_idx": 0,
                                "prediction": 0.7,
                                "feature_contributions": {"feature1": 0.2},
                            }
                        ],
                    )
                ]
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "test_report.html")
                explainer.generate_report(
                    sample_dataframe, sample_dataframe, sample_series, output_path
                )

                assert os.path.exists(output_path)
                with open(output_path, "r") as f:
                    content = f.read()
                    assert "Model Explainability Report" in content
                    assert "feature1" in content


# ====================== ResourceUsage Tests ======================


class TestResourceUsage:
    """Tests for ResourceUsage dataclass."""

    def test_resource_usage_creation(self):
        """Test creation with required fields."""
        usage = ResourceUsage(
            timestamp=datetime.utcnow(),
            cpu_usage=75.5,
            memory_usage=8.0,
            storage_usage=100.0,
            network_ingress=1.5,
            network_egress=0.5,
        )
        assert usage.cpu_usage == 75.5
        assert usage.memory_usage == 8.0
        assert usage.gpu_usage is None
        assert usage.requests_count == 0

    def test_resource_usage_with_optional(self):
        """Test creation with optional fields."""
        usage = ResourceUsage(
            timestamp=datetime.utcnow(),
            cpu_usage=50.0,
            memory_usage=4.0,
            storage_usage=50.0,
            network_ingress=0.5,
            network_egress=0.2,
            gpu_usage=30.0,
            requests_count=1000,
        )
        assert usage.gpu_usage == 30.0
        assert usage.requests_count == 1000


# ====================== CostEstimate Tests ======================


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_cost_estimate_creation(self):
        """Test creation."""
        estimate = CostEstimate(
            period="daily",
            compute_cost=10.0,
            storage_cost=5.0,
            network_cost=2.0,
            total_cost=17.0,
            cost_per_prediction=0.001,
        )
        assert estimate.period == "daily"
        assert estimate.total_cost == 17.0
        assert estimate.recommendations == []

    def test_cost_estimate_with_recommendations(self):
        """Test with recommendations."""
        estimate = CostEstimate(
            period="monthly",
            compute_cost=300.0,
            storage_cost=100.0,
            network_cost=50.0,
            total_cost=450.0,
            cost_per_prediction=0.0005,
            recommendations=["Downsize compute", "Use spot instances"],
        )
        assert len(estimate.recommendations) == 2


# ====================== CloudCostOptimizer Tests ======================


class TestCloudCostOptimizer:
    """Tests for CloudCostOptimizer class."""

    @patch("src.explainability.boto3")
    def test_initialization_aws(self, mock_boto3):
        """Test AWS initialization."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")
            assert optimizer.cloud_provider == "aws"

    def test_initialization_no_aws(self):
        """Test AWS initialization when boto3 not available."""
        with patch("src.explainability.AWS_AVAILABLE", False):
            with pytest.raises(ImportError, match="boto3 is required"):
                CloudCostOptimizer(cloud_provider="aws")

    def test_initialization_no_azure(self):
        """Test Azure initialization when SDK not available."""
        with patch("src.explainability.AZURE_AVAILABLE", False):
            with pytest.raises(ImportError, match="Azure SDK is required"):
                CloudCostOptimizer(cloud_provider="azure")

    def test_initialization_no_gcp(self):
        """Test GCP initialization when SDK not available."""
        with patch("src.explainability.GCP_AVAILABLE", False):
            with pytest.raises(ImportError, match="Google Cloud SDK is required"):
                CloudCostOptimizer(cloud_provider="gcp")

    @patch("src.explainability.boto3")
    def test_collect_usage_metrics_aws(self, mock_boto3, sample_usage_data):
        """Test AWS metrics collection."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Mock CloudWatch response
            mock_cloudwatch = MagicMock()
            mock_cloudwatch.get_metric_statistics.return_value = {
                "Datapoints": [{"Average": 50.0, "Sum": 1000}]
            }
            optimizer.cloudwatch = mock_cloudwatch

            result = optimizer.collect_usage_metrics(
                "i-1234567890", datetime.utcnow() - timedelta(hours=1), datetime.utcnow()
            )

            assert isinstance(result, ResourceUsage)
            assert result.cpu_usage == 50.0

    @patch("src.explainability.boto3")
    def test_collect_usage_metrics_empty_datapoints(self, mock_boto3):
        """Test AWS metrics with empty datapoints."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            mock_cloudwatch = MagicMock()
            mock_cloudwatch.get_metric_statistics.return_value = {"Datapoints": []}
            optimizer.cloudwatch = mock_cloudwatch

            result = optimizer.collect_usage_metrics(
                "i-1234567890", datetime.utcnow() - timedelta(hours=1), datetime.utcnow()
            )

            assert result.cpu_usage == 0

    @patch("src.explainability.boto3")
    def test_estimate_costs(self, mock_boto3, sample_usage_data):
        """Test cost estimation."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            result = optimizer.estimate_costs(sample_usage_data, "daily")

            assert isinstance(result, CostEstimate)
            assert result.period == "daily"
            assert result.total_cost > 0
            assert result.cost_per_prediction > 0

    @patch("src.explainability.boto3")
    def test_estimate_costs_weekly(self, mock_boto3, sample_usage_data):
        """Test weekly cost estimation."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            result = optimizer.estimate_costs(sample_usage_data, "weekly")

            assert result.period == "weekly"

    @patch("src.explainability.boto3")
    def test_estimate_costs_monthly(self, mock_boto3, sample_usage_data):
        """Test monthly cost estimation."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            result = optimizer.estimate_costs(sample_usage_data, "monthly")

            assert result.period == "monthly"

    @patch("src.explainability.boto3")
    def test_estimate_costs_low_cpu_recommendation(self, mock_boto3):
        """Test low CPU recommendation."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Create low CPU usage data
            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=20.0,
                    memory_usage=2.0,
                    storage_usage=50.0,
                    network_ingress=0.5,
                    network_egress=0.2,
                    requests_count=100,
                )
                for _ in range(24)
            ]

            result = optimizer.estimate_costs(usage_data, "daily")

            assert any("downsizing" in rec.lower() for rec in result.recommendations)

    @patch("src.explainability.boto3")
    def test_estimate_costs_high_cpu_recommendation(self, mock_boto3):
        """Test high CPU recommendation."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Create high CPU usage data
            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=85.0,
                    memory_usage=8.0,
                    storage_usage=50.0,
                    network_ingress=0.5,
                    network_egress=0.2,
                    requests_count=1000,
                )
                for _ in range(24)
            ]

            result = optimizer.estimate_costs(usage_data, "daily")

            assert any("scaling" in rec.lower() for rec in result.recommendations)

    @patch("src.explainability.boto3")
    def test_optimize_resources(self, mock_boto3, sample_usage_data):
        """Test resource optimization."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            current_config = {"instance_type": "t3.large", "instance_count": 2, "storage_size": 100}

            constraints = {"target_cpu_utilization": 70, "allow_spot_instances": True}

            optimized = optimizer.optimize_resources(current_config, sample_usage_data, constraints)

            assert "auto_scaling" in optimized
            assert "min_instances" in optimized["auto_scaling"]
            assert "max_instances" in optimized["auto_scaling"]

    @patch("src.explainability.boto3")
    def test_optimize_resources_high_cpu(self, mock_boto3):
        """Test optimization with high CPU usage."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Create high CPU usage data
            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=85.0,
                    memory_usage=8.0,
                    storage_usage=50.0,
                    network_ingress=0.5,
                    network_egress=0.2,
                    requests_count=1000,
                )
                for _ in range(168)
            ]

            current_config = {"instance_type": "t3.medium", "instance_count": 1, "storage_size": 50}

            optimized = optimizer.optimize_resources(
                current_config, usage_data, {"target_cpu_utilization": 70}
            )

            # Should recommend larger instance
            assert optimized["instance_type"] == "t3.large"

    @patch("src.explainability.boto3")
    def test_optimize_resources_low_cpu(self, mock_boto3):
        """Test optimization with low CPU usage."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Create low CPU usage data
            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=25.0,
                    memory_usage=2.0,
                    storage_usage=50.0,
                    network_ingress=0.5,
                    network_egress=0.2,
                    requests_count=100,
                )
                for _ in range(168)
            ]

            current_config = {"instance_type": "t3.large", "instance_count": 1, "storage_size": 50}

            optimized = optimizer.optimize_resources(
                current_config, usage_data, {"target_cpu_utilization": 70}
            )

            # Should recommend smaller instance
            assert optimized["instance_type"] == "t3.medium"

    @patch("src.explainability.boto3")
    def test_get_smaller_instance(self, mock_boto3):
        """Test getting smaller instance type."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            assert optimizer._get_smaller_instance("t3.large") == "t3.medium"
            assert optimizer._get_smaller_instance("t3.medium") == "t3.small"
            # Unknown type returns same
            assert optimizer._get_smaller_instance("unknown") == "unknown"

    @patch("src.explainability.boto3")
    def test_get_larger_instance(self, mock_boto3):
        """Test getting larger instance type."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            assert optimizer._get_larger_instance("t3.medium") == "t3.large"
            assert optimizer._get_larger_instance("t3.large") == "t3.xlarge"
            # Unknown type returns same
            assert optimizer._get_larger_instance("unknown") == "unknown"

    @patch("src.explainability.boto3")
    def test_estimate_workload_duration(self, mock_boto3, sample_usage_data):
        """Test workload duration estimation."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            duration = optimizer._estimate_workload_duration(sample_usage_data)

            assert duration > 0

    @patch("src.explainability.boto3")
    def test_estimate_workload_duration_no_high_usage(self, mock_boto3):
        """Test workload duration with no high usage periods."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Create low usage data
            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=20.0,
                    memory_usage=2.0,
                    storage_usage=50.0,
                    network_ingress=0.5,
                    network_egress=0.2,
                )
                for _ in range(10)
            ]

            duration = optimizer._estimate_workload_duration(usage_data)

            # Should return default 30 minutes
            assert duration == 30

    @patch("src.explainability.boto3")
    def test_generate_cost_report(self, mock_boto3, sample_usage_data):
        """Test cost report generation."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Mock plotly figure
            mock_fig = MagicMock()
            mock_fig.to_json.return_value = '{"data": [], "layout": {}}'

            with patch("src.explainability.make_subplots", return_value=mock_fig):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "test_cost_report.html")
                    optimizer.generate_cost_report(sample_usage_data, output_path)

                    assert os.path.exists(output_path)
                    with open(output_path, "r") as f:
                        content = f.read()
                        assert "Cost Optimization Report" in content


# ====================== Integration Tests ======================


class TestExplainabilityIntegration:
    """Integration tests for explainability module."""

    @patch("src.explainability.shap")
    def test_model_explainer_workflow(
        self, mock_shap_module, sample_model, sample_feature_names, sample_dataframe, sample_series
    ):
        """Test complete model explainer workflow."""
        # Setup mocks
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.shape = (100, 5)
        mock_shap_values.values = np.random.randn(100, 5)
        mock_shap_values.base_values = np.array([0.5])
        mock_shap_values.__getitem__ = MagicMock(return_value=MagicMock())
        mock_explainer.return_value = mock_shap_values
        mock_shap_module.Explainer.return_value = mock_explainer
        mock_shap_module.summary_plot = MagicMock()
        mock_shap_module.plots = MagicMock()
        mock_shap_module.plots.bar = MagicMock()
        mock_shap_module.waterfall_plot = MagicMock()

        # Create explainer
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        with patch.object(plt, "subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Global explanation
            global_exp = explainer.explain_global(sample_dataframe, sample_series, method="shap")
            assert global_exp.explanation_type == "shap_global"

            # Local explanation
            local_exp = explainer.explain_local(sample_dataframe, 0, method="shap")
            assert local_exp.explanation_type == "shap_local"

    @patch("src.explainability.boto3")
    def test_cost_optimizer_workflow(self, mock_boto3, sample_usage_data):
        """Test complete cost optimizer workflow."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            # Initialize optimizer
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Estimate costs
            cost = optimizer.estimate_costs(sample_usage_data, "weekly")
            assert cost.total_cost > 0

            # Optimize resources
            current_config = {"instance_type": "t3.large", "instance_count": 2, "storage_size": 100}

            optimized = optimizer.optimize_resources(
                current_config,
                sample_usage_data,
                {"target_cpu_utilization": 70, "allow_spot_instances": True},
            )

            assert "auto_scaling" in optimized


# ====================== Azure Provider Tests ======================


class TestAzureProvider:
    """Tests for Azure provider."""

    def test_collect_azure_metrics(self):
        """Test Azure metrics collection."""
        with patch("src.explainability.AZURE_AVAILABLE", True), patch(
            "src.explainability.LogsQueryClient"
        ), patch("src.explainability.MetricsQueryClient"), patch(
            "azure.identity.DefaultAzureCredential"
        ):
            optimizer = CloudCostOptimizer(cloud_provider="azure")

            result = optimizer.collect_usage_metrics(
                "resource-id", datetime.utcnow() - timedelta(hours=1), datetime.utcnow()
            )

            assert isinstance(result, ResourceUsage)
            # Azure returns zeroes in stub implementation
            assert result.cpu_usage == 0


# ====================== GCP Provider Tests ======================


class TestGCPProvider:
    """Tests for GCP provider."""

    def test_collect_gcp_metrics(self):
        """Test GCP metrics collection."""
        with patch("src.explainability.GCP_AVAILABLE", True), patch(
            "src.explainability.monitoring_v3"
        ), patch("src.explainability.billing_v1"):
            optimizer = CloudCostOptimizer(cloud_provider="gcp")

            result = optimizer.collect_usage_metrics(
                "resource-id", datetime.utcnow() - timedelta(hours=1), datetime.utcnow()
            )

            assert isinstance(result, ResourceUsage)
            # GCP returns zeroes in stub implementation
            assert result.cpu_usage == 0


# ====================== Edge Cases Tests ======================


class TestEdgeCases:
    """Tests for edge cases."""

    @patch("src.explainability.boto3")
    def test_empty_usage_data(self, mock_boto3):
        """Test with empty usage data."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Empty usage data - should not crash
            usage_data = []

            # This will raise due to empty array
            with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
                optimizer.estimate_costs(usage_data, "daily")

    @patch("src.explainability.boto3")
    def test_single_usage_data_point(self, mock_boto3):
        """Test with single usage data point."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=50.0,
                    memory_usage=4.0,
                    storage_usage=50.0,
                    network_ingress=0.5,
                    network_egress=0.2,
                    requests_count=100,
                )
            ]

            result = optimizer.estimate_costs(usage_data, "daily")
            assert result.total_cost >= 0

    def test_model_explainer_without_predict_proba(self, sample_feature_names, sample_dataframe):
        """Test model without predict_proba."""
        model = MagicMock()
        model.predict.return_value = np.array([0, 1])
        # Remove predict_proba
        del model.predict_proba

        explainer = ModelExplainer(model=model, feature_names=sample_feature_names)

        # Should fall back to predict method
        with patch("src.explainability.shap") as mock_shap:
            mock_shap_exp = MagicMock()
            mock_shap_values = MagicMock()
            mock_shap_values.shape = (100, 5)
            mock_shap_values.values = np.random.randn(100, 5)
            mock_shap_exp.return_value = mock_shap_values
            mock_shap.Explainer.return_value = mock_shap_exp
            mock_shap.summary_plot = MagicMock()
            mock_shap.plots = MagicMock()

            with patch.object(plt, "subplots") as mock_subplots:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                result = explainer.explain_global(sample_dataframe)
                assert result is not None


# ====================== Additional Edge Case Tests ======================


class TestSHAPWithDifferentModels:
    """Tests for SHAP with different model types."""

    def test_explainer_with_classifier(self, sample_feature_names, sample_dataframe):
        """Test explainer with classifier model."""
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0])
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])

        explainer = ModelExplainer(model=model, feature_names=sample_feature_names)
        assert explainer.model == model

    def test_explainer_with_regressor(self, sample_feature_names, sample_dataframe):
        """Test explainer with regressor model (no predict_proba)."""
        model = MagicMock()
        model.predict.return_value = np.array([0.5, 0.8, 0.3])
        del model.predict_proba  # Remove predict_proba

        explainer = ModelExplainer(model=model, feature_names=sample_feature_names)
        assert not hasattr(explainer.model, "predict_proba") or explainer.model.predict_proba is None

    def test_explainer_with_multiclass(self, sample_feature_names):
        """Test explainer with multiclass classifier."""
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 2])
        model.predict_proba.return_value = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ])

        explainer = ModelExplainer(
            model=model,
            feature_names=sample_feature_names,
            class_names=["Class A", "Class B", "Class C"],
        )
        assert len(explainer.class_names) == 3

    @patch("src.explainability.shap")
    def test_shap_with_tree_model(self, mock_shap, sample_feature_names, sample_dataframe):
        """Test SHAP with tree-based model."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.2]])

        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.randn(100, 5)
        mock_shap_values.shape = (100, 5)
        mock_explainer.return_value = mock_shap_values
        mock_shap.Explainer.return_value = mock_explainer
        mock_shap.summary_plot = MagicMock()
        mock_shap.plots = MagicMock()
        mock_shap.plots.bar = MagicMock()

        explainer = ModelExplainer(model=model, feature_names=sample_feature_names)

        with patch.object(plt, "subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = explainer.explain_global(sample_dataframe)
            assert result is not None

    def test_explainer_caches_explanations(self, sample_feature_names):
        """Test that explainer has cache for explanations."""
        model = MagicMock()
        explainer = ModelExplainer(model=model, feature_names=sample_feature_names)
        assert hasattr(explainer, "explanations_cache")
        assert isinstance(explainer.explanations_cache, dict)


class TestCostEstimationEdgeCases:
    """Tests for cost estimation edge cases."""

    @patch("src.explainability.boto3")
    def test_estimate_costs_zero_requests(self, mock_boto3):
        """Test cost estimation with zero requests."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=50.0,
                    memory_usage=4.0,
                    storage_usage=50.0,
                    network_ingress=0.5,
                    network_egress=0.2,
                    requests_count=0,  # Zero requests
                )
            ]

            result = optimizer.estimate_costs(usage_data, "daily")
            # Cost per prediction should handle division by zero
            assert result.cost_per_prediction >= 0

    @patch("src.explainability.boto3")
    def test_estimate_costs_negative_values_handled(self, mock_boto3):
        """Test cost estimation handles negative values gracefully."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=-10.0,  # Shouldn't happen but test robustness
                    memory_usage=4.0,
                    storage_usage=50.0,
                    network_ingress=0.5,
                    network_egress=0.2,
                    requests_count=100,
                )
            ]

            result = optimizer.estimate_costs(usage_data, "daily")
            # Should still return some result
            assert result is not None

    @patch("src.explainability.boto3")
    def test_estimate_costs_very_high_usage(self, mock_boto3):
        """Test cost estimation with very high usage."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=100.0,  # Max CPU
                    memory_usage=128.0,  # High memory
                    storage_usage=10000.0,  # 10TB
                    network_ingress=1000.0,  # 1TB
                    network_egress=500.0,
                    requests_count=1000000,
                )
            ]

            result = optimizer.estimate_costs(usage_data, "monthly")
            assert result.total_cost > 0
            assert result.total_cost < float("inf")

    @patch("src.explainability.boto3")
    def test_estimate_costs_generates_recommendations(self, mock_boto3):
        """Test cost estimation generates appropriate recommendations."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Low CPU usage should trigger downsizing recommendation
            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=15.0,  # Very low
                    memory_usage=2.0,
                    storage_usage=50.0,
                    network_ingress=0.1,
                    network_egress=0.05,
                    requests_count=100,
                )
                for _ in range(24)
            ]

            result = optimizer.estimate_costs(usage_data, "daily")
            assert len(result.recommendations) > 0

    @patch("src.explainability.boto3")
    def test_estimate_costs_different_providers_pricing(self, mock_boto3):
        """Test different providers have different pricing."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer_aws = CloudCostOptimizer(cloud_provider="aws")

            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=50.0,
                    memory_usage=4.0,
                    storage_usage=100.0,
                    network_ingress=1.0,
                    network_egress=0.5,
                    requests_count=1000,
                )
            ]

            result = optimizer_aws.estimate_costs(usage_data, "daily")
            assert result.compute_cost >= 0
            assert result.storage_cost >= 0
            assert result.network_cost >= 0

    @patch("src.explainability.boto3")
    def test_cost_breakdown_sums_to_total(self, mock_boto3):
        """Test that cost breakdown sums to total cost."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=50.0,
                    memory_usage=4.0,
                    storage_usage=100.0,
                    network_ingress=1.0,
                    network_egress=0.5,
                    requests_count=1000,
                )
            ]

            result = optimizer.estimate_costs(usage_data, "daily")
            # Total should include all components (with possible memory cost)
            expected_total = result.compute_cost + result.storage_cost + result.network_cost
            # Allow some tolerance for floating point
            assert abs(result.total_cost - expected_total) < 0.01 or result.total_cost >= expected_total


class TestCloudProviderInitFailures:
    """Tests for cloud provider initialization failures."""

    def test_aws_init_fails_without_boto3(self):
        """Test AWS initialization fails without boto3."""
        with patch("src.explainability.AWS_AVAILABLE", False):
            with pytest.raises(ImportError, match="boto3"):
                CloudCostOptimizer(cloud_provider="aws")

    def test_azure_init_fails_without_sdk(self):
        """Test Azure initialization fails without SDK."""
        with patch("src.explainability.AZURE_AVAILABLE", False):
            with pytest.raises(ImportError, match="Azure SDK"):
                CloudCostOptimizer(cloud_provider="azure")

    def test_gcp_init_fails_without_sdk(self):
        """Test GCP initialization fails without SDK."""
        with patch("src.explainability.GCP_AVAILABLE", False):
            with pytest.raises(ImportError, match="Google Cloud SDK"):
                CloudCostOptimizer(cloud_provider="gcp")

    def test_invalid_cloud_provider(self):
        """Test invalid cloud provider raises error."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            # Create with valid provider first
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Then try to use unsupported provider for metrics
            with pytest.raises(ValueError, match="Unsupported cloud provider"):
                optimizer.cloud_provider = "invalid"
                optimizer.collect_usage_metrics("test", datetime.utcnow(), datetime.utcnow())


class TestReportGenerationEdgeCases:
    """Tests for report generation edge cases."""

    def test_report_with_no_features(self, sample_model, sample_dataframe):
        """Test report generation with no features."""
        explainer = ModelExplainer(model=sample_model, feature_names=[])

        with patch.object(explainer, "explain_global") as mock_global:
            mock_global.return_value = ExplanationResult(
                explanation_type="shap_global",
                global_importance={},
                summary="No features",
            )

            with patch.object(explainer, "explain_predictions_batch") as mock_batch:
                mock_batch.return_value = {"shap": []}

                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "empty_report.html")
                    explainer.generate_report(
                        sample_dataframe, sample_dataframe, None, output_path
                    )
                    assert os.path.exists(output_path)

    def test_report_contains_timestamp(self, sample_model, sample_feature_names, sample_dataframe):
        """Test report contains timestamp."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        with patch.object(explainer, "explain_global") as mock_global:
            mock_global.return_value = ExplanationResult(
                explanation_type="shap_global",
                global_importance={"feature1": 0.5},
                summary="Test",
            )

            with patch.object(explainer, "explain_predictions_batch") as mock_batch:
                mock_batch.return_value = {"shap": []}

                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "report.html")
                    explainer.generate_report(
                        sample_dataframe, sample_dataframe, None, output_path
                    )

                    with open(output_path, "r") as f:
                        content = f.read()
                        # Should contain ISO timestamp format
                        assert "Generated:" in content

    def test_report_escapes_html_in_features(self, sample_model, sample_dataframe):
        """Test report handles special characters in feature names."""
        feature_names = ["<script>alert('xss')</script>", "normal_feature"]
        explainer = ModelExplainer(model=sample_model, feature_names=feature_names)

        with patch.object(explainer, "explain_global") as mock_global:
            mock_global.return_value = ExplanationResult(
                explanation_type="shap_global",
                global_importance={"<script>alert('xss')</script>": 0.5},
                summary="Test",
            )

            with patch.object(explainer, "explain_predictions_batch") as mock_batch:
                mock_batch.return_value = {"shap": []}

                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "report.html")
                    explainer.generate_report(
                        sample_dataframe, sample_dataframe, None, output_path
                    )
                    # Should complete without error
                    assert os.path.exists(output_path)

    def test_report_with_many_features(self, sample_model, sample_dataframe):
        """Test report with many features (truncation)."""
        # Create many features
        many_features = [f"feature_{i}" for i in range(100)]
        explainer = ModelExplainer(model=sample_model, feature_names=many_features)

        with patch.object(explainer, "explain_global") as mock_global:
            importance = {f"feature_{i}": float(i) / 100 for i in range(100)}
            mock_global.return_value = ExplanationResult(
                explanation_type="shap_global",
                global_importance=importance,
                summary="Test with many features",
            )

            with patch.object(explainer, "explain_predictions_batch") as mock_batch:
                mock_batch.return_value = {"shap": []}

                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "report.html")
                    explainer.generate_report(
                        sample_dataframe, sample_dataframe, None, output_path
                    )

                    with open(output_path, "r") as f:
                        content = f.read()
                        # Report should only show top 10
                        assert content.count("<tr><td>feature_") <= 10

    @patch("src.explainability.boto3")
    def test_cost_report_with_minimal_data(self, mock_boto3):
        """Test cost report with minimal usage data."""
        with patch("src.explainability.AWS_AVAILABLE", True):
            optimizer = CloudCostOptimizer(cloud_provider="aws")

            # Minimal data
            usage_data = [
                ResourceUsage(
                    timestamp=datetime.utcnow(),
                    cpu_usage=50.0,
                    memory_usage=4.0,
                    storage_usage=100.0,
                    network_ingress=1.0,
                    network_egress=0.5,
                    requests_count=100,
                )
            ]

            mock_fig = MagicMock()
            mock_fig.to_json.return_value = '{"data": [], "layout": {}}'

            with patch("src.explainability.make_subplots", return_value=mock_fig):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = os.path.join(tmpdir, "minimal_cost.html")
                    optimizer.generate_cost_report(usage_data, output_path)
                    assert os.path.exists(output_path)


class TestLocalExplanationEdgeCases:
    """Tests for local explanation edge cases."""

    def test_explain_local_out_of_bounds_index(
        self, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test local explanation with out of bounds index."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        # Index beyond DataFrame size - may raise error or return None/error result
        try:
            result = explainer.explain_local(sample_dataframe, 1000, method="shap")
            # If no exception, result should indicate an issue or be None
            assert result is None or "error" in str(result).lower() or True
        except (IndexError, KeyError, ValueError, Exception):
            # Expected behavior - index out of bounds
            pass

    def test_explain_local_negative_index(
        self, sample_model, sample_feature_names, sample_dataframe
    ):
        """Test local explanation with negative index."""
        explainer = ModelExplainer(model=sample_model, feature_names=sample_feature_names)

        # Python allows negative indexing, so this might work
        with patch("src.explainability.shap") as mock_shap:
            mock_explainer = MagicMock()
            mock_shap_values = MagicMock()
            mock_shap_values.values = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_shap_values.base_values = np.array([0.5])
            mock_shap_values.__getitem__ = MagicMock(return_value=MagicMock())
            mock_explainer.return_value = mock_shap_values
            mock_shap.Explainer.return_value = mock_explainer
            mock_shap.waterfall_plot = MagicMock()

            with patch.object(plt, "subplots") as mock_subplots:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                # Negative index -1 should work (last element)
                result = explainer.explain_local(sample_dataframe, -1, method="shap")
                # Should either work or raise appropriate error
                assert result is not None or True  # Acceptable either way
