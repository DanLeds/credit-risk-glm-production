"""
Model Explainability and Cost Optimization System
=================================================
Advanced model interpretation and cloud resource cost management.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression
from interpret.blackbox import ShapKernel
from alibi.explainers import AnchorTabular, CounterFactual
from prometheus_client import CollectorRegistry, Gauge, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Optional cloud provider imports
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    boto3 = None
    AWS_AVAILABLE = False

try:
    from azure.monitor.query import LogsQueryClient, MetricsQueryClient
    AZURE_AVAILABLE = True
except ImportError:
    LogsQueryClient = None
    MetricsQueryClient = None
    AZURE_AVAILABLE = False

try:
    from google.cloud import monitoring_v3, billing_v1
    GCP_AVAILABLE = True
except ImportError:
    monitoring_v3 = None
    billing_v1 = None
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)


# ====================== Model Explainability ======================

@dataclass
class ExplanationResult:
    """Container for model explanation results."""
    
    explanation_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    global_importance: Optional[Dict[str, float]] = None
    local_explanations: Optional[List[Dict[str, Any]]] = None
    visualizations: Optional[List[Any]] = None
    summary: Optional[str] = None
    confidence_scores: Optional[Dict[str, float]] = None
    

class ModelExplainer:
    """
    Comprehensive model explainability system supporting multiple explanation methods.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        categorical_features: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize model explainer.
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
            categorical_features: List of categorical feature names
            class_names: List of class names for classification
        """
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.class_names = class_names or ["Class 0", "Class 1"]
        self.explanations_cache = {}
        
    def explain_global(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        method: str = "shap"
    ) -> ExplanationResult:
        """
        Generate global model explanations.
        
        Args:
            X_train: Training data
            y_train: Training labels
            method: Explanation method (shap, permutation, morris)
            
        Returns:
            ExplanationResult with global importance
        """
        logger.info(f"Generating global explanations using {method}...")
        
        if method == "shap":
            return self._explain_global_shap(X_train)
        elif method == "permutation":
            return self._explain_global_permutation(X_train, y_train)
        elif method == "morris":
            return self._explain_global_morris(X_train)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
            
    def _explain_global_shap(self, X_train: pd.DataFrame) -> ExplanationResult:
        """Generate SHAP-based global explanations."""
        # Create SHAP explainer
        if hasattr(self.model, 'predict_proba'):
            explainer = shap.Explainer(self.model.predict_proba, X_train)
        else:
            explainer = shap.Explainer(self.model.predict, X_train)
            
        # Calculate SHAP values
        shap_values = explainer(X_train)
        
        # Get feature importance
        if len(shap_values.shape) == 3:  # Multi-class
            importance = np.abs(shap_values.values).mean(axis=(0, 2))
        else:
            importance = np.abs(shap_values.values).mean(axis=0)
            
        global_importance = dict(zip(self.feature_names, importance))
        
        # Create visualizations
        figs = []
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, feature_names=self.feature_names, show=False)
        figs.append(fig)
        
        # Feature importance bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        figs.append(fig)
        
        return ExplanationResult(
            explanation_type="shap_global",
            global_importance=global_importance,
            visualizations=figs,
            summary=f"Top 3 important features: {', '.join(list(sorted(global_importance.keys(), key=lambda x: global_importance[x], reverse=True))[:3])}"
        )
        
    def _explain_global_permutation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> ExplanationResult:
        """Generate permutation importance explanations."""
        result = permutation_importance(
            self.model,
            X_train,
            y_train,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        importance = dict(zip(self.feature_names, result.importances_mean))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_idx = result.importances_mean.argsort()
        ax.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=[self.feature_names[i] for i in sorted_idx]
        )
        ax.set_xlabel("Permutation Importance")
        ax.set_title("Feature Importance via Permutation")
        
        return ExplanationResult(
            explanation_type="permutation",
            global_importance=importance,
            visualizations=[fig],
            confidence_scores={
                feat: std for feat, std in zip(
                    self.feature_names,
                    result.importances_std
                )
            }
        )
        
    def _explain_global_morris(self, X_train: pd.DataFrame) -> ExplanationResult:
        """Generate Morris sensitivity analysis."""
        from SALib.sample import morris as morris_sampler
        from SALib.analyze import morris as morris_analyzer
        
        # Define problem
        problem = {
            'num_vars': len(self.feature_names),
            'names': self.feature_names,
            'bounds': [[X_train[col].min(), X_train[col].max()] 
                      for col in self.feature_names]
        }
        
        # Generate samples
        param_values = morris_sampler.sample(
            problem,
            N=100,
            num_levels=4,
            optimal_trajectories=10
        )
        
        # Evaluate model
        Y = np.array([
            self.model.predict(param_values[i:i+1])[0]
            for i in range(len(param_values))
        ])
        
        # Analyze
        Si = morris_analyzer.analyze(
            problem,
            param_values,
            Y,
            conf_level=0.95,
            print_to_console=False
        )
        
        importance = dict(zip(self.feature_names, Si['mu_star']))
        
        return ExplanationResult(
            explanation_type="morris_sensitivity",
            global_importance=importance,
            confidence_scores=dict(zip(self.feature_names, Si['sigma']))
        )
        
    def explain_local(
        self,
        X: pd.DataFrame,
        instance_idx: int,
        method: str = "shap"
    ) -> ExplanationResult:
        """
        Generate local explanations for a specific instance.
        
        Args:
            X: Data containing the instance
            instance_idx: Index of instance to explain
            method: Explanation method (shap, lime, anchor, counterfactual)
            
        Returns:
            ExplanationResult with local explanations
        """
        logger.info(f"Generating local explanation for instance {instance_idx} using {method}...")
        
        if method == "shap":
            return self._explain_local_shap(X, instance_idx)
        elif method == "lime":
            return self._explain_local_lime(X, instance_idx)
        elif method == "anchor":
            return self._explain_local_anchor(X, instance_idx)
        elif method == "counterfactual":
            return self._explain_local_counterfactual(X, instance_idx)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
            
    def _explain_local_shap(
        self,
        X: pd.DataFrame,
        instance_idx: int
    ) -> ExplanationResult:
        """Generate SHAP-based local explanation."""
        # Get the instance
        instance = X.iloc[instance_idx:instance_idx+1]
        
        # Create explainer
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(instance)
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False)
        
        # Extract explanation
        feature_contributions = dict(
            zip(self.feature_names, shap_values.values[0])
        )
        
        return ExplanationResult(
            explanation_type="shap_local",
            local_explanations=[{
                "instance_idx": instance_idx,
                "feature_contributions": feature_contributions,
                "prediction": float(self.model.predict(instance)[0]),
                "base_value": float(shap_values.base_values[0])
            }],
            visualizations=[fig]
        )
        
    def _explain_local_lime(
        self,
        X: pd.DataFrame,
        instance_idx: int
    ) -> ExplanationResult:
        """Generate LIME-based local explanation."""
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification' if hasattr(self.model, 'predict_proba') else 'regression'
        )
        
        # Get explanation
        instance = X.iloc[instance_idx].values
        
        if hasattr(self.model, 'predict_proba'):
            exp = explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=len(self.feature_names)
            )
        else:
            exp = explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=len(self.feature_names)
            )
            
        # Extract explanation
        feature_contributions = dict(exp.as_list())
        
        # Create visualization
        fig = exp.as_pyplot_figure()
        
        return ExplanationResult(
            explanation_type="lime_local",
            local_explanations=[{
                "instance_idx": instance_idx,
                "feature_contributions": feature_contributions,
                "prediction": float(self.model.predict(instance.reshape(1, -1))[0]),
                "local_accuracy": exp.score
            }],
            visualizations=[fig]
        )
        
    def _explain_local_anchor(
        self,
        X: pd.DataFrame,
        instance_idx: int
    ) -> ExplanationResult:
        """Generate Anchor-based local explanation."""
        # Prepare data
        categorical_idx = [
            i for i, feat in enumerate(self.feature_names)
            if feat in self.categorical_features
        ]
        
        # Create Anchor explainer
        explainer = AnchorTabular(
            predictor=self.model.predict,
            feature_names=self.feature_names,
            categorical_names={idx: ['0', '1'] for idx in categorical_idx}
        )
        
        explainer.fit(X.values)
        
        # Get explanation
        instance = X.iloc[instance_idx].values
        explanation = explainer.explain(instance)
        
        return ExplanationResult(
            explanation_type="anchor_local",
            local_explanations=[{
                "instance_idx": instance_idx,
                "anchor": explanation.anchor,
                "precision": explanation.precision,
                "coverage": explanation.coverage
            }]
        )
        
    def _explain_local_counterfactual(
        self,
        X: pd.DataFrame,
        instance_idx: int
    ) -> ExplanationResult:
        """Generate counterfactual explanation."""
        from alibi.explainers import CounterFactual
        
        # Get instance
        instance = X.iloc[instance_idx:instance_idx+1].values
        
        # Create counterfactual explainer
        cf = CounterFactual(
            self.model.predict,
            shape=instance.shape,
            target_proba=0.5,
            tol=0.01,
            max_iter=1000
        )
        
        # Generate counterfactual
        explanation = cf.explain(instance)
        
        if explanation.cf is not None:
            cf_instance = explanation.cf['X'][0]
            changes = {
                self.feature_names[i]: {
                    'original': instance[0][i],
                    'counterfactual': cf_instance[i],
                    'change': cf_instance[i] - instance[0][i]
                }
                for i in range(len(self.feature_names))
                if abs(cf_instance[i] - instance[0][i]) > 1e-6
            }
        else:
            changes = None
            
        return ExplanationResult(
            explanation_type="counterfactual",
            local_explanations=[{
                "instance_idx": instance_idx,
                "changes_needed": changes,
                "counterfactual_found": explanation.cf is not None
            }]
        )
        
    def explain_predictions_batch(
        self,
        X: pd.DataFrame,
        sample_size: int = 100,
        methods: List[str] = ["shap", "lime"]
    ) -> Dict[str, List[ExplanationResult]]:
        """
        Generate explanations for a batch of predictions.
        
        Args:
            X: Input data
            sample_size: Number of instances to explain
            methods: List of explanation methods to use
            
        Returns:
            Dictionary mapping methods to explanation results
        """
        results = {}
        
        # Sample indices
        indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
        
        for method in methods:
            method_results = []
            for idx in indices:
                try:
                    exp = self.explain_local(X, idx, method)
                    method_results.append(exp)
                except Exception as e:
                    logger.warning(f"Failed to explain instance {idx} with {method}: {str(e)}")
                    
            results[method] = method_results
            
        return results
        
    def generate_report(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series = None,
        output_path: str = "explanation_report.html"
    ):
        """
        Generate comprehensive explainability report.
        
        Args:
            X_train: Training data
            X_test: Test data
            y_train: Training labels
            output_path: Path to save HTML report
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explainability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ font-weight: bold; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Model Explainability Report</h1>
            <p>Generated: {timestamp}</p>
        """
        
        # Global explanations
        global_exp = self.explain_global(X_train, y_train)
        
        html_content += f"""
            <div class="section">
                <h2>Global Feature Importance</h2>
                <p>{global_exp.summary}</p>
                <table>
                    <tr><th>Feature</th><th>Importance</th></tr>
        """
        
        for feat, imp in sorted(
            global_exp.global_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            html_content += f"<tr><td>{feat}</td><td>{imp:.4f}</td></tr>"
            
        html_content += """
                </table>
            </div>
        """
        
        # Sample local explanations
        local_exps = self.explain_predictions_batch(X_test, sample_size=5, methods=["shap"])
        
        html_content += """
            <div class="section">
                <h2>Sample Local Explanations</h2>
        """
        
        for i, exp in enumerate(local_exps.get("shap", [])[:3]):
            if exp.local_explanations:
                local = exp.local_explanations[0]
                html_content += f"""
                <h3>Instance {local['instance_idx']}</h3>
                <p>Prediction: {local['prediction']:.4f}</p>
                <table>
                    <tr><th>Feature</th><th>Contribution</th></tr>
                """
                
                for feat, contrib in sorted(
                    local['feature_contributions'].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]:
                    html_content += f"<tr><td>{feat}</td><td>{contrib:.4f}</td></tr>"
                    
                html_content += "</table>"
                
        html_content += """
            </div>
        </body>
        </html>
        """
        
        html_content = html_content.format(
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Explainability report saved to {output_path}")


# ====================== Cost Optimization ======================

@dataclass
class ResourceUsage:
    """Container for resource usage metrics."""
    
    timestamp: datetime
    cpu_usage: float  # Percentage
    memory_usage: float  # GB
    storage_usage: float  # GB
    network_ingress: float  # GB
    network_egress: float  # GB
    gpu_usage: Optional[float] = None  # Percentage
    requests_count: int = 0
    

@dataclass
class CostEstimate:
    """Container for cost estimates."""
    
    period: str  # daily, weekly, monthly
    compute_cost: float
    storage_cost: float
    network_cost: float
    total_cost: float
    cost_per_prediction: float
    recommendations: List[str] = field(default_factory=list)
    

class CloudCostOptimizer:
    """
    Multi-cloud cost optimization system.
    """
    
    def __init__(self, cloud_provider: str = "aws"):
        """
        Initialize cost optimizer.
        
        Args:
            cloud_provider: Cloud provider (aws, azure, gcp)
        """
        self.cloud_provider = cloud_provider
        self.usage_history = []
        self.cost_history = []
        
        # Initialize cloud clients
        if cloud_provider == "aws":
            self._init_aws_clients()
        elif cloud_provider == "azure":
            self._init_azure_clients()
        elif cloud_provider == "gcp":
            self._init_gcp_clients()
            
        # Cost metrics for Prometheus
        self.registry = CollectorRegistry()
        self.cost_gauge = Gauge(
            'cloud_cost_usd',
            'Cloud costs in USD',
            ['resource_type', 'period'],
            registry=self.registry
        )
        
    def _init_aws_clients(self):
        """Initialize AWS clients."""
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for AWS. Install with: pip install boto3")
        self.cloudwatch = boto3.client('cloudwatch')
        self.ce_client = boto3.client('ce')  # Cost Explorer
        self.ec2_client = boto3.client('ec2')

    def _init_azure_clients(self):
        """Initialize Azure clients."""
        if not AZURE_AVAILABLE:
            raise ImportError("Azure SDK is required. Install with: pip install azure-monitor-query azure-identity")
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()

        self.metrics_client = MetricsQueryClient(credential)
        self.logs_client = LogsQueryClient(credential)

    def _init_gcp_clients(self):
        """Initialize GCP clients."""
        if not GCP_AVAILABLE:
            raise ImportError("Google Cloud SDK is required. Install with: pip install google-cloud-monitoring google-cloud-billing")
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.billing_client = billing_v1.CloudBillingClient()
        
    def collect_usage_metrics(
        self,
        resource_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> ResourceUsage:
        """
        Collect resource usage metrics.
        
        Args:
            resource_id: Resource identifier
            start_time: Start time for metrics
            end_time: End time for metrics
            
        Returns:
            ResourceUsage object
        """
        if self.cloud_provider == "aws":
            return self._collect_aws_metrics(resource_id, start_time, end_time)
        elif self.cloud_provider == "azure":
            return self._collect_azure_metrics(resource_id, start_time, end_time)
        elif self.cloud_provider == "gcp":
            return self._collect_gcp_metrics(resource_id, start_time, end_time)
            
    def _collect_aws_metrics(
        self,
        instance_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> ResourceUsage:
        """Collect AWS metrics."""
        metrics = {}
        
        # CPU Utilization
        cpu_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average']
        )
        metrics['cpu_usage'] = np.mean([
            point['Average'] for point in cpu_response['Datapoints']
        ]) if cpu_response['Datapoints'] else 0
        
        # Memory (requires CloudWatch agent)
        memory_response = self.cloudwatch.get_metric_statistics(
            Namespace='CWAgent',
            MetricName='mem_used_percent',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average']
        )
        metrics['memory_usage'] = np.mean([
            point['Average'] for point in memory_response['Datapoints']
        ]) if memory_response['Datapoints'] else 0
        
        # Network
        network_in = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='NetworkIn',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Sum']
        )
        metrics['network_ingress'] = sum([
            point['Sum'] for point in network_in['Datapoints']
        ]) / (1024**3) if network_in['Datapoints'] else 0  # Convert to GB
        
        return ResourceUsage(
            timestamp=end_time,
            cpu_usage=metrics['cpu_usage'],
            memory_usage=metrics['memory_usage'],
            storage_usage=0,  # Would need EBS metrics
            network_ingress=metrics['network_ingress'],
            network_egress=0,
            requests_count=0
        )
        
    def _collect_azure_metrics(
        self,
        resource_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> ResourceUsage:
        """Collect Azure metrics."""
        # Implementation for Azure metrics
        return ResourceUsage(
            timestamp=end_time,
            cpu_usage=0,
            memory_usage=0,
            storage_usage=0,
            network_ingress=0,
            network_egress=0
        )
        
    def _collect_gcp_metrics(
        self,
        resource_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> ResourceUsage:
        """Collect GCP metrics."""
        # Implementation for GCP metrics
        return ResourceUsage(
            timestamp=end_time,
            cpu_usage=0,
            memory_usage=0,
            storage_usage=0,
            network_ingress=0,
            network_egress=0
        )
        
    def estimate_costs(
        self,
        usage_data: List[ResourceUsage],
        period: str = "daily"
    ) -> CostEstimate:
        """
        Estimate costs based on usage.
        
        Args:
            usage_data: List of resource usage data
            period: Cost period (daily, weekly, monthly)
            
        Returns:
            CostEstimate object

        Raises:
            ValueError: If usage_data is empty
        """
        if not usage_data:
            raise ValueError("usage_data cannot be empty")

        # Cloud provider pricing (simplified)
        pricing = {
            "aws": {
                "compute_per_hour": 0.0464,  # t3.medium
                "memory_per_gb_hour": 0.005,
                "storage_per_gb_month": 0.10,
                "network_per_gb": 0.09
            },
            "azure": {
                "compute_per_hour": 0.0416,
                "memory_per_gb_hour": 0.0045,
                "storage_per_gb_month": 0.08,
                "network_per_gb": 0.087
            },
            "gcp": {
                "compute_per_hour": 0.0475,
                "memory_per_gb_hour": 0.0055,
                "storage_per_gb_month": 0.12,
                "network_per_gb": 0.085
            }
        }
        
        provider_pricing = pricing.get(self.cloud_provider, pricing["aws"])
        
        # Calculate average usage
        avg_cpu = np.mean([u.cpu_usage for u in usage_data])
        avg_memory = np.mean([u.memory_usage for u in usage_data])
        total_storage = np.mean([u.storage_usage for u in usage_data])
        total_network = sum([u.network_ingress + u.network_egress for u in usage_data])
        total_requests = sum([u.requests_count for u in usage_data])
        
        # Calculate costs
        hours = {"daily": 24, "weekly": 168, "monthly": 720}[period]
        
        compute_cost = avg_cpu / 100 * provider_pricing["compute_per_hour"] * hours
        memory_cost = avg_memory * provider_pricing["memory_per_gb_hour"] * hours
        storage_cost = total_storage * provider_pricing["storage_per_gb_month"] / 30 * (hours / 24)
        network_cost = total_network * provider_pricing["network_per_gb"]
        
        total_cost = compute_cost + memory_cost + storage_cost + network_cost
        cost_per_prediction = total_cost / max(total_requests, 1)
        
        # Generate recommendations
        recommendations = []
        
        if avg_cpu < 30:
            recommendations.append("Consider downsizing compute instances - CPU usage is low")
        elif avg_cpu > 80:
            recommendations.append("Consider scaling horizontally - CPU usage is high")
            
        if avg_memory < 40:
            recommendations.append("Consider reducing memory allocation")
            
        if total_network > 100:  # GB
            recommendations.append("High network usage - consider data caching or CDN")
            
        if cost_per_prediction > 0.01:
            recommendations.append("High cost per prediction - consider batch processing")
            
        # Update Prometheus metrics
        self.cost_gauge.labels(resource_type="compute", period=period).set(compute_cost)
        self.cost_gauge.labels(resource_type="storage", period=period).set(storage_cost)
        self.cost_gauge.labels(resource_type="network", period=period).set(network_cost)
        self.cost_gauge.labels(resource_type="total", period=period).set(total_cost)
        
        return CostEstimate(
            period=period,
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            network_cost=network_cost,
            total_cost=total_cost,
            cost_per_prediction=cost_per_prediction,
            recommendations=recommendations
        )
        
    def optimize_resources(
        self,
        current_config: Dict[str, Any],
        usage_history: List[ResourceUsage],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize resource allocation based on usage patterns.
        
        Args:
            current_config: Current resource configuration
            usage_history: Historical usage data
            constraints: Optimization constraints
            
        Returns:
            Optimized configuration
        """
        # Analyze usage patterns
        cpu_percentiles = np.percentile(
            [u.cpu_usage for u in usage_history],
            [50, 75, 90, 95]
        )
        memory_percentiles = np.percentile(
            [u.memory_usage for u in usage_history],
            [50, 75, 90, 95]
        )
        
        optimized_config = current_config.copy()
        
        # CPU optimization
        target_cpu_utilization = constraints.get("target_cpu_utilization", 70)
        if cpu_percentiles[3] < target_cpu_utilization - 20:  # P95 < 50%
            optimized_config["instance_type"] = self._get_smaller_instance(
                current_config["instance_type"]
            )
        elif cpu_percentiles[1] > target_cpu_utilization + 10:  # P75 > 80%
            optimized_config["instance_type"] = self._get_larger_instance(
                current_config["instance_type"]
            )
            
        # Auto-scaling configuration
        optimized_config["auto_scaling"] = {
            "min_instances": max(1, int(cpu_percentiles[0] / target_cpu_utilization)),
            "max_instances": max(2, int(cpu_percentiles[3] / target_cpu_utilization) + 1),
            "target_cpu": target_cpu_utilization
        }
        
        # Storage optimization
        storage_usage_trend = np.polyfit(
            range(len(usage_history)),
            [u.storage_usage for u in usage_history],
            1
        )[0]
        
        if storage_usage_trend > 0:
            # Growing storage needs
            optimized_config["storage_size"] = int(
                max([u.storage_usage for u in usage_history]) * 1.5
            )
            
        # Spot/Preemptible instances for batch workloads
        if constraints.get("allow_spot_instances", False):
            avg_duration = self._estimate_workload_duration(usage_history)
            if avg_duration < 60:  # Less than 60 minutes average
                optimized_config["use_spot_instances"] = True
                optimized_config["spot_max_price"] = constraints.get(
                    "spot_max_price",
                    current_config.get("on_demand_price", 0.05) * 0.7
                )
                
        return optimized_config
        
    def _get_smaller_instance(self, current_type: str) -> str:
        """Get smaller instance type."""
        instance_families = {
            "aws": {
                "t3.medium": "t3.small",
                "t3.large": "t3.medium",
                "t3.xlarge": "t3.large",
                "m5.large": "t3.large",
                "m5.xlarge": "m5.large"
            },
            "azure": {
                "Standard_B2s": "Standard_B1s",
                "Standard_B2ms": "Standard_B2s",
                "Standard_D2s_v3": "Standard_B2ms"
            },
            "gcp": {
                "n1-standard-2": "n1-standard-1",
                "n1-standard-4": "n1-standard-2",
                "n1-highmem-2": "n1-standard-2"
            }
        }
        
        return instance_families.get(self.cloud_provider, {}).get(
            current_type,
            current_type
        )
        
    def _get_larger_instance(self, current_type: str) -> str:
        """Get larger instance type."""
        instance_families = {
            "aws": {
                "t3.small": "t3.medium",
                "t3.medium": "t3.large",
                "t3.large": "t3.xlarge",
                "m5.large": "m5.xlarge",
                "m5.xlarge": "m5.2xlarge"
            },
            "azure": {
                "Standard_B1s": "Standard_B2s",
                "Standard_B2s": "Standard_B2ms",
                "Standard_B2ms": "Standard_D2s_v3"
            },
            "gcp": {
                "n1-standard-1": "n1-standard-2",
                "n1-standard-2": "n1-standard-4",
                "n1-standard-4": "n1-standard-8"
            }
        }
        
        return instance_families.get(self.cloud_provider, {}).get(
            current_type,
            current_type
        )
        
    def _estimate_workload_duration(self, usage_history: List[ResourceUsage]) -> float:
        """Estimate average workload duration in minutes."""
        # Simplified: estimate based on CPU usage patterns
        cpu_usage = [u.cpu_usage for u in usage_history]
        
        # Find periods of high usage
        high_usage_threshold = 50
        high_usage_periods = []
        current_period = 0
        
        for usage in cpu_usage:
            if usage > high_usage_threshold:
                current_period += 1
            else:
                if current_period > 0:
                    high_usage_periods.append(current_period)
                current_period = 0
                
        if high_usage_periods:
            return np.mean(high_usage_periods) * 5  # Assuming 5-minute intervals
        return 30  # Default 30 minutes
        
    def generate_cost_report(
        self,
        usage_data: List[ResourceUsage],
        output_path: str = "cost_report.html"
    ):
        """
        Generate cost optimization report.
        
        Args:
            usage_data: List of resource usage data
            output_path: Path to save HTML report
        """
        # Calculate costs for different periods
        daily_cost = self.estimate_costs(usage_data[-24:], "daily")
        weekly_cost = self.estimate_costs(usage_data[-168:], "weekly")
        monthly_cost = self.estimate_costs(usage_data, "monthly")
        
        # Create visualizations with Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cost Breakdown", "Usage Trends", "Cost Trends", "Optimization Opportunities")
        )
        
        # Cost breakdown pie chart
        labels = ['Compute', 'Storage', 'Network']
        values = [monthly_cost.compute_cost, monthly_cost.storage_cost, monthly_cost.network_cost]
        
        fig.add_trace(
            go.Pie(labels=labels, values=values, hole=0.3),
            row=1, col=1
        )
        
        # Usage trends
        timestamps = [u.timestamp for u in usage_data]
        fig.add_trace(
            go.Scatter(x=timestamps, y=[u.cpu_usage for u in usage_data], name="CPU %"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=[u.memory_usage for u in usage_data], name="Memory GB"),
            row=1, col=2
        )
        
        # Cost trends
        cost_history = []
        for i in range(0, len(usage_data), 24):
            batch = usage_data[i:i+24]
            if batch:
                cost = self.estimate_costs(batch, "daily")
                cost_history.append(cost.total_cost)
                
        fig.add_trace(
            go.Scatter(y=cost_history, name="Daily Cost ($)"),
            row=2, col=1
        )
        
        # Optimization opportunities
        savings_potential = {
            "Right-sizing": monthly_cost.total_cost * 0.2,
            "Reserved Instances": monthly_cost.total_cost * 0.3,
            "Spot Instances": monthly_cost.total_cost * 0.15,
            "Auto-scaling": monthly_cost.total_cost * 0.1
        }
        
        fig.add_trace(
            go.Bar(x=list(savings_potential.keys()), y=list(savings_potential.values())),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Cost Optimization Report")
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cost Optimization Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .metric {{ font-weight: bold; color: #0066cc; }}
                .cost {{ font-size: 24px; font-weight: bold; }}
                .recommendation {{ 
                    background-color: #f0f8ff; 
                    padding: 10px; 
                    margin: 5px 0; 
                    border-left: 3px solid #0066cc; 
                }}
            </style>
        </head>
        <body>
            <h1>Cloud Cost Optimization Report</h1>
            <p>Generated: {datetime.utcnow().isoformat()}</p>
            
            <h2>Cost Summary</h2>
            <p>Daily Cost: <span class="cost">${daily_cost.total_cost:.2f}</span></p>
            <p>Weekly Cost: <span class="cost">${weekly_cost.total_cost:.2f}</span></p>
            <p>Monthly Cost: <span class="cost">${monthly_cost.total_cost:.2f}</span></p>
            <p>Cost per Prediction: <span class="metric">${monthly_cost.cost_per_prediction:.4f}</span></p>
            
            <h2>Recommendations</h2>
        """
        
        for rec in monthly_cost.recommendations:
            html_content += f'<div class="recommendation">{rec}</div>'
            
        html_content += f"""
            <h2>Detailed Analysis</h2>
            <div id="plotly-div"></div>
            
            <script>
                var plotlyData = {fig.to_json()};
                Plotly.newPlot('plotly-div', plotlyData.data, plotlyData.layout);
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Cost report saved to {output_path}")


# ====================== Example Usage ======================

def main():
    """Example usage of explainability and cost optimization."""
    
    # Generate sample model and data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'feature4': np.random.randn(1000),
        'feature5': np.random.randn(1000)
    })
    y = (X['feature1'] + X['feature2'] * 0.5 + np.random.randn(1000) * 0.1 > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # === Explainability ===
    print("="*60)
    print("MODEL EXPLAINABILITY")
    print("="*60)
    
    explainer = ModelExplainer(
        model=model,
        feature_names=list(X.columns),
        class_names=["Class 0", "Class 1"]
    )
    
    # Global explanations
    global_exp = explainer.explain_global(X_train, y_train, method="shap")
    print("\nGlobal Feature Importance:")
    for feat, imp in sorted(
        global_exp.global_importance.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {feat}: {imp:.4f}")
        
    # Local explanation
    local_exp = explainer.explain_local(X_test, 0, method="shap")
    print("\nLocal Explanation for Instance 0:")
    if local_exp.local_explanations:
        for feat, contrib in list(
            local_exp.local_explanations[0]['feature_contributions'].items()
        )[:3]:
            print(f"  {feat}: {contrib:.4f}")
            
    # Generate report
    explainer.generate_report(X_train, X_test, y_train, "model_explanation.html")
    print("\nExplanation report saved to model_explanation.html")
    
    # === Cost Optimization ===
    print("\n" + "="*60)
    print("COST OPTIMIZATION")
    print("="*60)
    
    optimizer = CloudCostOptimizer(cloud_provider="aws")
    
    # Generate sample usage data
    usage_data = []
    for i in range(168):  # 1 week of hourly data
        usage = ResourceUsage(
            timestamp=datetime.utcnow() - timedelta(hours=168-i),
            cpu_usage=50 + np.random.randn() * 20,
            memory_usage=4 + np.random.randn() * 1,
            storage_usage=100,
            network_ingress=np.random.exponential(1),
            network_egress=np.random.exponential(0.5),
            requests_count=np.random.poisson(1000)
        )
        usage_data.append(usage)
        
    # Estimate costs
    cost_estimate = optimizer.estimate_costs(usage_data, "weekly")
    print(f"\nWeekly Cost Estimate: ${cost_estimate.total_cost:.2f}")
    print(f"Cost per Prediction: ${cost_estimate.cost_per_prediction:.4f}")
    
    print("\nRecommendations:")
    for rec in cost_estimate.recommendations:
        print(f"  - {rec}")
        
    # Optimize configuration
    current_config = {
        "instance_type": "t3.large",
        "instance_count": 2,
        "storage_size": 100
    }
    
    optimized = optimizer.optimize_resources(
        current_config,
        usage_data,
        {"target_cpu_utilization": 70, "allow_spot_instances": True}
    )
    
    print("\nOptimized Configuration:")
    print(f"  Instance Type: {optimized['instance_type']}")
    print(f"  Auto-scaling: {optimized['auto_scaling']}")
    print(f"  Use Spot: {optimized.get('use_spot_instances', False)}")
    
    # Generate cost report
    optimizer.generate_cost_report(usage_data, "cost_optimization.html")
    print("\nCost report saved to cost_optimization.html")


if __name__ == "__main__":
    main()
