"""
Load Testing and Performance Benchmarking Suite
================================================
Comprehensive tools for testing API performance and model inference speed.
"""

import os
import time
import json
import asyncio
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random

import aiohttp
import numpy as np
from locust import HttpUser, task, between, LoadTestShape
from locust.env import Environment
from locust.log import setup_logging
import matplotlib.pyplot as plt
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


# ====================== Performance Metrics ======================


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time_seconds: float = 0.0
    min_response_time_ms: float = float("inf")
    max_response_time_ms: float = 0.0
    mean_response_time_ms: float = 0.0
    median_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    response_times: List[float] = None
    errors: Dict[str, int] = None

    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.errors is None:
            self.errors = {}

    def calculate_statistics(self):
        """Calculate statistical metrics from response times."""
        if not self.response_times:
            return

        self.min_response_time_ms = min(self.response_times)
        self.max_response_time_ms = max(self.response_times)
        self.mean_response_time_ms = statistics.mean(self.response_times)
        self.median_response_time_ms = statistics.median(self.response_times)

        sorted_times = sorted(self.response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)

        self.p95_response_time_ms = (
            sorted_times[p95_index] if p95_index < len(sorted_times) else self.max_response_time_ms
        )
        self.p99_response_time_ms = (
            sorted_times[p99_index] if p99_index < len(sorted_times) else self.max_response_time_ms
        )

        if self.total_time_seconds > 0:
            self.requests_per_second = self.total_requests / self.total_time_seconds

        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_time_seconds": self.total_time_seconds,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "mean_response_time_ms": self.mean_response_time_ms,
            "median_response_time_ms": self.median_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "requests_per_second": self.requests_per_second,
            "error_rate": self.error_rate,
            "errors": self.errors,
        }


# ====================== Load Testing with Locust ======================


class GLMModelUser(HttpUser):
    """Locust user for load testing GLM model API."""

    wait_time = between(1, 3)

    def on_start(self):
        """Setup before tests."""
        self.test_features = self._generate_test_features()

    def _generate_test_features(self) -> List[Dict[str, float]]:
        """Generate random test features."""
        features = []
        for _ in range(100):
            features.append(
                {
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                    "feature3": np.random.randn(),
                    "feature4": np.random.randn(),
                    "feature5": np.random.randn(),
                }
            )
        return features

    @task(weight=80)
    def predict_single(self):
        """Test single prediction endpoint."""
        features = random.choice(self.test_features)

        with self.client.post(
            "/predict", json={"features": features}, catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "prediction" in result:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(weight=15)
    def predict_batch(self):
        """Test batch prediction endpoint."""
        batch_size = random.randint(5, 20)
        batch_features = random.sample(self.test_features, batch_size)

        with self.client.post(
            "/predict/batch", json={"data": batch_features}, catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "predictions" in result and len(result["predictions"]) == batch_size:
                        response.success()
                    else:
                        response.failure("Invalid batch response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(weight=5)
    def get_model_info(self):
        """Test model info endpoint."""
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class StagesShape(LoadTestShape):
    """
    Custom load test shape with multiple stages.
    """

    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},
        {"duration": 120, "users": 50, "spawn_rate": 5},
        {"duration": 180, "users": 100, "spawn_rate": 10},
        {"duration": 240, "users": 200, "spawn_rate": 10},
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 360, "users": 10, "spawn_rate": 2},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None


# ====================== Async Performance Tester ======================


class AsyncPerformanceTester:
    """
    Asynchronous performance testing for high-concurrency scenarios.
    """

    def __init__(self, base_url: str, max_concurrent: int = 100):
        """
        Initialize async tester.

        Args:
            base_url: API base URL
            max_concurrent: Maximum concurrent requests
        """
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _make_request(
        self, session: aiohttp.ClientSession, endpoint: str, method: str = "POST", data: Dict = None
    ) -> Tuple[float, int, Optional[Dict]]:
        """
        Make a single async request.

        Returns:
            Tuple of (response_time_ms, status_code, response_data)
        """
        async with self.semaphore:
            url = f"{self.base_url}{endpoint}"

            start_time = time.perf_counter()
            try:
                async with session.request(
                    method, url, json=data, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response_time = (time.perf_counter() - start_time) * 1000

                    if response.status == 200:
                        result = await response.json()
                        return response_time, response.status, result
                    else:
                        return response_time, response.status, None

            except Exception as e:
                response_time = (time.perf_counter() - start_time) * 1000
                return response_time, 0, {"error": str(e)}

    async def run_load_test(
        self,
        num_requests: int,
        endpoint: str = "/predict",
        method: str = "POST",
        data_generator=None,
    ) -> PerformanceMetrics:
        """
        Run async load test.

        Args:
            num_requests: Total number of requests
            endpoint: API endpoint
            method: HTTP method
            data_generator: Function to generate request data

        Returns:
            Performance metrics
        """
        metrics = PerformanceMetrics()
        start_time = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                data = data_generator() if data_generator else None
                task = self._make_request(session, endpoint, method, data)
                tasks.append(task)

            # Execute all tasks
            results = await asyncio.gather(*tasks)

        # Process results
        for response_time, status_code, response_data in results:
            metrics.total_requests += 1
            metrics.response_times.append(response_time)

            if status_code == 200:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                error_key = f"status_{status_code}"
                metrics.errors[error_key] = metrics.errors.get(error_key, 0) + 1

        metrics.total_time_seconds = time.perf_counter() - start_time
        metrics.calculate_statistics()

        return metrics

    async def run_stress_test(
        self,
        duration_seconds: int,
        requests_per_second: int,
        endpoint: str = "/predict",
        data_generator=None,
    ) -> PerformanceMetrics:
        """
        Run stress test with constant load.

        Args:
            duration_seconds: Test duration
            requests_per_second: Target RPS
            endpoint: API endpoint
            data_generator: Function to generate request data

        Returns:
            Performance metrics
        """
        metrics = PerformanceMetrics()
        start_time = time.perf_counter()
        request_interval = 1.0 / requests_per_second

        async with aiohttp.ClientSession() as session:
            tasks = []
            current_time = 0

            while current_time < duration_seconds:
                data = data_generator() if data_generator else None
                task = asyncio.create_task(self._make_request(session, endpoint, "POST", data))
                tasks.append(task)

                await asyncio.sleep(request_interval)
                current_time = time.perf_counter() - start_time

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

        # Process results
        for response_time, status_code, response_data in results:
            metrics.total_requests += 1
            metrics.response_times.append(response_time)

            if status_code == 200:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1

        metrics.total_time_seconds = time.perf_counter() - start_time
        metrics.calculate_statistics()

        return metrics


# ====================== Benchmark Suite ======================


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for model performance.
    """

    def __init__(self, api_url: str):
        """
        Initialize benchmark suite.

        Args:
            api_url: API URL to test
        """
        self.api_url = api_url
        self.results = []

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.latency_gauge = Gauge(
            "benchmark_latency_ms",
            "API latency in milliseconds",
            ["endpoint", "percentile"],
            registry=self.registry,
        )
        self.throughput_gauge = Gauge(
            "benchmark_throughput_rps",
            "Throughput in requests per second",
            ["endpoint"],
            registry=self.registry,
        )
        self.error_rate_gauge = Gauge(
            "benchmark_error_rate", "Error rate", ["endpoint"], registry=self.registry
        )

    def run_benchmark(self, test_name: str, test_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run comprehensive benchmark.

        Args:
            test_name: Name of the benchmark
            test_configs: List of test configurations

        Returns:
            Benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Running Benchmark: {test_name}")
        print(f"{'='*60}\n")

        results = {"test_name": test_name, "timestamp": datetime.utcnow().isoformat(), "tests": []}

        for config in test_configs:
            print(f"Running: {config['name']}")

            if config["type"] == "load":
                metrics = self._run_load_test(config)
            elif config["type"] == "stress":
                metrics = self._run_stress_test(config)
            elif config["type"] == "spike":
                metrics = self._run_spike_test(config)
            elif config["type"] == "soak":
                metrics = self._run_soak_test(config)
            else:
                print(f"Unknown test type: {config['type']}")
                continue

            test_result = {"config": config, "metrics": metrics.to_dict()}
            results["tests"].append(test_result)

            # Update Prometheus metrics
            self._update_prometheus_metrics(config["name"], metrics)

            # Print summary
            self._print_summary(config["name"], metrics)

        # Generate report
        self._generate_report(results)

        return results

    def _run_load_test(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run load test."""
        tester = AsyncPerformanceTester(self.api_url, config.get("max_concurrent", 100))

        # Create data generator
        def data_generator():
            return {"features": {f"feature{i}": np.random.randn() for i in range(1, 6)}}

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        metrics = loop.run_until_complete(
            tester.run_load_test(
                num_requests=config.get("num_requests", 1000),
                endpoint=config.get("endpoint", "/predict"),
                data_generator=data_generator,
            )
        )

        loop.close()
        return metrics

    def _run_stress_test(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run stress test."""
        tester = AsyncPerformanceTester(self.api_url, config.get("max_concurrent", 200))

        def data_generator():
            return {"features": {f"feature{i}": np.random.randn() for i in range(1, 6)}}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        metrics = loop.run_until_complete(
            tester.run_stress_test(
                duration_seconds=config.get("duration", 60),
                requests_per_second=config.get("rps", 100),
                endpoint=config.get("endpoint", "/predict"),
                data_generator=data_generator,
            )
        )

        loop.close()
        return metrics

    def _run_spike_test(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run spike test with sudden traffic increase."""
        metrics = PerformanceMetrics()

        # Normal load -> Spike -> Normal load
        phases = [
            {"duration": 30, "rps": 10},  # Normal
            {"duration": 10, "rps": 200},  # Spike
            {"duration": 30, "rps": 10},  # Recovery
        ]

        for phase in phases:
            tester = AsyncPerformanceTester(self.api_url, 300)

            def data_generator():
                return {"features": {f"feature{i}": np.random.randn() for i in range(1, 6)}}

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            phase_metrics = loop.run_until_complete(
                tester.run_stress_test(
                    duration_seconds=phase["duration"],
                    requests_per_second=phase["rps"],
                    endpoint=config.get("endpoint", "/predict"),
                    data_generator=data_generator,
                )
            )

            loop.close()

            # Aggregate metrics
            metrics.total_requests += phase_metrics.total_requests
            metrics.successful_requests += phase_metrics.successful_requests
            metrics.failed_requests += phase_metrics.failed_requests
            metrics.response_times.extend(phase_metrics.response_times)

        metrics.calculate_statistics()
        return metrics

    def _run_soak_test(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run soak test for extended duration."""
        # Run at moderate load for extended time
        return self._run_stress_test(
            {
                "duration": config.get("duration", 3600),  # 1 hour default
                "rps": config.get("rps", 50),
                "endpoint": config.get("endpoint", "/predict"),
                "max_concurrent": 100,
            }
        )

    def _update_prometheus_metrics(self, test_name: str, metrics: PerformanceMetrics):
        """Update Prometheus metrics."""
        endpoint = test_name.replace(" ", "_").lower()

        self.latency_gauge.labels(endpoint=endpoint, percentile="p50").set(
            metrics.median_response_time_ms
        )
        self.latency_gauge.labels(endpoint=endpoint, percentile="p95").set(
            metrics.p95_response_time_ms
        )
        self.latency_gauge.labels(endpoint=endpoint, percentile="p99").set(
            metrics.p99_response_time_ms
        )
        self.throughput_gauge.labels(endpoint=endpoint).set(metrics.requests_per_second)
        self.error_rate_gauge.labels(endpoint=endpoint).set(metrics.error_rate)

        # Push to gateway if configured
        if os.environ.get("PROMETHEUS_GATEWAY"):
            push_to_gateway(
                os.environ["PROMETHEUS_GATEWAY"], job="benchmark", registry=self.registry
            )

    def _print_summary(self, test_name: str, metrics: PerformanceMetrics):
        """Print test summary."""
        print(f"\n{test_name} Results:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Successful: {metrics.successful_requests}")
        print(f"  Failed: {metrics.failed_requests}")
        print(f"  Error Rate: {metrics.error_rate:.2%}")
        print(f"  RPS: {metrics.requests_per_second:.2f}")
        print(f"  Response Times (ms):")
        print(f"    Min: {metrics.min_response_time_ms:.2f}")
        print(f"    Median: {metrics.median_response_time_ms:.2f}")
        print(f"    Mean: {metrics.mean_response_time_ms:.2f}")
        print(f"    P95: {metrics.p95_response_time_ms:.2f}")
        print(f"    P99: {metrics.p99_response_time_ms:.2f}")
        print(f"    Max: {metrics.max_response_time_ms:.2f}")

        if metrics.errors:
            print(f"  Errors: {metrics.errors}")

    def _generate_report(self, results: Dict[str, Any]):
        """Generate HTML report with visualizations."""
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Response time distribution
        all_response_times = []
        labels = []
        for test in results["tests"]:
            if test["metrics"]["response_times"]:
                all_response_times.append(test["metrics"]["response_times"])
                labels.append(test["config"]["name"])

        if all_response_times:
            axes[0, 0].boxplot(all_response_times, labels=labels)
            axes[0, 0].set_title("Response Time Distribution")
            axes[0, 0].set_ylabel("Response Time (ms)")
            axes[0, 0].tick_params(axis="x", rotation=45)

        # Throughput comparison
        throughputs = [test["metrics"]["requests_per_second"] for test in results["tests"]]
        test_names = [test["config"]["name"] for test in results["tests"]]

        axes[0, 1].bar(test_names, throughputs)
        axes[0, 1].set_title("Throughput Comparison")
        axes[0, 1].set_ylabel("Requests per Second")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Error rates
        error_rates = [test["metrics"]["error_rate"] * 100 for test in results["tests"]]

        axes[1, 0].bar(test_names, error_rates, color="red", alpha=0.7)
        axes[1, 0].set_title("Error Rates")
        axes[1, 0].set_ylabel("Error Rate (%)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Percentiles comparison
        percentiles = ["P50", "P95", "P99"]
        x = np.arange(len(test_names))
        width = 0.25

        for i, percentile in enumerate(["median", "p95", "p99"]):
            values = [
                test["metrics"][f"{percentile}_response_time_ms"] for test in results["tests"]
            ]
            axes[1, 1].bar(x + i * width, values, width, label=percentiles[i])

        axes[1, 1].set_title("Response Time Percentiles")
        axes[1, 1].set_ylabel("Response Time (ms)")
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels(test_names, rotation=45)
        axes[1, 1].legend()

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"benchmark_report_{timestamp}.png"
        plt.savefig(plot_filename)
        plt.close()

        # Generate HTML report
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report - {results['test_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Benchmark Report: {results['test_name']}</h1>
    <p>Generated: {results['timestamp']}</p>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Total Requests</th>
            <th>Success Rate</th>
            <th>RPS</th>
            <th>P50 (ms)</th>
            <th>P95 (ms)</th>
            <th>P99 (ms)</th>
        </tr>
"""

        for test in results["tests"]:
            metrics = test["metrics"]
            success_rate = (
                (metrics["successful_requests"] / metrics["total_requests"] * 100)
                if metrics["total_requests"] > 0
                else 0
            )

            html_report += f"""
        <tr>
            <td>{test['config']['name']}</td>
            <td>{metrics['total_requests']}</td>
            <td class="{'success' if success_rate >= 95 else 'error'}">{success_rate:.1f}%</td>
            <td>{metrics['requests_per_second']:.2f}</td>
            <td>{metrics['median_response_time_ms']:.2f}</td>
            <td>{metrics['p95_response_time_ms']:.2f}</td>
            <td>{metrics['p99_response_time_ms']:.2f}</td>
        </tr>
"""

        html_report += f"""
    </table>
    
    <h2>Visualizations</h2>
    <img src="{plot_filename}" alt="Benchmark Charts">
    
    <h2>Detailed Metrics</h2>
"""

        for test in results["tests"]:
            metrics = test["metrics"]
            html_report += f"""
    <h3>{test['config']['name']}</h3>
    <ul>
        <li><span class="metric">Total Time:</span> {metrics['total_time_seconds']:.2f} seconds</li>
        <li><span class="metric">Min Response Time:</span> {metrics['min_response_time_ms']:.2f} ms</li>
        <li><span class="metric">Max Response Time:</span> {metrics['max_response_time_ms']:.2f} ms</li>
        <li><span class="metric">Mean Response Time:</span> {metrics['mean_response_time_ms']:.2f} ms</li>
        <li><span class="metric">Error Rate:</span> {metrics['error_rate']:.2%}</li>
    </ul>
"""

        html_report += """
</body>
</html>
"""

        # Save HTML report
        report_filename = f"benchmark_report_{timestamp}.html"
        with open(report_filename, "w") as f:
            f.write(html_report)

        print(f"\nReports generated:")
        print(f"  - HTML: {report_filename}")
        print(f"  - Charts: {plot_filename}")


# ====================== Example Usage ======================


def main():
    """Run comprehensive benchmark suite."""

    api_url = "http://localhost:5000"

    # Initialize benchmark suite
    suite = BenchmarkSuite(api_url)

    # Define test configurations
    test_configs = [
        {
            "name": "Baseline Load Test",
            "type": "load",
            "num_requests": 1000,
            "max_concurrent": 50,
            "endpoint": "/predict",
        },
        {
            "name": "Stress Test",
            "type": "stress",
            "duration": 60,
            "rps": 100,
            "max_concurrent": 200,
            "endpoint": "/predict",
        },
        {"name": "Spike Test", "type": "spike", "endpoint": "/predict"},
        {
            "name": "Batch Endpoint Test",
            "type": "load",
            "num_requests": 500,
            "max_concurrent": 25,
            "endpoint": "/predict/batch",
        },
    ]

    # Run benchmark
    results = suite.run_benchmark(test_name="GLM Model API Performance", test_configs=test_configs)

    # Save results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nBenchmark completed successfully!")

    # Run Locust tests (optional)
    if False:  # Set to True to run Locust tests
        setup_logging("INFO", None)

        # Setup Environment and Runner
        env = Environment(user_classes=[GLMModelUser], shape_class=StagesShape)
        env.create_local_runner()

        # Start test
        env.runner.start_shape()

        # Run for specified time
        time.sleep(360)  # 6 minutes for all stages

        # Stop and print stats
        env.runner.quit()

        # Print statistics
        stats = env.stats
        print("\nLocust Test Results:")
        print(f"Total Requests: {stats.total.num_requests}")
        print(f"Total Failures: {stats.total.num_failures}")
        print(f"Average Response Time: {stats.total.avg_response_time:.2f} ms")
        print(f"Min Response Time: {stats.total.min_response_time:.2f} ms")
        print(f"Max Response Time: {stats.total.max_response_time:.2f} ms")


if __name__ == "__main__":
    main()
