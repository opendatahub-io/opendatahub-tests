"""EvalHub Prometheus scraping tests.

Test Plan Reference: RHAISTRAT-1507
Test Cases: TC-SCRAPE-001, TC-SCRAPE-002, TC-SCRAPE-003
"""

import json
import time
from datetime import UTC, datetime

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.service_monitor import ServiceMonitor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-prometheus-scraping"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubPrometheusScraping:
    """Tests for Prometheus scraping of EvalHub metrics (RHAISTRAT-1507)."""

    def _get_prometheus_pod(self, admin_client: DynamicClient) -> Pod:
        """Get the user-workload-monitoring Prometheus pod."""
        pods = list(
            Pod.get(
                client=admin_client,
                namespace="openshift-user-workload-monitoring",
                label_selector="app.kubernetes.io/name=prometheus",
            )
        )
        assert pods, (
            "No Prometheus pods found in openshift-user-workload-monitoring namespace. "
            "Ensure user workload monitoring is enabled."
        )
        return pods[0]

    def _query_prometheus_targets(
        self,
        admin_client: DynamicClient,
        prometheus_pod: Pod,
    ) -> dict:
        """Query Prometheus targets API and return the response."""
        exec_result = prometheus_pod.execute(
            command=[
                "curl",
                "-s",
                "http://localhost:9090/api/v1/targets",
            ],
            container="prometheus",
        )
        assert exec_result, "Failed to query Prometheus targets API"
        return json.loads(exec_result)

    def _query_prometheus_instant(
        self,
        admin_client: DynamicClient,
        prometheus_pod: Pod,
        query: str,
    ) -> dict:
        """Query Prometheus instant query API."""
        exec_result = prometheus_pod.execute(
            command=[
                "curl",
                "-s",
                "--data-urlencode",
                f"query={query}",
                "http://localhost:9090/api/v1/query",
            ],
            container="prometheus",
        )
        assert exec_result, f"Failed to query Prometheus with query: {query}"
        return json.loads(exec_result)

    def test_prometheus_discovers_evalhub_scrape_target_up(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-SCRAPE-001: Verify Prometheus discovers EvalHub as scrape target with state UP."""
        prometheus_pod = self._get_prometheus_pod(admin_client=admin_client)

        def _check_target_up() -> bool:
            """Check if EvalHub target is discovered and UP."""
            targets_response = self._query_prometheus_targets(
                admin_client=admin_client,
                prometheus_pod=prometheus_pod,
            )

            if targets_response.get("status") != "success":
                return False

            active_targets = targets_response.get("data", {}).get("activeTargets", [])
            evalhub_targets = [
                target for target in active_targets if "evalhub" in target.get("labels", {}).get("job", "").lower()
            ]

            if not evalhub_targets:
                return False

            for target in evalhub_targets:
                if target.get("health") == "up":
                    return True

            return False

        try:
            for _ in TimeoutSampler(
                wait_timeout=90,
                sleep=5,
                func=_check_target_up,
            ):
                if _check_target_up():
                    break
        except TimeoutExpiredError as err:
            targets_response = self._query_prometheus_targets(
                admin_client=admin_client,
                prometheus_pod=prometheus_pod,
            )
            msg = (
                f"Prometheus did not discover EvalHub target with status 'up' within 90 seconds. "
                f"Targets response: {json.dumps(targets_response, indent=2)}"
            )
            raise AssertionError(msg) from err

        targets_response = self._query_prometheus_targets(
            admin_client=admin_client,
            prometheus_pod=prometheus_pod,
        )
        active_targets = targets_response.get("data", {}).get("activeTargets", [])
        evalhub_targets = [
            target for target in active_targets if "evalhub" in target.get("labels", {}).get("job", "").lower()
        ]

        evalhub_target = evalhub_targets[0]

        assert evalhub_target.get("health") == "up", (
            f"Expected EvalHub target health 'up', got '{evalhub_target.get('health')}'. "
            f"Last error: {evalhub_target.get('lastError')}"
        )

        scrape_url = evalhub_target.get("scrapeUrl", "")
        assert "/metrics" in scrape_url, f"Expected scrapeUrl to contain '/metrics', got '{scrape_url}'"
        assert scrape_url.startswith("https://"), f"Expected scrapeUrl to use HTTPS scheme, got '{scrape_url}'"

        last_scrape_str = evalhub_target.get("lastScrape")
        assert last_scrape_str, "Expected 'lastScrape' field to be present in target"

        last_scrape_time = datetime.fromisoformat(date_string=last_scrape_str)
        now = datetime.now(tz=UTC)
        scrape_age_seconds = (now - last_scrape_time).total_seconds()

        assert scrape_age_seconds < 60, (
            f"Last scrape timestamp is too old: {scrape_age_seconds:.1f} seconds ago. "
            f"Expected scrape within the last 60 seconds. Last scrape: {last_scrape_str}"
        )

    def test_evalhub_metrics_queryable_via_prometheus_api(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> None:
        """TC-SCRAPE-002: Verify EvalHub metrics are queryable via Prometheus query API."""
        prometheus_pod = self._get_prometheus_pod(admin_client=admin_client)

        def _check_up_metric() -> bool:
            """Check if 'up' metric for EvalHub is available and equals 1."""
            query = 'up{job=~".*evalhub.*"}'
            try:
                query_response = self._query_prometheus_instant(
                    admin_client=admin_client,
                    prometheus_pod=prometheus_pod,
                    query=query,
                )

                if query_response.get("status") != "success":
                    return False

                results = query_response.get("data", {}).get("result", [])
                if not results:
                    return False

                for result in results:
                    value = result.get("value", [])
                    if len(value) >= 2 and value[1] == "1":
                        return True

                return False
            except json.JSONDecodeError, AssertionError:
                return False

        try:
            for _ in TimeoutSampler(
                wait_timeout=90,
                sleep=5,
                func=_check_up_metric,
            ):
                if _check_up_metric():
                    break
        except TimeoutExpiredError as err:
            msg = (
                "EvalHub 'up' metric not available in Prometheus within 90 seconds. "
                "Expected 'up{job=~\".*evalhub.*\"}' to return value 1."
            )
            raise AssertionError(msg) from err

        query_response = self._query_prometheus_instant(
            admin_client=admin_client,
            prometheus_pod=prometheus_pod,
            query='up{job=~".*evalhub.*"}',
        )

        results = query_response.get("data", {}).get("result", [])
        up_result = results[0]
        up_result.get("value", [])

        metric_labels = up_result.get("metric", {})
        assert "namespace" in metric_labels, f"Expected 'namespace' label in metric, got labels: {metric_labels.keys()}"
        assert "pod" in metric_labels or "instance" in metric_labels, (
            f"Expected 'pod' or 'instance' label in metric, got labels: {metric_labels.keys()}"
        )

        query_response = self._query_prometheus_instant(
            admin_client=admin_client,
            prometheus_pod=prometheus_pod,
            query='{job=~".*evalhub.*"}',
        )

        evalhub_metrics = query_response.get("data", {}).get("result", [])
        assert len(evalhub_metrics) > 0, (
            "No EvalHub-specific metrics found in Prometheus. "
            "Expected at least one metric with job label containing 'evalhub'."
        )

    def test_prometheus_scrape_interval_matches_servicemonitor(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-SCRAPE-003: Verify Prometheus scrape interval matches ServiceMonitor config (30s)."""
        endpoints = evalhub_service_monitor.instance.spec.endpoints
        endpoint = endpoints[0]
        assert endpoint.interval == "30s", f"Expected ServiceMonitor scrape interval '30s', got '{endpoint.interval}'"

        prometheus_pod = self._get_prometheus_pod(admin_client=admin_client)

        def _check_target_discovered() -> bool:
            """Check if EvalHub target is discovered."""
            targets_response = self._query_prometheus_targets(
                admin_client=admin_client,
                prometheus_pod=prometheus_pod,
            )
            active_targets = targets_response.get("data", {}).get("activeTargets", [])
            evalhub_targets = [
                target for target in active_targets if "evalhub" in target.get("labels", {}).get("job", "").lower()
            ]
            return len(evalhub_targets) > 0

        try:
            for _ in TimeoutSampler(
                wait_timeout=90,
                sleep=5,
                func=_check_target_discovered,
            ):
                if _check_target_discovered():
                    break
        except TimeoutExpiredError as err:
            raise AssertionError("EvalHub target not discovered by Prometheus within 90 seconds") from err

        targets_response = self._query_prometheus_targets(
            admin_client=admin_client,
            prometheus_pod=prometheus_pod,
        )

        active_targets = targets_response.get("data", {}).get("activeTargets", [])
        evalhub_targets = [
            target for target in active_targets if "evalhub" in target.get("labels", {}).get("job", "").lower()
        ]

        evalhub_target = evalhub_targets[0]
        scrape_interval = evalhub_target.get("scrapeInterval")

        assert scrape_interval == "30s", (
            f"Expected Prometheus scrape interval '30s' for EvalHub target, got '{scrape_interval}'"
        )

        initial_scrape_str = evalhub_target.get("lastScrape")
        initial_scrape_time = datetime.fromisoformat(date_string=initial_scrape_str)

        time.sleep(35)

        targets_response = self._query_prometheus_targets(
            admin_client=admin_client,
            prometheus_pod=prometheus_pod,
        )
        active_targets = targets_response.get("data", {}).get("activeTargets", [])
        evalhub_targets = [
            target for target in active_targets if "evalhub" in target.get("labels", {}).get("job", "").lower()
        ]

        updated_target = evalhub_targets[0]
        updated_scrape_str = updated_target.get("lastScrape")
        updated_scrape_time = datetime.fromisoformat(date_string=updated_scrape_str)

        scrape_delta_seconds = (updated_scrape_time - initial_scrape_time).total_seconds()
        assert 25 <= scrape_delta_seconds <= 40, (
            f"Expected scrape interval ~30 seconds, observed {scrape_delta_seconds:.1f} seconds"
        )
