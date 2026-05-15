"""EvalHub ServiceMonitor E2E workflow tests.

Test Plan Reference: RHAISTRAT-1507
Test Cases: TC-E2E-001, TC-E2E-002, TC-E2E-003
"""

import json
import time

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from ocp_resources.service_monitor import ServiceMonitor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import EVALHUB_APP_LABEL
from utilities.constants import Timeout

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-e2e-happy-path"},
        ),
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.model_explainability
class TestEvalHubE2EHappyPath:
    """E2E: Deploy EvalHub → ServiceMonitor → Prometheus scrapes → metrics queryable (RHAISTRAT-1507)."""

    def test_deploy_evalhub_to_queryable_metrics(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-E2E-001: Validate complete happy path from EvalHub deployment to queryable metrics."""
        assert evalhub_deployment.instance.status.availableReplicas > 0, (
            f"EvalHub deployment '{evalhub_deployment.name}' has no available replicas"
        )

        endpoints = evalhub_service_monitor.instance.spec.endpoints
        endpoint = endpoints[0]
        assert endpoint.port == "https" and endpoint.path == "/metrics" and endpoint.interval == "30s"

        owner_refs = evalhub_service_monitor.instance.metadata.ownerReferences
        evalhub_owner_ref = next((ref for ref in owner_refs if ref.kind == "EvalHub"), None)
        assert evalhub_owner_ref and evalhub_owner_ref.name == evalhub_cr.name

        prometheus_pods = list(
            Pod.get(
                client=admin_client,
                namespace="openshift-user-workload-monitoring",
                label_selector="app.kubernetes.io/name=prometheus",
            )
        )
        if not prometheus_pods:
            pytest.skip("User workload monitoring not enabled - cannot verify Prometheus scraping")

        prometheus_pod = prometheus_pods[0]
        evalhub_up_query = f'up{{namespace="{model_namespace.name}", job=~".*evalhub.*"}}'

        def _check_metric_up() -> bool:
            try:
                exec_result = prometheus_pod.execute(
                    command=[
                        "curl",
                        "-s",
                        "--data-urlencode",
                        f"query={evalhub_up_query}",
                        "http://localhost:9090/api/v1/query",
                    ],
                    container="prometheus",
                )
                if not exec_result:
                    return False
                response = json.loads(exec_result)
                results = response.get("data", {}).get("result", [])
                return any(r.get("value", [None, "0"])[1] == "1" for r in results)
            except json.JSONDecodeError, KeyError, IndexError, TypeError:
                return False

        try:
            for _ in TimeoutSampler(wait_timeout=90, sleep=5, func=_check_metric_up):
                if _check_metric_up():
                    LOGGER.info("E2E happy path successful: EvalHub → ServiceMonitor → Prometheus → queryable")
                    break
        except TimeoutExpiredError as err:
            raise AssertionError("Prometheus did not scrape EvalHub within 90 seconds") from err


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-e2e-network-policy"},
        ),
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.model_explainability
class TestEvalHubE2ENetworkPolicy:
    """E2E: ServiceMonitor and NetworkPolicy enable scraping in default-deny namespace (RHAISTRAT-1507)."""

    def test_servicemonitor_and_networkpolicy_enable_scraping(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """TC-E2E-002: Validate ServiceMonitor and NetworkPolicy created atomically for default-deny namespace."""
        with (
            NetworkPolicy(
                client=admin_client,
                name="default-deny-ingress",
                namespace=model_namespace.name,
                pod_selector={},
                policy_types=["Ingress"],
                ingress=[],
                wait_for_resource=True,
            ),
            EvalHub(
                client=admin_client,
                name="evalhub-netpol-test",
                namespace=model_namespace.name,
                database={"type": "sqlite"},
                wait_for_resource=True,
            ) as evalhub_cr,
        ):
            deployment = Deployment(
                client=admin_client,
                name=evalhub_cr.name,
                namespace=model_namespace.name,
            )
            deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)

            time.sleep(10)

            service_monitors = list(
                ServiceMonitor.get(
                    client=admin_client,
                    namespace=model_namespace.name,
                    label_selector=f"app={EVALHUB_APP_LABEL}",
                )
            )
            assert any(evalhub_cr.name in sm.name for sm in service_monitors)

            network_policies = list(NetworkPolicy.get(client=admin_client, namespace=model_namespace.name))
            evalhub_netpol = next(
                (np for np in network_policies if "evalhub" in np.name.lower() and np.name != "default-deny-ingress"),
                None,
            )
            assert evalhub_netpol is not None, "NetworkPolicy not created for EvalHub metrics scraping"

            ingress_rules = evalhub_netpol.instance.spec.get("ingress", [])
            assert ingress_rules, f"NetworkPolicy '{evalhub_netpol.name}' has no ingress rules"

            LOGGER.info("E2E NetworkPolicy test successful: Both ServiceMonitor and NetworkPolicy created")


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-e2e-cleanup"},
        ),
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.model_explainability
class TestEvalHubE2ECleanup:
    """E2E: Deploy EvalHub → delete EvalHub → ServiceMonitor cleaned up (RHAISTRAT-1507)."""

    def test_evalhub_deletion_triggers_servicemonitor_cleanup(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """TC-E2E-003: Validate complete lifecycle with automatic cleanup via garbage collection."""
        evalhub_name = "evalhub-cleanup-test"
        with EvalHub(
            client=admin_client,
            name=evalhub_name,
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            wait_for_resource=True,
        ) as evalhub_cr:
            deployment = Deployment(
                client=admin_client,
                name=evalhub_cr.name,
                namespace=model_namespace.name,
            )
            deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)

            service_monitors = list(
                ServiceMonitor.get(
                    client=admin_client,
                    namespace=model_namespace.name,
                    label_selector=f"app={EVALHUB_APP_LABEL}",
                )
            )
            evalhub_sm = next((sm for sm in service_monitors if evalhub_name in sm.name), None)
            assert evalhub_sm is not None, f"ServiceMonitor not created for EvalHub '{evalhub_name}'"
            sm_name = evalhub_sm.name

        # EvalHub deleted, wait for ServiceMonitor GC
        try:
            for _ in TimeoutSampler(
                wait_timeout=60,
                sleep=2,
                func=lambda: (
                    not ServiceMonitor(
                        client=admin_client,
                        name=sm_name,
                        namespace=model_namespace.name,
                    ).exists
                ),
            ):
                if not ServiceMonitor(client=admin_client, name=sm_name, namespace=model_namespace.name).exists:
                    LOGGER.info(f"E2E cleanup successful: ServiceMonitor '{sm_name}' removed via garbage collection")
                    break
        except TimeoutExpiredError as err:
            raise AssertionError(f"ServiceMonitor '{sm_name}' not GC'd within 60 seconds") from err
