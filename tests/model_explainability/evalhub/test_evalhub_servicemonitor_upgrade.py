"""EvalHub ServiceMonitor upgrade scenario tests.

Test Plan Reference: RHAISTRAT-1507
Test Cases: TC-UPGRADE-001 (pre/post), TC-UPGRADE-002, TC-UPGRADE-003
"""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.service_monitor import ServiceMonitor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import EVALHUB_APP_LABEL
from utilities.constants import Timeout

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-servicemonitor-upgrade"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.model_explainability
class TestEvalHubServiceMonitorUpgrade:
    """Upgrade scenario tests for EvalHub ServiceMonitor (RHAISTRAT-1507)."""

    @pytest.mark.pre_upgrade
    def test_existing_evalhub_before_upgrade(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """TC-UPGRADE-001 (Pre): Deploy EvalHub before operator upgrade.

        This is the pre-upgrade step that deploys an EvalHub instance before the
        operator is upgraded to the version with ServiceMonitor support.

        Expected Results:
        - EvalHub instance deploys successfully with pre-upgrade operator
        - No ServiceMonitor exists (pre-upgrade operator version)
        """
        # Deploy EvalHub with metrics enabled
        with EvalHub(
            client=admin_client,
            name="evalhub-upgrade-test",
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            wait_for_resource=True,
        ) as evalhub_upgrade:
            deployment = Deployment(
                client=admin_client,
                name=evalhub_upgrade.name,
                namespace=model_namespace.name,
            )
            deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)

            # Verify no ServiceMonitor exists (pre-upgrade)
            service_monitors = list(
                ServiceMonitor.get(
                    client=admin_client,
                    namespace=model_namespace.name,
                    label_selector=f"app={EVALHUB_APP_LABEL}",
                )
            )
            evalhub_service_monitors = [sm for sm in service_monitors if evalhub_upgrade.name in sm.name]

            assert not evalhub_service_monitors, (
                f"Unexpected ServiceMonitor found before operator upgrade: "
                f"{[sm.name for sm in evalhub_service_monitors]}. "
                f"Pre-upgrade operator should not create ServiceMonitors."
            )

    @pytest.mark.post_upgrade
    def test_existing_evalhub_gets_servicemonitor_after_upgrade(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """TC-UPGRADE-001 (Post): Verify existing EvalHub gets ServiceMonitor after upgrade.

        After upgrading the TrustyAI operator to the version with ServiceMonitor support,
        existing EvalHub instances should automatically receive a ServiceMonitor resource
        during the next reconciliation cycle.

        Expected Results:
        - ServiceMonitor is created for the existing EvalHub instance
        - ServiceMonitor has correct configuration (HTTPS, /metrics, 30s interval)
        - ServiceMonitor has ownerReference pointing to the EvalHub CR
        - Prometheus discovers the new scrape target
        """
        evalhub_name = "evalhub-upgrade-test"

        # Wait for operator reconciliation to process existing EvalHub (up to 120 seconds)
        try:
            for _ in TimeoutSampler(
                wait_timeout=120,
                sleep=5,
                func=lambda: (
                    len(
                        list(
                            ServiceMonitor.get(
                                client=admin_client,
                                namespace=model_namespace.name,
                                label_selector=f"app={EVALHUB_APP_LABEL}",
                            )
                        )
                    )
                    > 0
                ),
            ):
                service_monitors = list(
                    ServiceMonitor.get(
                        client=admin_client,
                        namespace=model_namespace.name,
                        label_selector=f"app={EVALHUB_APP_LABEL}",
                    )
                )
                evalhub_service_monitors = [sm for sm in service_monitors if evalhub_name in sm.name]

                if evalhub_service_monitors:
                    service_monitor = evalhub_service_monitors[0]

                    # Verify ServiceMonitor configuration
                    endpoints = service_monitor.instance.spec.endpoints
                    assert endpoints, f"ServiceMonitor '{service_monitor.name}' has no endpoints"

                    endpoint = endpoints[0]
                    assert endpoint.port == "https", f"Expected endpoint port 'https', got '{endpoint.port}'"
                    assert endpoint.path == "/metrics", f"Expected endpoint path '/metrics', got '{endpoint.path}'"
                    assert endpoint.interval == "30s", f"Expected scrape interval '30s', got '{endpoint.interval}'"
                    assert endpoint.scheme == "https", f"Expected endpoint scheme 'https', got '{endpoint.scheme}'"

                    # Verify ownerReference
                    owner_refs = service_monitor.instance.metadata.ownerReferences
                    assert owner_refs, f"ServiceMonitor '{service_monitor.name}' has no ownerReferences"

                    evalhub_owner_ref = next(
                        (ref for ref in owner_refs if ref.kind == "EvalHub"),
                        None,
                    )
                    assert evalhub_owner_ref is not None, "ServiceMonitor missing EvalHub ownerReference"
                    assert evalhub_owner_ref.name == evalhub_name, (
                        f"Expected ownerReference name '{evalhub_name}', got '{evalhub_owner_ref.name}'"
                    )

                    break

        except TimeoutExpiredError as err:
            msg = (
                f"ServiceMonitor was not created for existing EvalHub '{evalhub_name}' "
                f"within 120 seconds after operator upgrade. Expected operator to reconcile "
                f"and create ServiceMonitor for pre-existing EvalHub instances."
            )
            raise AssertionError(msg) from err
