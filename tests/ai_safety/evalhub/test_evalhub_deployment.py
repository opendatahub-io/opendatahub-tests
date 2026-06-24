import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from tests.ai_safety.evalhub.constants import (
    EVALHUB_API_GROUP,
    EVALHUB_APP_LABEL,
    EVALHUB_COMPONENT_LABEL,
    EVALHUB_CONTAINER_NAME,
    EVALHUB_KUBE_RBAC_PROXY_CONTAINER,
    EVALHUB_PLURAL,
    EVALHUB_SERVICE_NAME,
)


@pytest.mark.smoke
@pytest.mark.ai_safety
def test_evalhub_crd_exists(
    admin_client: DynamicClient,
) -> None:
    """Verify EvalHub CRD exists on the cluster."""
    crd_name = f"{EVALHUB_PLURAL}.{EVALHUB_API_GROUP}"

    crd_resource = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
        ensure_exists=True,
    )

    assert crd_resource.exists, f"CRD {crd_name} does not exist on the cluster"


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-deployment"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubDeployment:
    """Tests for EvalHub deployment topology (pods, containers, labels)."""

    def test_evalhub_pod_has_expected_containers(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_deployment: Deployment,
    ) -> None:
        """
        Given: EvalHub CR is deployed and the deployment is ready.
        When: The pod containers are inspected.
        Then: Exactly one pod runs with both evalhub and kube-rbac-proxy containers present.
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={EVALHUB_APP_LABEL},component={EVALHUB_COMPONENT_LABEL}",
            )
        )

        assert len(pods) == 1, f"Expected 1 EvalHub pod, found {len(pods)}"

        pod = pods[0]
        containers = pod.instance.spec.containers
        container_names = [container.name for container in containers]
        assert len(containers) == 2, f"Expected 2 containers in EvalHub pod, found {len(containers)}: {container_names}"
        assert EVALHUB_CONTAINER_NAME in container_names, (
            f"Expected '{EVALHUB_CONTAINER_NAME}' container, got {container_names}"
        )
        assert EVALHUB_KUBE_RBAC_PROXY_CONTAINER in container_names, (
            f"Expected '{EVALHUB_KUBE_RBAC_PROXY_CONTAINER}' container, got {container_names}"
        )

        # Verify pod labels match what the operator sets in deployment.go lines 64-68
        pod_labels = pod.instance.metadata.labels
        expected_labels = {
            "app": EVALHUB_APP_LABEL,
            "instance": EVALHUB_SERVICE_NAME,
            "component": EVALHUB_COMPONENT_LABEL,
        }
        for key, expected_value in expected_labels.items():
            actual_value = pod_labels.get(key)
            assert actual_value == expected_value, (
                f"Expected label '{key}={expected_value}', got '{key}={actual_value}'"
            )
