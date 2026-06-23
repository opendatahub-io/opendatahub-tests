import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_APP_LABEL,
    EVALHUB_MCP_COMPONENT_LABEL,
    EVALHUB_MCP_CONFIGMAP_SUFFIX,
    EVALHUB_MCP_CONTAINER_NAME,
    EVALHUB_MCP_CR_NAME,
    EVALHUB_MCP_KUBE_RBAC_PROXY_CONTAINER,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-deployment"},
        ),
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.ai_safety
class TestEvalHubMcpDeployment:
    """Tests for EvalHub MCP deployment topology on OpenShift."""

    def test_evalhub_mcp_pod_has_expected_containers(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_mcp_mt_deployment: Deployment,
    ) -> None:
        """
        Given: EvalHub CR has MCP enabled and deployment is ready
        When: MCP pod containers are inspected
        Then: Pod runs evalhub-mcp and kube-rbac-proxy containers
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=(
                    f"app={EVALHUB_MCP_APP_LABEL},"
                    f"instance={EVALHUB_MCP_CR_NAME},"
                    f"component={EVALHUB_MCP_COMPONENT_LABEL}"
                ),
            )
        )
        assert len(pods) == 1, f"Expected 1 MCP pod, found {len(pods)}"

        container_names = [container.name for container in pods[0].instance.spec.containers]
        assert EVALHUB_MCP_CONTAINER_NAME in container_names, (
            f"Expected '{EVALHUB_MCP_CONTAINER_NAME}' container, got {container_names}"
        )
        assert EVALHUB_MCP_KUBE_RBAC_PROXY_CONTAINER in container_names, (
            f"Expected '{EVALHUB_MCP_KUBE_RBAC_PROXY_CONTAINER}' sidecar, got {container_names}"
        )

    def test_evalhub_mcp_pod_labels_match_operator(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_mcp_mt_deployment: Deployment,
    ) -> None:
        """
        Given: EvalHub operator has provisioned the MCP deployment
        When: MCP pod labels are inspected
        Then: Labels match operator conventions for app, instance, and component
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"component={EVALHUB_MCP_COMPONENT_LABEL},instance={EVALHUB_MCP_CR_NAME}",
            )
        )
        assert pods, "No MCP pods found"
        labels = pods[0].instance.metadata.labels
        assert labels.get("app") == EVALHUB_MCP_APP_LABEL
        assert labels.get("instance") == EVALHUB_MCP_CR_NAME
        assert labels.get("component") == EVALHUB_MCP_COMPONENT_LABEL

    def test_evalhub_mcp_configmap_mounted_in_pod(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_mcp_mt_deployment: Deployment,
    ) -> None:
        """
        Given: Operator-generated MCP config ConfigMap exists
        When: MCP pod volumes are inspected
        Then: Pod mounts the MCP configuration volume
        """
        configmap_name = f"{EVALHUB_MCP_CR_NAME}{EVALHUB_MCP_CONFIGMAP_SUFFIX}"
        configmap = ConfigMap(
            client=admin_client,
            name=configmap_name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        assert configmap.exists

        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"component={EVALHUB_MCP_COMPONENT_LABEL},instance={EVALHUB_MCP_CR_NAME}",
            )
        )
        assert pods, "No MCP pods found"
        volume_names = [volume.name for volume in pods[0].instance.spec.volumes or []]
        assert any("mcp" in name.lower() or "config" in name.lower() for name in volume_names), (
            f"Expected MCP config volume in pod, got volumes: {volume_names}"
        )
