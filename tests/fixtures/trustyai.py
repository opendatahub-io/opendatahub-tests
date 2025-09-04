import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment

from typing import Generator, Any

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import py_config

from utilities.constants import Annotations, TRUSTYAI_SERVICE_NAME


@pytest.fixture(scope="class")
def trustyai_operator_deployment(admin_client: DynamicClient) -> Deployment:
    return Deployment(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="function")
def patched_trustyai_configmap_allow_online(
    admin_client: DynamicClient, trustyai_operator_deployment: Deployment
) -> Generator[ConfigMap, Any, Any]:
    """
    Patches the TrustyAI Operator ConfigMap in order to set allowOnline and allowCodeExecution to true.
    These options are needed to run some LMEval tasks, which rely on having access to the internet
    and running arbitrary code. The deployment needs to be restarted in order for these changes to be applied.
    """
    trustyai_service_operator: str = "trustyai-service-operator"

    configmap: ConfigMap = ConfigMap(
        client=admin_client,
        name=f"{trustyai_service_operator}-config",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
    with ResourceEditor(
        patches={
            configmap: {
                "metadata": {"annotations": {Annotations.OpenDataHubIo.MANAGED: "false"}},
                "data": {
                    "lmes-allow-online": "true",
                    "lmes-allow-code-execution": "true",
                },
            }
        }
    ):
        num_replicas: int = trustyai_operator_deployment.replicas
        trustyai_operator_deployment.scale_replicas(replica_count=0)
        trustyai_operator_deployment.scale_replicas(replica_count=num_replicas)
        trustyai_operator_deployment.wait_for_replicas()
        yield configmap
