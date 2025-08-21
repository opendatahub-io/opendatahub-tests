from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import py_config

from utilities.constants import Annotations


@pytest.fixture(scope="function")
def patched_trustyai_operator_configmap_allow_online(admin_client: DynamicClient) -> Generator[ConfigMap, Any, Any]:
    namespace: str = py_config["applications_namespace"]
    trustyai_service_operator: str = "trustyai-service-operator"

    configmap: ConfigMap = ConfigMap(
        client=admin_client, name=f"{trustyai_service_operator}-config", namespace=namespace, ensure_exists=True
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
        deployment: Deployment = Deployment(
            client=admin_client,
            name=f"{trustyai_service_operator}-controller-manager",
            namespace=namespace,
            ensure_exists=True,
        )
        num_replicas: int = deployment.replicas
        deployment.scale_replicas(replica_count=0)
        deployment.scale_replicas(replica_count=num_replicas)
        deployment.wait_for_replicas()
        yield configmap
