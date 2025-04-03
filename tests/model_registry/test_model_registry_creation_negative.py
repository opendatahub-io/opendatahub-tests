import pytest
from typing import Self
from simple_logger.logger import get_logger

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.model_registry import ModelRegistry
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from utilities.constants import DscComponents, Annotations
from tests.model_registry.constants import (
    MR_NAMESPACE,
    MR_OPERATOR_NAME,
    MR_INSTANCE_NAME,
    ISTIO_CONFIG_DICT,
    DB_RESOURCES_NAME,
)
from kubernetes.dynamic.exceptions import UnprocessibleEntityError

LOGGER = get_logger(name=__name__)
CUSTOM_NEGATIVE_NS = "model-registry-negative-ns"


@pytest.mark.parametrize(
    "model_registry_namespace, updated_dsc_component_state_scope_class, expected_namespace",
    [
        pytest.param(
            {"namespace_name": CUSTOM_NEGATIVE_NS},
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": MR_NAMESPACE,
                    },
                }
            },
            MR_NAMESPACE,
        ),
        pytest.param(
            {"namespace_name": MR_NAMESPACE},
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": CUSTOM_NEGATIVE_NS,
                    },
                },
            },
            CUSTOM_NEGATIVE_NS,
        ),
    ],
    indirect=["model_registry_namespace", "updated_dsc_component_state_scope_class"],
)
class TestModelRegistryCreationNegative:
    def test_registering_model_negative(
        self: Self,
        current_client_token: str,
        model_registry_namespace: Namespace,
        updated_dsc_component_state_scope_class: DataScienceCluster,
        model_registry_db_deployment: Deployment,
        model_registry_db_secret: Secret,
        expected_namespace: str,
    ):
        my_sql_dict: dict[str, str] = {
            "host": f"{model_registry_db_deployment.name}.{model_registry_db_deployment.namespace}.svc.cluster.local",
            "database": model_registry_db_secret.string_data["database-name"],
            "passwordSecret": {"key": "database-password", "name": DB_RESOURCES_NAME},
            "port": 3306,
            "skipDBCreation": False,
            "username": model_registry_db_secret.string_data["database-user"],
        }
        with pytest.raises(
            UnprocessibleEntityError,
            match=f"namespace must be {expected_namespace}",
        ):
            with ModelRegistry(
                name=MR_INSTANCE_NAME,
                namespace=model_registry_namespace.name,
                label={
                    Annotations.KubernetesIo.NAME: MR_INSTANCE_NAME,
                    Annotations.KubernetesIo.INSTANCE: MR_INSTANCE_NAME,
                    Annotations.KubernetesIo.PART_OF: MR_OPERATOR_NAME,
                    Annotations.KubernetesIo.CREATED_BY: MR_OPERATOR_NAME,
                },
                grpc={},
                rest={},
                istio=ISTIO_CONFIG_DICT,
                mysql=my_sql_dict,
                wait_for_resource=True,
            ):
                return
