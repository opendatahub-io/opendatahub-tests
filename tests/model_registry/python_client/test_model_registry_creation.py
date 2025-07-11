import pytest
from typing import Self, Any
from simple_logger.logger import get_logger
from pytest_testconfig import config as py_config

from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace
from tests.model_registry.utils import (
    execute_model_registry_get_command,
    validate_no_grpc_container,
    validate_mlmd_removal_in_model_registry_pod_log,
)
from utilities.constants import DscComponents
from tests.model_registry.constants import MODEL_NAME, MODEL_DICT
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel

LOGGER = get_logger(name=__name__)

CUSTOM_NAMESPACE = "model-registry-custom-ns"


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class, registered_model",
    [
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": CUSTOM_NAMESPACE,
                    },
                }
            },
            MODEL_DICT,
        ),
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": py_config["model_registry_namespace"],
                    },
                },
            },
            MODEL_DICT,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "registered_model")
class TestModelRegistryCreation:
    """
    Tests the creation of a model registry. If the component is set to 'Removed' it will be switched to 'Managed'
    for the duration of this test module.
    """

    @pytest.mark.smoke
    def test_registering_model(
        self: Self,
        model_registry_client: ModelRegistryClient,
        registered_model: RegisteredModel,
    ):
        model = model_registry_client.get_registered_model(name=MODEL_NAME)
        expected_attrs = {
            "id": registered_model.id,
            "name": registered_model.name,
            "description": registered_model.description,
            "owner": registered_model.owner,
            "state": registered_model.state,
        }
        errors = [
            f"Unexpected {attr} expected: {expected}, received {getattr(model, attr)}"
            for attr, expected in expected_attrs.items()
            if getattr(model, attr) != expected
        ]
        if errors:
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))

    def test_model_registry_operator_env(
        self,
        updated_dsc_component_state_scope_class: Namespace,
        model_registry_namespace: str,
        model_registry_operator_pod: Pod,
    ):
        namespace_env = []
        for container in model_registry_operator_pod.instance.spec.containers:
            for env in container.env:
                if env.name == "REGISTRIES_NAMESPACE" and env.value == model_registry_namespace:
                    namespace_env.append({container.name: env})
        if not namespace_env:
            pytest.fail("Missing environment variable REGISTRIES_NAMESPACE")

    def test_model_registry_grpc_container_removal(self, model_registry_deployment_containers: list[dict[str, Any]]):
        """
        RHOAIENG-26239: Test to ensure removal of grpc container from model registry deployment
        Steps:
            Create metadata database
            Deploys model registry using the same
            Check model registry deployment for grpc container. It should not be present
        """
        validate_no_grpc_container(deployment_containers=model_registry_deployment_containers)

    def test_model_registry_pod_log_mlmd_removal(
        self, model_registry_deployment_containers: list[dict[str, Any]], model_registry_pod: Pod
    ):
        """
        RHOAIENG-26239: Test to ensure removal of grpc container from model registry deployment
        Steps:
            Create metadata database
            Deploys model registry using the same
            Check model registry deployment for grpc container. It should not be present
        """
        validate_mlmd_removal_in_model_registry_pod_log(
            deployment_containers=model_registry_deployment_containers, pod_object=model_registry_pod
        )

    @pytest.mark.parametrize(
        "endpoint",
        [
            pytest.param(
                "oauth/healthz",
            ),
            pytest.param(
                "readyz/isDirty",
            ),
        ],
    )
    def test_model_registry_endpoint_response(
        self, model_registry_rest_url: str, model_registry_rest_headers: dict[str, str], endpoint: str
    ):
        """
        RHOAIENG-26239: Test to ensure model registry endpoints are responsive
        Steps:
            Create metadata database
            Deploys model registry using the same
            Ensure endpoint is responsive via get call
        """
        output = execute_model_registry_get_command(
            url=f"{model_registry_rest_url}/{endpoint}", headers=model_registry_rest_headers, json_output=False
        )
        assert output["raw_output"].lower() == "OK".lower()
