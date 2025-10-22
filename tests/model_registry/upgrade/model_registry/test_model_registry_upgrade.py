import pytest
from typing import Self, Any

from kubernetes.dynamic import DynamicClient

from ocp_resources.pod import Pod
from tests.model_registry.constants import MODEL_NAME, MODEL_DICT
from model_registry.types import RegisteredModel
from model_registry import ModelRegistry as ModelRegistryClient
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from simple_logger.logger import get_logger
from tests.model_registry.utils import (
    get_and_validate_registered_model,
    validate_no_grpc_container,
    validate_mlmd_removal_in_model_registry_pod_log,
)

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "registered_model",
    [
        pytest.param(
            MODEL_DICT,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("model_registry_metadata_db_resources", "model_registry_instance")
class TestPreUpgradeModelRegistry:
    @pytest.mark.pre_upgrade
    def test_registering_model_pre_upgrade(
        self: Self,
        model_registry_client: list[ModelRegistryClient],
        registered_model: RegisteredModel,
    ):
        errors = get_and_validate_registered_model(
            model_registry_client=model_registry_client[0], model_name=MODEL_NAME, registered_model=registered_model
        )
        if errors:
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))


class TestPostUpgradeModelRegistry:
    @pytest.mark.post_upgrade
    def test_retrieving_model_post_upgrade(
        self: Self,
        model_registry_client: list[ModelRegistryClient],
        model_registry_instance,
    ):
        errors = get_and_validate_registered_model(
            model_registry_client=model_registry_client[0],
            model_name=MODEL_NAME,
        )
        if errors:
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))

    @pytest.mark.post_upgrade
    def test_model_registry_instance_api_version_post_upgrade(
        self: Self,
        model_registry_instance,
    ):
        # the following is valid for 2.22+
        api_version = model_registry_instance[0].instance.apiVersion
        expected_version = f"{ModelRegistry.ApiGroup.MODELREGISTRY_OPENDATAHUB_IO}/{ModelRegistry.ApiVersion.V1BETA1}"
        assert api_version == expected_version

    @pytest.mark.post_upgrade
    def test_model_registry_instance_spec_post_upgrade(
        self: Self,
        model_registry_instance,
    ):
        model_registry_instance_spec = model_registry_instance[0].instance.spec
        assert not model_registry_instance_spec.istio
        assert model_registry_instance_spec.kubeRBACProxy.serviceRoute == "enabled"

    @pytest.mark.post_upgrade
    def test_model_registry_grpc_container_removal_post_upgrade(
        self, model_registry_deployment_containers: list[dict[str, Any]]
    ):
        """
        RHOAIENG-29161: Test to ensure removal of grpc container from model registry deployment post upgrade
        Steps:
            Check model registry deployment for grpc container. It should not be present
        """
        validate_no_grpc_container(deployment_containers=model_registry_deployment_containers)

    @pytest.mark.post_upgrade
    def test_model_registry_pod_log_mlmd_removal(
        self, model_registry_deployment_containers: list[dict[str, Any]], model_registry_pod: Pod
    ):
        """
        RHOAIENG-29161: Test to ensure removal of grpc container from model registry deployment
        Steps:
            Create metadata database
            Deploys model registry using the same
            Check model registry deployment for grpc container. It should not be present
        """
        validate_mlmd_removal_in_model_registry_pod_log(
            deployment_containers=model_registry_deployment_containers, pod_object=model_registry_pod
        )

    @pytest.mark.post_upgrade
    def test_model_registry_storage_version(self, admin_client: DynamicClient):
        """
        RHOAIENG-28213: Test to ensure v1beta1 is found in crd storedVersion
        Steps:
            After upgrade check if the storedVersion for CRD contains v1beta1
        """
        mr_crd = CustomResourceDefinition(name="modelregistries.modelregistry.opendatahub.io")
        assert mr_crd.exists
        expected_stored_version = "v1beta1"
        stored_version = mr_crd.instance.status.storedVersions
        assert expected_stored_version in stored_version, f"Expected {expected_stored_version}, found: {stored_version}"
