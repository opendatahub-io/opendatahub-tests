import pytest
from typing import Self, Any

from ocp_resources.pod import Pod
from tests.model_registry.constants import MODEL_NAME, MODEL_DICT
from model_registry.types import RegisteredModel
from model_registry import ModelRegistry as ModelRegistryClient
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from simple_logger.logger import get_logger
from tests.model_registry.rest_api.utils import ModelRegistryV1Alpha1
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
@pytest.mark.usefixtures("pre_upgrade_dsc_patch")
class TestPreUpgradeModelRegistry:
    @pytest.mark.pre_upgrade
    def test_registering_model_pre_upgrade(
        self: Self,
        model_registry_client: ModelRegistryClient,
        registered_model: RegisteredModel,
    ):
        errors = get_and_validate_registered_model(
            model_registry_client=model_registry_client, model_name=MODEL_NAME, registered_model=registered_model
        )
        if errors:
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))

    # TODO: if we are in <=2.21, we can create a servicemesh MR here instead of oauth (v1alpha1), and then in
    # post-upgrade check that it automatically gets converted to oauth (v1beta1) - to be done in 2.21 branch directly.


@pytest.mark.usefixtures("post_upgrade_dsc_patch")
class TestPostUpgradeModelRegistry:
    @pytest.mark.post_upgrade
    def test_retrieving_model_post_upgrade(
        self: Self,
        model_registry_client: ModelRegistryClient,
        model_registry_instance: ModelRegistry,
    ):
        errors = get_and_validate_registered_model(
            model_registry_client=model_registry_client,
            model_name=MODEL_NAME,
        )
        if errors:
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))

    @pytest.mark.post_upgrade
    def test_model_registry_instance_api_version_post_upgrade(
        self: Self,
        model_registry_instance: ModelRegistry,
    ):
        # the following is valid for 2.22+
        api_version = model_registry_instance.instance.apiVersion
        expected_version = f"{ModelRegistry.ApiGroup.MODELREGISTRY_OPENDATAHUB_IO}/{ModelRegistry.ApiVersion.V1BETA1}"
        assert api_version == expected_version

    @pytest.mark.post_upgrade
    def test_model_registry_instance_spec_post_upgrade(
        self: Self,
        model_registry_instance: ModelRegistry,
    ):
        model_registry_instance_spec = model_registry_instance.instance.spec
        assert not model_registry_instance_spec.istio
        assert model_registry_instance_spec.oauthProxy.serviceRoute == "enabled"

    @pytest.mark.post_upgrade
    def test_model_registry_instance_status_conversion_post_upgrade(
        self: Self,
        model_registry_instance: ModelRegistry,
    ):
        # TODO: After v1alpha1 is removed (2.24?) this has to be removed
        mr_instance = ModelRegistryV1Alpha1(
            name=model_registry_instance.name, namespace=model_registry_instance.namespace, ensure_exists=True
        ).instance
        status = mr_instance.status.to_dict()
        LOGGER.info(f"Validating MR status {status}")
        if not status:
            pytest.fail(f"Empty status found for {mr_instance}")

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
