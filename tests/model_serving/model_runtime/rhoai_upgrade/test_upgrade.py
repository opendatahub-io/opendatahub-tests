import uuid

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.rhoai_upgrade.constant import (
    APPLICATIONS_NAMESPACE,
    OVMS_SERVING_RUNTIME_TEMPLATE_DICT,
    SERVING_RUNTIME_INSTANCE_NAME,
    SERVING_RUNTIME_TEMPLATE_NAME,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("admin_client")
class TestServingRuntimeLifecycle:
    """
    Tests to validate the lifecycle of ServingRuntime resources
    including creation, verification, and deletion.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.order(1)
    def test_create_and_validate_serving_runtime_template(self, admin_client: DynamicClient) -> None:
        serving_runtime_template = Template(
            client=admin_client,
            namespace=APPLICATIONS_NAMESPACE,
            kind_dict=OVMS_SERVING_RUNTIME_TEMPLATE_DICT,
        )
        serving_runtime_template.deploy()
        assert serving_runtime_template.exists, (
            f"Failed to create Serving Runtime template '{SERVING_RUNTIME_TEMPLATE_NAME}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.order(2)
    def test_create_and_validate_serving_runtime_instance(self, admin_client: DynamicClient) -> None:
        serving_runtime_instance = ServingRuntimeFromTemplate(
            client=admin_client,
            name=SERVING_RUNTIME_INSTANCE_NAME,
            namespace=APPLICATIONS_NAMESPACE,
            template_name=SERVING_RUNTIME_TEMPLATE_NAME,
        )
        serving_runtime_instance.deploy()
        assert serving_runtime_instance.exists, (
            f"Failed to create Serving Runtime instance '{SERVING_RUNTIME_INSTANCE_NAME}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.order(3)
    def test_delete_and_validate_serving_runtime_instance(self, admin_client: DynamicClient) -> None:
        serving_runtime_instance = ServingRuntime(
            client=admin_client,
            name=SERVING_RUNTIME_INSTANCE_NAME,
            namespace=APPLICATIONS_NAMESPACE,
            ensure_exists=True,
        )
        serving_runtime_instance.clean_up()
        assert not serving_runtime_instance.exists, (
            f"ServingRuntime instance '{SERVING_RUNTIME_INSTANCE_NAME}' was not deleted"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.order(4)
    def test_delete_and_validate_serving_runtime_template(self, admin_client: DynamicClient) -> None:
        serving_runtime_template = Template(
            client=admin_client,
            name=SERVING_RUNTIME_TEMPLATE_NAME,
            namespace=APPLICATIONS_NAMESPACE,
            ensure_exists=True,
        )
        serving_runtime_template.clean_up()
        assert not serving_runtime_template.exists, (
            f"ServingRuntime template '{SERVING_RUNTIME_TEMPLATE_NAME}' was not deleted"
        )

    @pytest.mark.post_upgrade
    def test_serving_runtime_instance_creation_from_existing_cluster_templates(
        self, admin_client: DynamicClient
    ) -> None:
        serving_runtime_template_name_list = [
            template.name
            for template in Template.get(
                client=admin_client,
                namespace=APPLICATIONS_NAMESPACE,
                singular_name="template",
            )
        ]
        assert len(serving_runtime_template_name_list) > 0, "No ServingRuntime templates found in cluster"

        LOGGER.info(f"Available Serving Runtime templates: {serving_runtime_template_name_list}")

        for template_name in serving_runtime_template_name_list:
            instance_name = f"test-runtime-{uuid.uuid4().hex[:6]}"

            serving_runtime_instance = ServingRuntimeFromTemplate(
                client=admin_client,
                name=instance_name,
                namespace=APPLICATIONS_NAMESPACE,
                template_name=template_name,
            )
            serving_runtime_instance.deploy()
            assert serving_runtime_instance.exists, (
                f"Failed to create Serving Runtime instance '{instance_name}' for template '{template_name}'"
            )

            serving_runtime_instance.clean_up()
            assert not serving_runtime_instance.exists, (
                f"ServingRuntime '{instance_name}' was not deleted for template '{template_name}'"
            )
