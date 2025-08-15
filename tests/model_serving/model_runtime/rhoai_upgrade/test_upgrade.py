import uuid

import pytest
from pytest_testconfig import config as py_config
from kubernetes.dynamic import DynamicClient
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.rhoai_upgrade.constant import (
    SERVING_RUNTIME_INSTANCE_NAME,
    SERVING_RUNTIME_TEMPLATE_NAME,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = get_logger(name=__name__)


class TestServingRuntimeLifecycle:
    """
    Tests to validate the lifecycle of ServingRuntime resources
    including creation, verification, and deletion.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.order(1)
    def test_serving_runtime_template_lifecycle(self, serving_runtime_template: Template) -> None:
        assert serving_runtime_template.exists, (
            f"Failed to validate Serving Runtime template lifecycle'{SERVING_RUNTIME_TEMPLATE_NAME}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.order(2)
    def test_serving_runtime_instance_lifecycle(self, serving_runtime_instance: ServingRuntime) -> None:
        assert serving_runtime_instance.exists, (
            f"Failed to validate Serving Runtime instance lifecycle'{SERVING_RUNTIME_INSTANCE_NAME}'"
        )

    @pytest.mark.post_upgrade
    def test_serving_runtime_instance_lifecycle_from_existing_cluster_templates(
        self, admin_client: DynamicClient
    ) -> None:
        serving_runtime_templates = Template.get(
            client=admin_client,
            namespace=py_config["applications_namespace"],
            singular_name="template",
        )
        serving_runtime_template_names = [template.name for template in serving_runtime_templates]
        assert serving_runtime_template_names, "No ServingRuntime templates found in cluster"

        LOGGER.info(f"Available Serving Runtime templates: {serving_runtime_template_names}")

        for serving_runtime_template_name in serving_runtime_template_names:
            instance_name = f"test-runtime-{uuid.uuid4().hex[:6]}"

            serving_runtime_instance = ServingRuntimeFromTemplate(
                client=admin_client,
                name=instance_name,
                namespace=py_config["applications_namespace"],
                template_name=serving_runtime_template_name,
            )

            try:
                serving_runtime_instance.deploy()
                assert serving_runtime_instance.exists, (
                    f"Failed to create Serving Runtime instance '{instance_name}'"
                    f" for template '{serving_runtime_template_name}'"
                )

            finally:
                serving_runtime_instance.clean_up(wait=True)
                assert not serving_runtime_instance.exists, (
                    f"ServingRuntime '{instance_name}' was not deleted for template '{serving_runtime_template_name}'"
                )
