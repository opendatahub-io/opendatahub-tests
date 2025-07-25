import uuid
import pytest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.rhoai_upgrade.constant import (
    SERVING_RUNTIME_TEMPLATE_FILE_PATH,
    SERVING_RUNTIME_TEMPLATE_NAME,
    SERVING_RUNTIME_INSTANCE_NAME,
)

from tests.model_serving.model_runtime.rhoai_upgrade.utils import (
    create_serving_runtime_template,
    create_serving_runtime_instance,
    delete_serving_runtime_template,
    delete_serving_runtime_instance,
    get_serving_runtime_template,
    get_serving_runtime_instance,
    get_serving_runtime_template_name_list,
)

LOGGER = get_logger(name=__name__)


class TestServingRuntimeLifecycle:
    @pytest.mark.post_upgrade
    @pytest.mark.order(1)
    def test_create_and_validate_serving_runtime_template(self, admin_client: DynamicClient) -> None:
        """
        Create and verify the serving runtime template is created.
        """

        serving_runtime_template = create_serving_runtime_template(
            client=admin_client, yaml_file=SERVING_RUNTIME_TEMPLATE_FILE_PATH
        )
        assert serving_runtime_template.exists, "Serving Runtime Template was not created successfully"

    @pytest.mark.post_upgrade
    @pytest.mark.order(2)
    def test_create_and_validate_serving_runtime_instance(self, admin_client: DynamicClient) -> None:
        """
        Create and verify the serving runtime instance is created.
        """

        serving_runtime_instance = create_serving_runtime_instance(
            client=admin_client,
            serving_runtime_template_name=SERVING_RUNTIME_TEMPLATE_NAME,
            serving_runtime_instance_name=SERVING_RUNTIME_INSTANCE_NAME,
        )
        assert serving_runtime_instance.exists, (
            f"Failed to create Serving Runtime instance '{SERVING_RUNTIME_INSTANCE_NAME}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.order(3)
    def test_delete_and_validate_serving_runtime_instance(self, admin_client: DynamicClient) -> None:
        """
        Delete and verify the serving runtime instance is removed.
        """

        serving_runtime_instance = get_serving_runtime_instance(
            client=admin_client, serving_runtime_instance_name=SERVING_RUNTIME_INSTANCE_NAME
        )
        delete_serving_runtime_instance(serving_runtime_instance=serving_runtime_instance)

        serving_runtime_instance = get_serving_runtime_instance(
            client=admin_client, serving_runtime_instance_name=SERVING_RUNTIME_INSTANCE_NAME
        )
        assert not serving_runtime_instance.exists, f"ServingRuntime '{SERVING_RUNTIME_INSTANCE_NAME}' was not deleted"

    @pytest.mark.post_upgrade
    @pytest.mark.order(4)
    def test_delete_and_validate_serving_runtime_template(self, admin_client: DynamicClient) -> None:
        """
        Delete and verify the serving runtime template is removed.
        """

        serving_runtime_template = get_serving_runtime_template(
            client=admin_client, serving_runtime_template_name=SERVING_RUNTIME_TEMPLATE_NAME
        )
        delete_serving_runtime_template(serving_runtime_template=serving_runtime_template)

        serving_runtime_template = get_serving_runtime_template(
            client=admin_client, serving_runtime_template_name=SERVING_RUNTIME_TEMPLATE_NAME
        )
        assert not serving_runtime_template.exists, f"ServingRuntime '{SERVING_RUNTIME_TEMPLATE_NAME}' was not deleted"

    @pytest.mark.post_upgrade
    def test_serving_runtime_instance_creation_from_existing_cluster_templates(
        self, admin_client: DynamicClient
    ) -> None:
        """
        Discover existing templates from the cluster, find one with a ServingRuntime,
        and create/delete a runtime instance from it.
        """

        serving_runtime_template_name_list = get_serving_runtime_template_name_list(client=admin_client)

        LOGGER.debug(f"Available Serving Runtime templates are {serving_runtime_template_name_list}")

        for serving_runtime_template_name in serving_runtime_template_name_list:
            serving_runtime_instance_name = f"test-runtime-{uuid.uuid4().hex[:6]}"

            serving_runtime_instance = create_serving_runtime_instance(
                client=admin_client,
                serving_runtime_template_name=serving_runtime_template_name,
                serving_runtime_instance_name=serving_runtime_instance_name,
            )
            assert serving_runtime_instance.exists, (
                f"Failed to create Serving Runtime instance '{serving_runtime_instance_name}' "
                f"for template '{serving_runtime_template_name}'"
            )

            delete_serving_runtime_instance(serving_runtime_instance=serving_runtime_instance)
            serving_runtime_instance = get_serving_runtime_instance(
                client=admin_client, serving_runtime_instance_name=serving_runtime_instance_name
            )
            assert not serving_runtime_instance.exists, (
                f"ServingRuntime '{serving_runtime_instance_name}' was not deleted "
                f"for template '{serving_runtime_template_name}'"
            )
