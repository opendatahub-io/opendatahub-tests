import pytest
from pytest_testconfig import config as py_config

from ocp_resources.template import Template
from utilities.serving_runtime import ServingRuntimeFromTemplate
from ocp_resources.serving_runtime import ServingRuntime
from kubernetes.dynamic import DynamicClient
from tests.model_serving.model_runtime.rhoai_upgrade.constant import (
    SERVING_RUNTIME_TEMPLATE_NAME,
    SERVING_RUNTIME_INSTANCE_NAME,
    OVMS_SERVING_RUNTIME_TEMPLATE_DICT,
)


@pytest.fixture(scope="class")
def serving_runtime_template(admin_client: DynamicClient) -> Template:
    """
    Class-scoped fixture that deploys a ServingRuntime Template and cleans it up after tests.
    """
    template = Template(
        client=admin_client,
        namespace=py_config["applications_namespace"],
        kind_dict=OVMS_SERVING_RUNTIME_TEMPLATE_DICT,
    )
    template.deploy()

    yield template

    if template.exists:
        template.clean_up(wait=True)


@pytest.fixture(scope="class")
def serving_runtime_instance(admin_client: DynamicClient) -> ServingRuntime:
    """
    Class-scoped fixture that deploys a ServingRuntime from a template
    and cleans it up after all tests in the class.
    """
    instance = ServingRuntimeFromTemplate(
        client=admin_client,
        name=SERVING_RUNTIME_INSTANCE_NAME,
        namespace=py_config["applications_namespace"],
        template_name=SERVING_RUNTIME_TEMPLATE_NAME,
    )
    instance.deploy()

    yield instance

    if instance.exists:
        instance.clean_up(wait=True)
