from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.template import Template
from pytest_testconfig import config as py_config

from tests.model_serving.model_runtime.rhoai_upgrade.constant import (
    OVMS_SERVING_RUNTIME_TEMPLATE_DICT,
    SERVING_RUNTIME_INSTANCE_NAME,
    SERVING_RUNTIME_TEMPLATE_NAME,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def serving_runtime_template(admin_client: DynamicClient) -> Generator[Any, Any, Any]:
    """
    Class-scoped fixture that deploys a ServingRuntime Template and cleans it up after tests.
    """
    with Template(
        client=admin_client,
        namespace=py_config["applications_namespace"],
        kind_dict=OVMS_SERVING_RUNTIME_TEMPLATE_DICT,
    ) as template:
        yield template


@pytest.fixture(scope="class")
def serving_runtime_instance(admin_client: DynamicClient) -> Generator[Any, Any, Any]:
    """
    Class-scoped fixture that deploys a ServingRuntime from a template
    and cleans it up after all tests in the class.
    """
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=SERVING_RUNTIME_INSTANCE_NAME,
        namespace=py_config["applications_namespace"],
        template_name=SERVING_RUNTIME_TEMPLATE_NAME,
    ) as instance:
        yield instance
