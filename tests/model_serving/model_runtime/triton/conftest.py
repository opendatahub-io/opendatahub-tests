from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.triton.basic_model_deployment.utils import get_template_name
from tests.model_serving.model_runtime.triton.constant import RUNTIME_MAP
from utilities.constants import KServeDeploymentType
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def triton_pvc_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    protocol: str,
    supported_accelerator_type: str | None,
) -> Generator[ServingRuntime, Any, Any]:
    """Triton ServingRuntime fixture for PVC tests."""
    template_name = get_template_name(protocol=protocol, accelerator_type=supported_accelerator_type)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=RUNTIME_MAP.get(protocol, "triton-runtime"),
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
    ) as model_runtime:
        yield model_runtime
